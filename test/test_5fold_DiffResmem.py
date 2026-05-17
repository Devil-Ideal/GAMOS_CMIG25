import copy
import nibabel as nib
import SimpleITK as sitk
import numpy as np
from dataset.btcv_transunet_datasetings import get_5fold_test_full_loader
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.trainer_mix import Trainer
from monai.utils import set_determinism
from guided_diffusion_ori.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType,LossType
from guided_diffusion_ori.respace import SpacedDiffusion, space_timesteps
from guided_diffusion_ori.resample import UniformSampler
from medpy import metric
import os
import pandas as pd
import argparse
from unet.ResUnet import ResUnet_mem_sparse
import time
set_determinism(123)
max_epoch = 300
batch_size = 2
val_every = 10
num_gpus = 2
device = "cuda"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.01] = 0.01
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out
class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = ResUnet_mem_sparse(res_depth=50, num_classes=4, norm_cfg='IN', activation_cfg='LeakyReLU',
                                        in_channels=5, mem_size=2048, sparse=0.75)
        betas = get_named_beta_schedule("cosine", 50)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(50, [50]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(50, [10]),
                                            betas=betas,
                                            model_mean_type=ModelMeanType.START_X,
                                            model_var_type=ModelVarType.FIXED_LARGE,
                                            loss_type=LossType.MSE,
                                            )
        self.sampler = UniformSampler(50)
    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None,c_idx = None,aug=False):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise
        elif pred_type == "denoise":
            return self.model(image,aug = aug)
        elif pred_type == "ddim_sample":
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 4, 64, 192, 192),
                                                                model_kwargs={"image": image,
                                                                              "c_idx": c_idx})
            sample_return = torch.zeros((1, 4, 64, 192, 192))
            all_samples = sample_out["all_samples"]
            index = 0
            for sample in all_samples:
                sample_return += sample.cpu()
                index += 1
            return sample_return
class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[64, 192, 192],
                                        sw_batch_size=1,
                                        overlap=0.5)
        self.model = DiffUNet()
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.loss_func = nn.CrossEntropyLoss()
    def get_input(self, batch):
        image = batch["image"]
        label = batch["raw_label"]
        label = label.float()
        return image, label
    def convert_labels(self, labels):
        labels_new = []
        for i in range(1, 14):
            labels_new.append(labels == i)
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new
    def validation_step(self, batch):
        image, label = self.get_input(batch)
        idx = [0, 3, 1, 2]
        start_time = time.time()
        output = self.window_infer(image, self.model,  pred_type="ddim_sample", c_idx=None)
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.4f} seconds")
        return
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu()
        d, w, h = label.shape[2], label.shape[3], label.shape[4]
        output = torch.nn.functional.interpolate(output, mode="nearest", size=(d, w, h))
        output = output.numpy()
        target = label.cpu().numpy()
        asds, sens, precs = [], [], []
        c = 4
        for i in range(0, c):
            pred = output[:, i]
            gt = target[:, idx[i]]
            if pred.sum() > 0 and gt.sum() > 0:
                asd = metric.binary.asd(pred, gt)
                tp = (pred * gt).sum()
                fp = ((pred == 1) & (gt == 0)).sum()
                fn = ((pred == 0) & (gt == 1)).sum()
                sensitivity = tp / (tp + fn + 1e-8)
                precision = tp / (tp + fp + 1e-8)
            elif pred.sum() > 0 and gt.sum() == 0:
                asd = 0
                sensitivity = 1
                precision = 1
            else:
                asd = 0
                sensitivity = 0
                precision = 0
            asds.append(asd)
            sens.append(sensitivity)
            precs.append(precision)
        all_m = asds + sens + precs
        print(all_m)
        append_test_results(all_m)
        return all_m
def compute_statistics(probabilities, mask):
    values = probabilities[mask].flatten()
    max_val = values.max()
    min_val = values.min()
    median_val = np.median(values)
    mean_val = values.mean()
    quantiles = np.quantile(values, [0.1 * i for i in range(1, 11)])
    return [max_val, min_val, median_val, mean_val] + quantiles.tolist()
def save_statistics_to_csv(data, filename="statistics.csv", header=True):
    columns = ['fg_max', 'fg_min', 'fg_median', 'fg_mean'] +\
              [f'fg_quantile_{q}' for q in range(10, 110, 10)] +\
              ['bg_max', 'bg_min', 'bg_median', 'bg_mean'] +\
              [f'bg_quantile_{q}' for q in range(10, 110, 10)]
    df = pd.DataFrame(data, columns=columns)
    if os.path.exists(filename):
        header = False
    df.to_csv(filename, mode='a', index=False, header=header)
def append_test_results( result):
    with open(txt_file_path, 'a') as file:
        file.write(f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\t{result[4]}\t{result[5]}\t{result[6]}\t{result[7]}\t{result[8]}\t{result[9]}\t{result[10]}\t{result[11]}\n")
def append_mean_results(result):
    with open(txt_file_path, 'a') as file:
        result = [a.item() for a in result]
        file.write('mean\n')
        file.write(
            f"{result[0]}\t{result[1]}\t{result[2]}\t{result[3]}\t{result[4]}\t{result[5]}\t{result[6]}\t{result[7]}\t{result[8]}\t{result[9]}\t{result[10]}\t{result[11]}\n")
        file.write('end\n')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--fold', type=int,default=0, help='5 fold cross validation')
    parser.add_argument('--logdir', type=str, required=True, help='Path to a trained checkpoint (.pt)')
    parser.add_argument('--toy', action='store_true', help='Use bundled toy layout under data/toy')
    args = parser.parse_args()
    fold = args.fold
    if args.toy:
        os.environ.setdefault("DIFFRESMEM_LAYOUT", "toy")
        os.environ.setdefault("DIFFRESMEM_DATA_ROOT", "./data/toy")
        os.environ.setdefault("DIFFRESMEM_SPLITS_ROOT", "./dataset/splits")
    test1, test2, test3 = get_5fold_test_full_loader(fold)
    trainer = BraTSTrainer(env_type="pytorch",
                                    max_epochs=max_epoch,
                                    batch_size=batch_size,
                                    device=device,
                                    val_every=val_every,
                                    num_gpus=1,
                                    master_port=17751,
                                    training_script=__file__)
    logdir = args.logdir
    directory = os.path.dirname(logdir)
    txt_file_path = os.path.join(f'./checkpoints/5fold_DiffResmem{fold}', f'Diff_test_results{fold}.txt')
    trainer.load_state_dict2(logdir)
    v_mean, v_out = trainer.validation_single_gpu(val_dataset=test1)
    print(f"Spleen:v_mean is {v_mean}")
