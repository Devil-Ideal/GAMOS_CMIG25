from dataset.btcv_transunet_datasetings import get_5fold_loader
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice, hausdorff_distance_95
from light_training.trainer_resume import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
import argparse
from monai.losses.dice import DiceLoss
from unet.ResUnet import ResUnet_mem_sparse
from guided_diffusion_ori.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion_ori.respace import SpacedDiffusion, space_timesteps
from guided_diffusion_ori.resample import UniformSampler
from scipy.special import comb
import random
set_determinism(123)
import os
import numpy as np
import torch.nn.functional as F
import cupy as cp
max_epoch = 500
batch_size = 2
val_every = 25
num_gpus = 1
env = "pytorch"
device = "cuda"
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def kl_divergence(p_log, q_log):
    return F.kl_div(q_log, p_log.exp(), reduction='batchmean')
def js_divergence(p, q, epsilon=1e-10):
    p_safe = p + epsilon
    q_safe = q + epsilon
    p_normalized = p_safe / p_safe.sum()
    q_normalized = q_safe / q_safe.sum()
    m = 0.5 * (p_normalized + q_normalized)
    m_log = torch.log(m)
    p_log = torch.log(p_normalized)
    q_log = torch.log(q_normalized)
    kl_pm = kl_divergence(p_log, m_log)
    kl_qm = kl_divergence(q_log, m_log)
    js_div = 0.5 * (kl_pm + kl_qm)
    return js_div
class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0.,1.), background_threshold=0.01, nPoints=4, nTimes=100000):
        self.nPoints = nPoints
        self.nTimes = nTimes
        self.vrange = vrange
        self.background_threshold = background_threshold
        self._get_polynomial_array()
    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = torch.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = torch.stack(
            [bernstein_poly(i, self.nPoints - 1, t) for i in range(self.nPoints)]).float()
    def get_bezier_curve(self, points):
        points = torch.tensor(points)
        xPoints = points[:, 0]
        yPoints = points[:, 1]
        xvals = torch.matmul(xPoints, self.polynomial_array)
        yvals = torch.matmul(yPoints, self.polynomial_array)
        return xvals, yvals
    def numpy_interp(self,tensor_inputs, tensor_xvals, tensor_yvals):
        inputs = cp.asarray(tensor_inputs)
        xvals = cp.asarray(tensor_xvals)
        yvals = cp.asarray(tensor_yvals)
        interpolated_values = cp.interp(inputs, xvals, yvals)
        tensor_interpolated_values = torch.tensor(interpolated_values, device=tensor_inputs.device,
                                                  dtype=tensor_inputs.dtype)
        return tensor_interpolated_values
    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point, end_point = inputs.min(), inputs.max()
        xPoints = torch.tensor([start_point, end_point])
        yPoints = torch.tensor([start_point, end_point])
        for _ in range(self.nPoints-2):
            xPoints = torch.cat((xPoints[:1], torch.tensor([random.uniform(xPoints[0].item(), xPoints[-1].item())]), xPoints[1:]))
            yPoints = torch.cat((yPoints[:1], torch.tensor([random.uniform(yPoints[0].item(), yPoints[-1].item())]), yPoints[1:]))
        xvals, yvals = self.get_bezier_curve(torch.stack((xPoints, yPoints), dim=1))
        if inverse and random.random() <= inverse_prop:
            xvals = xvals.sort()[0]
        else:
            sorted_indices = xvals.sort()[1]
            xvals, yvals = xvals.sort()[0], yvals[sorted_indices]
        return self.numpy_interp(inputs, xvals, yvals)
    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = torch.tensor(max(min(random.gauss(1, 0.1), 1.1), 0.9)).float()
        location = torch.tensor(random.gauss(0, 0.5)).float()
        percentile_value = torch.quantile(inputs, slide_limit/100.0)
        location = torch.clamp(location, self.vrange[0] - percentile_value, self.vrange[1] - percentile_value)
        return torch.clamp(inputs * scale + location, self.vrange[0], self.vrange[1])
    def Global_Location_Scale_Augmentation(self, image):
        image = self.non_linear_transformation(image, inverse=False)
        image = self.location_scale_transformation(image).float()
        return image
    def Local_Location_Scale_Augmentation(self, image, mask):
        output_image = torch.zeros_like(image)
        mask = mask.int()
        output_image[mask == 0] = self.location_scale_transformation(self.non_linear_transformation(image[mask == 0], inverse=True, inverse_prop=1))
        for c in range(1, mask.max().item() + 1):
            if (mask == c).sum() == 0:
                continue
            output_image[mask == c] = self.location_scale_transformation(self.non_linear_transformation(image[mask == c], inverse=True, inverse_prop=0.5))
        if self.background_threshold >= self.vrange[0]:
            output_image[image <= self.background_threshold] = image[image <= self.background_threshold]
        return output_image
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
    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None, c_idx=None, aug=False):
        if pred_type == "q_sample":
            shape = x.shape
            noise = torch.randn(shape).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise
        elif pred_type == "get_T":
            shape = x.shape
            noise = torch.randn(batch_size, 4, 64, 192, 192).to(x.device)
            T = max(space_timesteps(50, [10]))
            t = torch.full((batch_size,), T, dtype=torch.long).to(x.device)
            return noise, t
        elif pred_type == "double_sample":
            shape = x.shape
            noise = torch.randn(shape).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            t2, weight = self.sampler.sample(x.shape[0], x.device)
            t1 = torch.min(t, t2)
            t2 = torch.max(t, t2)
            return self.diffusion.q_sample(x, t1, noise=noise), self.diffusion.q_sample(x, t2, noise=noise), t1, t2
        elif pred_type == "denoise":
            return self.model(x, t=step, image=image, aug=aug)
        elif pred_type == "ddim_sample":
            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 4, 64, 192, 192),
                                                                model_kwargs={"image": image, "c_idx": c_idx})
            sample_out = sample_out["pred_xstart"]
            return sample_out
class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py", checkpoint=None):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[64, 192, 192],
                                                 sw_batch_size=1,
                                                 overlap=0.5)
        self.model = DiffUNet()
        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=5,
                                                       max_epochs=max_epoch)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)
        self.location_scale = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)
        if checkpoint != None:
            self.load_state_dict2(checkpoint)
            checkpoint = torch.load(checkpoint)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scheduler.step()
            self.start_epoch = checkpoint['epoch'] + 1
        else:
            self.start_epoch = None
    def training_step(self, batch):
        if self.epoch < 300:
            image, label, c_idx = self.get_input(batch)
            if random.random() > 0.5:
                s_image = self.location_scale.Global_Location_Scale_Augmentation(image.clone())
            else:
                s_image = image.clone()
            if random.random() > 0.5:
                image = self.location_scale.Global_Location_Scale_Augmentation(image.clone())
            x_start = label
            x_T, T = self.model(x=x_start, pred_type="get_T")
            x_0_recycle = self.model(x=x_T, step=T, image=image, pred_type="denoise")
            x_0_recycle = x_0_recycle.clamp(-1, 1)
            x_t, t, noise = self.model(x=x_0_recycle.detach(), pred_type="q_sample")
            del x_0_recycle, x_T
            x_t = x_t.repeat(2, 1, 1, 1, 1)
            t = t.repeat(2)
            image = torch.cat((image, s_image), dim=0)
            pred_xstart = self.model(x=x_t, step=t, image=image, aug=True, pred_type="denoise")
            if env == 'DDP':
                att = self.model.module.model.ResCNN_DeformTR.att
            else:
                att = self.model.model.ResCNN_DeformTR.att
            b, c, d, h, w = pred_xstart.shape
            empty_tensor = torch.empty(b, 1, d, h, w).to(pred_xstart.device)
            for i in range(b):
                empty_tensor[i] = pred_xstart[i, c_idx[i]]
            pred_xstart = empty_tensor
            loss_dice = self.dice_loss(pred_xstart, label)
            loss_bce = self.bce(pred_xstart, label)
            pred_xstart = torch.sigmoid(pred_xstart)
            loss_mse = self.mse(pred_xstart, label)
            loss_con = js_divergence(att[:b], att[b:])
            loss = loss_dice + loss_bce + loss_mse + 0.15 * loss_con
            self.log("train_loss", loss, step=self.global_step)
            return loss
        else:
            image, label, c_idx = self.get_input(batch)
            with torch.no_grad():
                if random.random() > 0.5:
                    s_image = self.location_scale.Global_Location_Scale_Augmentation(image.clone())
                else:
                    s_image = image.clone()
                if random.random() > 0.5:
                    image = self.location_scale.Global_Location_Scale_Augmentation(image.clone())
                x_start = label
                b, c, d, h, w = label.shape
                x_T, T = self.model(x=x_start, pred_type="get_T")
                x_0_recycle = self.model(x=x_T, step=T, image=image, pred_type="denoise")
                x_0_recycle = x_0_recycle.clamp(-1, 1)
                x_t, x_t2, t1, t2 = self.model(x=x_0_recycle.detach(), pred_type="double_sample")
                del x_0_recycle, x_T
                pred_xstart = self.model(x=x_t, step=t1, image=image, pred_type="denoise")
                pseudo_label = torch.empty(b, 3, d, h, w).to(pred_xstart.device)
                for i in range(b):
                    idx = 0
                    for j in range(4):
                        if j != c_idx[i]:
                            pseudo_label[i, idx] = pred_xstart[i, j]
                            idx = idx + 1
                pseudo_label = torch.sigmoid(pseudo_label)
                x_t2 = x_t2.repeat(2, 1, 1, 1, 1)
                t2 = t2.repeat(2)
                image = torch.cat((image, s_image), dim=0)
            x0 = self.model(x=x_t2, step=t2, image=image, aug=True, pred_type="denoise")
            if env == 'DDP':
                att = self.model.module.model.ResCNN_DeformTR.att
            else:
                att = self.model.model.ResCNN_DeformTR.att
            empty_tensor = torch.empty(b, 1, d, h, w).to(x0.device)
            for i in range(b):
                empty_tensor[i] = x0[i, c_idx[i]]
            selected_channel_output = empty_tensor
            left_channel = torch.empty(b, 3, d, h, w).to(x0.device)
            for i in range(b):
                idx = 0
                for j in range(4):
                    if j != c_idx[i]:
                        left_channel[i, idx] = x0[i, j]
                        idx = idx + 1
            loss_dice = self.dice_loss(selected_channel_output, label)
            loss_bce = self.bce(selected_channel_output, label)
            sigmoid_sele = torch.sigmoid(selected_channel_output)
            loss_mse = self.mse(sigmoid_sele, label)
            loss_con = js_divergence(att[:b], att[b:])
            loss = loss_dice + loss_bce + loss_mse + 0.15 * loss_con
            consistency_loss = self.bce(left_channel, pseudo_label.to(x0.device).detach())
            total_loss = loss + 0.1 * consistency_loss
            self.log("train_loss", total_loss, step=self.global_step)
            return total_loss
    def self_training(self, batch):
        image, label, c_idx = self.get_input(batch)
        x_start = label
        x_start = (x_start) * 2 - 1
        x_t, x_t2, t = self.model(x=x_start, pred_type="double_sample")
        with torch.no_grad():
            pred_xstart = self.model(x=x_t, step=t, image=image, pred_type="denoise")
            pred_xstart = torch.sigmoid(pred_xstart)
            b, c, d, h, w = pred_xstart.shape
            pseudo_label = (pred_xstart > 0.5).float()
            for i in range(b):
                pseudo_label[i, c_idx[i]] = label[i]
        x0 = self.model(x=x_t2, step=t, image=image, pred_type="denoise")
        selected_channel_output = x0[torch.arange(b), c_idx, :, :, :]
        selected_channel_output = torch.unsqueeze(selected_channel_output, dim=1)
        loss_dice = self.dice_loss(selected_channel_output, label)
        loss_bce = self.bce(selected_channel_output, label)
        sigmoid_sele = torch.sigmoid(selected_channel_output)
        loss_mse = self.mse(sigmoid_sele, label)
        loss = loss_dice + loss_bce + loss_mse
        x0 = x0.flatten(0, 1)
        x0 = torch.unsqueeze(x0, dim=1)
        pseudo_label = pseudo_label.flatten(0, 1)
        pseudo_label = torch.unsqueeze(pseudo_label, dim=1)
        consistency_loss = self.dice_loss(x0, pseudo_label) + self.bce(x0, pseudo_label)
        total_loss = 0.7 * loss + 0.3 * consistency_loss
        self.log("train_loss", total_loss, step=self.global_step)
        return total_loss
    def get_input(self, batch):
        image = batch["image"]
        label = batch["label"]
        c_idx = batch['c_idx']
        label = label.float()
        return image, label, c_idx
    def convert_labels(self, labels):
        labels_new = []
        labels_new.append(labels == 3)
        labels_new = torch.cat(labels_new, dim=1)
        return labels_new
    def validation_end(self, mean_val_outputs):
        dices = mean_val_outputs
        print(dices)
        if isinstance(mean_val_outputs, list):
            mean_dice = sum(dices) / len(dices)
        else:
            mean_dice = dices
        self.log("mean_dice", mean_dice, step=self.epoch)
        if self.epoch == 0 or (self.epoch + 1) % val_every == 0:
            checkpoint = {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            }
            path = os.path.join(model_save_path, f"epoch{self.epoch:}_model_{mean_dice:.4f}.pt")
            save_dir = os.path.dirname(path)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(checkpoint, path)
        print(f" mean_dice is {mean_dice}")
    def validation_step(self, batch):
        image, label, c_idx = self.get_input(batch)
        output = self.window_infer(image, self.model, pred_type="ddim_sample", c_idx=c_idx)
        b, c, d, h, w = label.shape
        empty_tensor = torch.empty(b, c, d, h, w).to(label.device)
        for i in range(b):
            empty_tensor[i] = output[i, c_idx[i]]
        output = empty_tensor
        output = torch.sigmoid(output)
        output = (output > 0.5).float().cpu().numpy()
        target = label.cpu().numpy()
        dices = dice(output, target)
        return dices
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DiffResmem 5-fold training')
    parser.add_argument('--fold', type=int, default=0, help='5-fold index (0-4)')
    parser.add_argument('--toy', action='store_true', help='Use bundled toy layout under data/toy (default paths)')
    parser.add_argument('--max-epochs', type=int, default=None, help='Override training epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    args = parser.parse_args()
    if args.toy:
        os.environ.setdefault("DIFFRESMEM_LAYOUT", "toy")
        os.environ.setdefault("DIFFRESMEM_DATA_ROOT", "./data/toy")
        os.environ.setdefault("DIFFRESMEM_SPLITS_ROOT", "./dataset/splits")
    if args.max_epochs is not None:
        max_epoch = args.max_epochs
    elif args.toy:
        max_epoch = 2
    if args.batch_size is not None:
        batch_size = args.batch_size
    elif args.toy:
        batch_size = 1
    fold = args.fold
    logdir = f"./checkpoints/5fold_DiffResmem{fold}_/"
    model_save_path = os.path.join(logdir, "model")
    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=17351, checkpoint=None,
                           training_script=__file__)
    train_ds, val_ds, test_ds = get_5fold_loader(fold=fold)
    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
