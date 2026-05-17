from sklearn.model_selection import KFold
import os
import json
import math
import numpy as np
import torch
from monai import transforms, data
import SimpleITK as sitk
from tqdm import tqdm
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from monai.data import DataLoader
import copy
from dataset.paths import (
    SPLITS_ROOT,
    image_paths_from_split,
    resolve_abd_aug_paths,
    resolve_abd_aug_paths_processed,
    train_dataset_dir,
)
def resample_img(
    image: sitk.Image,
    out_spacing = (2.0, 2.0, 2.0),
    out_size = None,
    is_label: bool = False,
    pad_value = 0.,
) -> sitk.Image:
    """
    Resample images to target resolution spacing
    Ref: SimpleITK
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    out_spacing = list(out_spacing)[::-1]
    if out_size is None:
        out_size = [
            int(np.round(
                size * (spacing_in / spacing_out)
            ))
            for size, spacing_in, spacing_out in zip(original_size, original_spacing, out_spacing)
        ]
    if pad_value is None:
        pad_value = image.GetPixelIDValue()
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(list(out_spacing))
    resample.SetSize(out_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    image = resample.Execute(image)
    return image
class PretrainDataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d  = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        if 'abd2' in image_path:
            image_path = copy.deepcopy(data_path).replace('abd2', 'abd')
            label_path = image_path.replace('img', 'label')
            idx = 4
            c_idx = 3
        elif 'word' in image_path:
            idx = [3, 4]
            c_idx = 1
        elif 'amos' in image_path:
            idx = 6
            c_idx = 2
        elif 'abd' in image_path:
            idx = 3
            c_idx = 0
        image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        if idx == [3, 4]:
            seg_data[(seg_data != 3) & (seg_data != 4)] = 0
            seg_data[seg_data != 0] = 1
        else:
            seg_data[seg_data!=idx] = 0
            seg_data[seg_data != 0] = 1
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)
        return {
            "image": image_data,
            "label": seg_data,
            'c_idx': c_idx
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else :
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"Failed to read sample: {self.datalist[i]}\n")
                if i != len(self.datalist)-1:
                    return self.__getitem__(i+1)
                else :
                    return self.__getitem__(i-1)
        if self.transform is not None :
            image = self.transform(image)
        return image
    def __len__(self):
        return len(self.datalist)
class PretrainDataset_efficient(Dataset):
    def __init__(self, datalist, transform=None, cache=False,ratio=2) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        self.ratio = ratio
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        if 'processed_abd2' in image_path:
            image_path = copy.deepcopy(data_path).replace('processed_abd2', 'processed_abd')
            label_path = image_path.replace('img', 'label')
            idx = 4
            c_idx = 3
        elif 'processed_word' in image_path:
            idx = [3, 4]
            c_idx = 1
        elif 'processed_amos' in image_path:
            idx = 6
            c_idx = 2
        elif 'processed_abd' in image_path:
            idx = 3
            c_idx = 0
        image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        seg_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        if idx == [3, 4]:
            seg_data[(seg_data != 3) & (seg_data != 4)] = 0
            seg_data[seg_data != 0] = 1
        else:
            seg_data[seg_data != idx] = 0
            seg_data[seg_data != 0] = 1
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)
        return {
            "image": image_data,
            "label": seg_data,
            'c_idx': c_idx
        }
    def __getitem__(self, i):
        i = i % len(self.datalist)
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"Failed to read sample: {self.datalist[i]}\n")
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    return self.__getitem__(i - 1)
        if self.transform is not None:
            image = self.transform(image)
        return image
    def __len__(self):
        return len(self.datalist)*self.ratio
import glob
def get_loader_btcv(data_dir, cache=False):
    train_path = os.path.join(data_dir,'train','img')
    val_path = os.path.join(data_dir,'val','img')
    train_files = os.listdir(train_path)
    train_files = [os.path.join(train_path,p) for p in train_files]
    val_files = os.listdir(val_path)
    val_files = [os.path.join(val_path,p) for p in val_files]
    val_files = val_files[:5]
    test_path = os.path.join(data_dir, 'test', 'img')
    test_files = os.listdir(test_path)
    test_files = [os.path.join(test_path, p) for p in test_files]
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=(64,192,192)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64,192,192),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2),max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"],),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    train_ds = PretrainDataset(train_files, transform=train_transform, cache=cache)
    val_ds = PretrainDataset(val_files, transform=val_transform, cache=cache)
    test_ds = PretrainDataset2(test_files, transform=test_transform)
    loader = [train_ds, val_ds, test_ds]
    return loader
def get_mix_loader(data_dir, cache=False):
    train_p1 = './data/processed_abd/train'
    train_p2 = './data/processed_word/train'
    train_p3 = './data/processed_amos/train'
    test_p1 = './data/processed_abd/test'
    test_p2 = './data/processed_word/test'
    test_p3 = './data/processed_amos/test'
    train_img1 = os.path.join(train_p1,'img')
    train_img2 = os.path.join(train_p2,'img')
    train_img3 = os.path.join(train_p3, 'img')
    test_img1 = os.path.join(test_p1,'img')
    test_img2 = os.path.join(test_p2, 'img')
    test_img3 = os.path.join(test_p3, 'img')
    train_files1 = os.listdir(train_img1)
    train_files1 = [os.path.join(train_img1,p) for p in train_files1]
    train_files2 = os.listdir(train_img2)
    train_files2 = [os.path.join(train_img2, p) for p in train_files2]
    train_files3 = os.listdir(train_img3)
    train_files3 = [os.path.join(train_img3, p) for p in train_files3]
    train_files4 = copy.deepcopy(train_files1)
    train_files4 = [p.replace('processed_abd','processed_abd2') for p in train_files4]
    train_files = train_files1+train_files2+train_files3+train_files4
    test_files1 = os.listdir(test_img1)
    test_files1 = [os.path.join(test_img1,p) for p in test_files1]
    test_files2 = os.listdir(test_img2)
    test_files2 = [os.path.join(test_img2, p) for p in test_files2]
    test_files3 = os.listdir(test_img3)
    test_files3 = [os.path.join(test_img3, p) for p in test_files3]
    test_files4 = copy.deepcopy(test_files1)
    test_files4 = [p.replace('processed_abd','processed_abd2') for p in test_files4]
    test_files = test_files1+test_files2+test_files3+test_files4
    val_files = test_files1[:2]+test_files2[:2]+test_files3[:2]+test_files4[:2]
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=(64, 192, 192)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 192, 192),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2), max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    train_ds = PretrainDataset(train_files, transform=train_transform, cache=cache)
    val_ds = PretrainDataset(val_files, transform=val_transform, cache=cache)
    test_ds = PretrainDataset2(test_files, transform=test_transform)
    loader = [train_ds, val_ds, test_ds]
    return loader
def get_test_mix_loader():
    train_p1 = './data/processed_abd/train'
    train_p2 = './data/processed_word/train'
    train_p3 = './data/processed_amos/train'
    test_p1 = './data/processed_abd/test'
    test_p2 = './data/processed_word/test'
    test_p3 = './data/processed_amos/test'
    train_img1 = os.path.join(train_p1,'img')
    train_img2 = os.path.join(train_p2,'img')
    train_img3 = os.path.join(train_p3, 'img')
    test_img1 = os.path.join(test_p1,'img')
    test_img2 = os.path.join(test_p2, 'img')
    test_img3 = os.path.join(test_p3, 'img')
    train_files1 = os.listdir(train_img1)
    train_files1 = [os.path.join(train_img1,p) for p in train_files1]
    train_files2 = os.listdir(train_img2)
    train_files2 = [os.path.join(train_img2, p) for p in train_files2]
    train_files3 = os.listdir(train_img3)
    train_files3 = [os.path.join(train_img3, p) for p in train_files3]
    train_files4 = copy.deepcopy(train_files1)
    train_files4 = [p.replace('processed_abd','processed_abd2') for p in train_files4]
    train_files = train_files1+train_files2+train_files3+train_files4
    test_files1 = os.listdir(test_img1)
    test_files1 = [os.path.join(test_img1,p) for p in test_files1]
    test_files2 = os.listdir(test_img2)
    test_files2 = [os.path.join(test_img2, p) for p in test_files2]
    test_files3 = os.listdir(test_img3)
    test_files3 = [os.path.join(test_img3, p) for p in test_files3]
    test_files4 = copy.deepcopy(test_files1)
    test_files4 = [p.replace('processed_abd','processed_abd2') for p in test_files4]
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=(64, 192, 192)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 192, 192),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2), max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    test1 = PretrainDataset2(test_files1, transform=test_transform)
    test2 = PretrainDataset2(test_files2, transform=test_transform)
    test3 = PretrainDataset2(test_files3, transform=test_transform)
    test4 = PretrainDataset2(test_files4, transform=test_transform)
    loader = [test1,test2,test3,test4]
    return loader
def get_test_full_loader():
    train_p1 = './data/processed_abd/train'
    train_p2 = './data/processed_word/train'
    train_p3 = './data/processed_amos/train'
    test_p1 = './data/processed_abd/test'
    test_p2 = './data/processed_word/test'
    test_p3 = './data/processed_amos/test'
    train_img1 = os.path.join(train_p1,'img')
    train_img2 = os.path.join(train_p2,'img')
    train_img3 = os.path.join(train_p3, 'img')
    test_img1 = os.path.join(test_p1,'img')
    test_img2 = os.path.join(test_p2, 'img')
    test_img3 = os.path.join(test_p3, 'img')
    train_files1 = os.listdir(train_img1)
    train_files1 = [os.path.join(train_img1,p) for p in train_files1]
    train_files2 = os.listdir(train_img2)
    train_files2 = [os.path.join(train_img2, p) for p in train_files2]
    train_files3 = os.listdir(train_img3)
    train_files3 = [os.path.join(train_img3, p) for p in train_files3]
    train_files4 = copy.deepcopy(train_files1)
    train_files4 = [p.replace('processed_abd','processed_abd2') for p in train_files4]
    train_files = train_files1+train_files2+train_files3+train_files4
    test_files1 = os.listdir(test_img1)
    test_files1 = [os.path.join(test_img1,p) for p in test_files1]
    test_files2 = os.listdir(test_img2)
    test_files2 = [os.path.join(test_img2, p) for p in test_files2]
    test_files3 = os.listdir(test_img3)
    test_files3 = [os.path.join(test_img3, p) for p in test_files3]
    test_files4 = copy.deepcopy(test_files1)
    test_files4 = [p.replace('processed_abd','processed_abd2') for p in test_files4]
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=(64, 192, 192)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 192, 192),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2), max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    test1 = Abd_Dataset4test(test_files1, transform=test_transform)
    test2 = Word_Dataset4test(test_files2, transform=test_transform)
    test3 = AMOS_Dataset4test(test_files3, transform=test_transform)
    loader = [test1,test2,test3]
    return loader
def get_5fold_test_full_loader(fold):
    test_files1 = image_paths_from_split("abd", fold, "val", for_test=True)
    test_files2 = image_paths_from_split("word", fold, "val", for_test=True)
    test_files3 = image_paths_from_split("amos", fold, "val", for_test=True)
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=(64, 192, 192)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 192, 192),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2), max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    test_files1 = [test_files1[36]]
    test1 = Abd_Dataset4test(test_files1, transform=test_transform)
    test2 = Word_Dataset4test(test_files2, transform=test_transform)
    test3 = AMOS_Dataset4test(test_files3, transform=test_transform)
    loader = [test1,test2,test3]
    return loader
def get_BTCV_loader(test_path=None):
    if test_path is None:
        test_path = './data/processed_btcv/test'
    test_img = os.path.join(test_path,'img')
    test_files = os.listdir(test_img)
    test_files = [os.path.join(test_img, p) for p in test_files]
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    test = BTCV_Dataset4test(test_files, transform=test_transform)
    loader = test
    return loader
def get_BTCV_test_loader(test_path=None):
    if test_path is None:
        test_path = './data/processed_btcv/test'
    test_img = os.path.join(test_path,'img')
    test_files = os.listdir(test_img)
    test_files = [os.path.join(test_img, p) for p in test_files]
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    test = BTCV_Dataset4test(test_files, transform=test_transform)
    loader = test
    return loader
def BTCV_test_loader_for_visualize(test_path=None):
    if test_path is None:
        test_path = './data/processed_btcv/test'
    test_img = os.path.join(test_path,'img')
    test_files = os.listdir(test_img)
    test_files = [os.path.join(test_img, p) for p in test_files]
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "raw_label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "raw_label"), spatial_size=(64, 192, 192)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "raw_label"],
                label_key="raw_label",
                spatial_size=(64, 192, 192),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    test = BTCV_Dataset4vis(test_files, transform=test_transform)
    loader = test
    return loader
def write_paths_to_txt(file_paths, txt_file):
    with open(txt_file, 'w') as f:
        for path in file_paths:
            f.write("%s\n" % path)
def read_paths_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        paths = f.readlines()
        paths = [path.strip() for path in paths]
    return paths
def get_efficient_loader(data_dir, cache=False,spatial_size=(96, 96, 96)):
    train_p1 = './data/processed_abd/train'
    train_p2 = './data/processed_word/train'
    train_p3 = './data/processed_amos/train'
    test_p1 = './data/processed_abd/test'
    test_p2 = './data/processed_word/test'
    test_p3 = './data/processed_amos/test'
    train_img1 = os.path.join(train_p1,'img')
    train_img2 = os.path.join(train_p2,'img')
    train_img3 = os.path.join(train_p3, 'img')
    test_img1 = os.path.join(test_p1,'img')
    test_img2 = os.path.join(test_p2, 'img')
    test_img3 = os.path.join(test_p3, 'img')
    train_files1 = read_paths_from_txt('./dataset/train1_0.25.txt')
    train_files1 = [os.path.join(train_img1,p) for p in train_files1]
    train_files2 = read_paths_from_txt('./dataset/train2_0.25.txt')
    train_files2 = [os.path.join(train_img2, p) for p in train_files2]
    train_files3 = read_paths_from_txt('./dataset/train3_0.25.txt')
    train_files3 = [os.path.join(train_img3, p) for p in train_files3]
    train_files4 = copy.deepcopy(train_files1)
    train_files4 = [p.replace('processed_abd','processed_abd2') for p in train_files4]
    train_files = train_files1+train_files2+train_files3+train_files4
    test_files1 = os.listdir(test_img1)
    test_files1 = [os.path.join(test_img1,p) for p in test_files1]
    test_files2 = os.listdir(test_img2)
    test_files2 = [os.path.join(test_img2, p) for p in test_files2]
    test_files3 = os.listdir(test_img3)
    test_files3 = [os.path.join(test_img3, p) for p in test_files3]
    test_files4 = copy.deepcopy(test_files1)
    test_files4 = [p.replace('processed_abd','processed_abd2') for p in test_files4]
    test_files = test_files1+test_files2+test_files3+test_files4
    val_files = test_files1[:2]+test_files2[:2]+test_files3[:2]+test_files4[:2]
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=spatial_size),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2), max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    train_ds = PretrainDataset_efficient(train_files, transform=train_transform, cache=cache,ratio=2)
    val_ds = PretrainDataset(val_files, transform=val_transform, cache=cache)
    test_ds = PretrainDataset2(test_files, transform=test_transform)
    loader = [train_ds, val_ds, test_ds]
    return loader
def get_idea_loader(data_dir=None, cache=False,spatial_size=(96, 96, 96)):
    json_path = os.environ.get(
        "DIFFRESMEM_SPLIT_JSON",
        str(SPLITS_ROOT / "idea_split.json"),
    )
    with open(json_path, 'r', encoding='utf-8') as f:
        splits_data = json.load(f)
    train_p1 = str(train_dataset_dir("abd"))
    train_p2 = str(train_dataset_dir("word"))
    train_p3 = str(train_dataset_dir("amos"))
    train_img1 = os.path.join(train_p1, 'img')
    train_img2 = os.path.join(train_p2, 'img')
    train_img3 = os.path.join(train_p3, 'img')
    train_files1 = [os.path.join(train_img1, p) for p in splits_data['./data/processed_abd']['train']]
    train_files2 = [os.path.join(train_img2, p) for p in splits_data['./data/processed_word']['train']]
    train_files3 = [os.path.join(train_img3, p) for p in splits_data['./data/processed_amos']['train']]
    train_files4 = copy.deepcopy(train_files1)
    train_files4 = [p.replace('abd', 'abd2') for p in train_files4]
    train_files = train_files1 + train_files2 + train_files3 + train_files4
    test_files1 = [os.path.join(train_img1, p) for p in splits_data['./data/processed_abd']['test']]
    test_files2 = [os.path.join(train_img2, p) for p in splits_data['./data/processed_word']['test']]
    test_files3 = [os.path.join(train_img3, p) for p in splits_data['./data/processed_amos']['test']]
    test_files4 = copy.deepcopy(test_files1)
    test_files4 = [p.replace('abd', 'abd2') for p in test_files4]
    test_files = test_files1 + test_files2 + test_files3 + test_files4
    val_files = test_files1[:2] + test_files2[:2] + test_files3[:2] + test_files4[:2]
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=spatial_size),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2), max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    train_ds = PretrainDataset(train_files, transform=train_transform, cache=cache)
    val_ds = PretrainDataset(val_files, transform=val_transform, cache=cache)
    test_ds = PretrainDataset2(test_files, transform=test_transform)
    loader = [train_ds, val_ds, test_ds]
    return loader
class PretrainDataset2(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        if 'processed_abd2' in image_path:
            image_path = copy.deepcopy(data_path).replace('processed_abd2', 'processed_abd')
            label_path = image_path.replace('img', 'label')
            idx = 4
            c_idx = 3
        elif 'processed_word' in image_path:
            idx = [3, 4]
            c_idx = 1
        elif 'processed_amos' in image_path:
            idx = 6
            c_idx = 2
        elif 'processed_abd' in image_path:
            idx = 3
            c_idx = 0
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        if idx == [3, 4]:
            raw_label_data[(raw_label_data != 3) & (raw_label_data != 4)] = 0
            raw_label_data[raw_label_data != 0] = 1
        else:
            raw_label_data[raw_label_data != idx] = 0
            raw_label_data[raw_label_data != 0] = 1
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        seg_data = sitk.GetArrayFromImage(
            resample_img(sitk.ReadImage(label_path), out_spacing=[2.0, 1.5, 1.5], is_label=True))
        if idx == [3, 4]:
            seg_data[(seg_data != 3) & (seg_data != 4)] = 0
            seg_data[seg_data != 0] = 1
        else:
            seg_data[seg_data != idx] = 0
            seg_data[seg_data != 0] = 1
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)
        return {
            "image": image_data,
            "label": seg_data,
            "raw_label": raw_label_data,
            'c_idx':c_idx
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            try:
                image = self.read_data(self.datalist[i])
            except:
                with open("./bugs.txt", "a+") as f:
                    f.write(f"Failed to read sample: {self.datalist[i]}\n")
                if i != len(self.datalist) - 1:
                    return self.__getitem__(i + 1)
                else:
                    return self.__getitem__(i - 1)
        if self.transform is not None:
            image = self.transform(image)
        return image
    def __len__(self):
        return len(self.datalist)
def get_5fold_loader(fold, cache=False):
    train_files1 = image_paths_from_split("abd", fold, "train", for_test=False)
    train_files2 = image_paths_from_split("word", fold, "train", for_test=False)
    train_files3 = image_paths_from_split("amos", fold, "train", for_test=False)
    train_files4 = resolve_abd_aug_paths(train_files1)
    train_files = train_files1 + train_files2 + train_files3 + train_files4
    test_files1 = image_paths_from_split("abd", fold, "val", for_test=False)
    test_files2 = image_paths_from_split("word", fold, "val", for_test=False)
    test_files3 = image_paths_from_split("amos", fold, "val", for_test=False)
    test_files4 = resolve_abd_aug_paths(test_files1)
    test_files = test_files1 + test_files2 + test_files3 + test_files4
    val_files = test_files1[:2] + test_files2[:2] + test_files3[:2] + test_files4[:2]
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=(64, 192, 192)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 192, 192),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2), max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label"]),
        ]
    )
    train_ds = PretrainDataset(train_files, transform=train_transform, cache=cache)
    val_ds = PretrainDataset(val_files, transform=val_transform, cache=cache)
    test_ds = PretrainDataset2(test_files, transform=test_transform)
    loader = [train_ds, val_ds, test_ds]
    return loader
def get_test_5fold_loader(fold):
    train_files1 = image_paths_from_split("abd", fold, "train", for_test=True)
    train_files2 = image_paths_from_split("word", fold, "train", for_test=True)
    train_files3 = image_paths_from_split("amos", fold, "train", for_test=True)
    train_files4 = resolve_abd_aug_paths_processed(train_files1)
    train_files = train_files1 + train_files2 + train_files3 + train_files4
    test_files1 = image_paths_from_split("abd", fold, "val", for_test=True)
    test_files2 = image_paths_from_split("word", fold, "val", for_test=True)
    test_files3 = image_paths_from_split("amos", fold, "val", for_test=True)
    test_files4 = resolve_abd_aug_paths_processed(test_files1)
    train_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.SpatialPadd(keys=("image", "label"), spatial_size=(64, 192, 192)),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(64, 192, 192),
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, spatial_axes=(1, 2), max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"], ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=-175, a_max=250.0, b_min=0, b_max=1.0, clip=True
            ),
            transforms.ToTensord(keys=["image", "raw_label","label"]),
        ]
    )
    test1 = Abd_Dataset(test_files1, transform=test_transform)
    test2 = Word_Dataset(test_files2, transform=test_transform)
    test3 = AMOS_Dataset(test_files3, transform=test_transform)
    loader = [test1,test2,test3]
    return loader
class Abd_Dataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        seg_data = sitk.GetArrayFromImage(
            resample_img(sitk.ReadImage(label_path), out_spacing=[2.0, 1.5, 1.5], is_label=True))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)
        return {
            "image": image_data,
            "label": seg_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        idx = [3, 1, 4, 13, 2]
        shape = image['image'].shape
        selected_label = torch.empty((4,shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
class Abd_Dataset4test(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return {
            "image": image_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        idx = [3, 1, 4, 13, 2]
        shape = image['raw_label'].shape
        selected_label = torch.empty((4,shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['raw_label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['raw_label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['raw_label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
class Word_Dataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        seg_data = sitk.GetArrayFromImage(
            resample_img(sitk.ReadImage(label_path), out_spacing=[2.0, 1.5, 1.5], is_label=True))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)
        return {
            "image": image_data,
            "label": seg_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        idx = [2, 1, 8, 3, 4]
        shape = image['image'].shape
        selected_label = torch.empty((4,shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
class Word_Dataset4test(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return {
            "image": image_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        idx = [2, 1, 8, 3, 4]
        shape = image['raw_label'].shape
        selected_label = torch.empty((4, shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['raw_label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['raw_label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['raw_label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
class AMOS_Dataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        seg_data = sitk.GetArrayFromImage(
            resample_img(sitk.ReadImage(label_path), out_spacing=[2.0, 1.5, 1.5], is_label=True))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)
        return {
            "image": image_data,
            "label": seg_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        idx = [1, 6, 10, 2, 3]
        shape = image['image'].shape
        selected_label = torch.empty((4,shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
class AMOS_Dataset4test(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return {
            "image": image_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        idx = [1, 6, 10, 2, 3]
        shape = image['raw_label'].shape
        selected_label = torch.empty((4, shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['raw_label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['raw_label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['raw_label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
class BTCV_Dataset(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        seg_data = sitk.GetArrayFromImage(
            resample_img(sitk.ReadImage(label_path), out_spacing=[2.0, 1.5, 1.5], is_label=True))
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        seg_data = np.expand_dims(seg_data, axis=0).astype(np.int32)
        return {
            "image": image_data,
            "label": seg_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        idx = [1, 6, 11, 3, 2]
        shape = image['image'].shape
        selected_label = torch.empty((4,shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
class BTCV_Dataset4test(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return {
            "image": image_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        idx = [1, 6, 11, 3, 2]
        shape = image['raw_label'].shape
        selected_label = torch.empty((4, shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['raw_label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['raw_label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['raw_label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
class BTCV_Dataset4vis(Dataset):
    def __init__(self, datalist, transform=None, cache=False) -> None:
        super().__init__()
        self.transform = transform
        self.datalist = datalist
        self.cache = cache
        if cache:
            self.cache_data = []
            for i in tqdm(range(len(datalist)), total=len(datalist)):
                d = self.read_data(datalist[i])
                self.cache_data.append(d)
    def read_data(self, data_path):
        image_path = data_path
        label_path = data_path.replace('img', 'label')
        image_data = sitk.GetArrayFromImage(resample_img(sitk.ReadImage(image_path), out_spacing=[2.0, 1.5, 1.5]))
        raw_label_data = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        raw_label_data = np.expand_dims(raw_label_data, axis=0).astype(np.int32)
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return {
            "image": image_data,
            "raw_label": raw_label_data,
        }
    def __getitem__(self, i):
        if self.cache:
            image = self.cache_data[i]
        else:
            image = self.read_data(self.datalist[i])
        if self.transform is not None:
            image = self.transform(image)
        image = image[0]
        idx = [1, 6, 11, 3, 2]
        shape = image['raw_label'].shape
        selected_label = torch.empty((4, shape[1], shape[2], shape[3]))
        for i in range(3):
            raw_label_data = copy.deepcopy(image['raw_label'])
            raw_label_data[raw_label_data != idx[i]] = 0
            raw_label_data[raw_label_data != 0] = 1
            selected_label[i] = raw_label_data
        seg_data = copy.deepcopy(image['raw_label'])
        seg_data[(seg_data != idx[-2]) & (seg_data != idx[-1])] = 0
        seg_data[seg_data != 0] = 1
        selected_label[-1] = seg_data
        image['raw_label'] = selected_label
        return image
    def __len__(self):
        return len(self.datalist)
def convert_labels(labels,label_idx):
    labels_new = []
    labels_new.append(labels == label_idx)
    labels_new = torch.cat(labels_new, dim=1)
    return labels_new
if __name__ == '__main__':
    data_dir = "./data/processed_amos/"
    label_idx =6
    batch_size =1
    train_ds, val_ds, test_ds = get_5fold_loader(fold=0)
    train_dataloader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=0,
        drop_last=True, shuffle=True)
    for i, batch in enumerate(train_dataloader):
        data, target, c_idx = batch['image'],batch['label'], batch['c_idx']
        data = data.cpu()
        target = target.cpu()
        b, c, d, h, w = data.size()
        plt.figure(figsize=(15, 15))
        plt.subplot(231)
        plt.imshow(data[:, :, d // 2, :, :].squeeze(), cmap='gray')
        plt.title('Original Image')
        plt.subplot(232)
        plt.imshow(data[:, :, d // 4, :, :].squeeze(), cmap='gray')
        plt.title('Label id {}'.format(c_idx))
        plt.subplot(233)
        plt.imshow(data[:, :, d // 2 + d // 4, :, :].squeeze(), cmap='gray')
        plt.subplot(234)
        plt.imshow(target[:, :, d // 2, :, :].squeeze(), cmap='viridis')
        plt.subplot(235)
        plt.imshow(target[:, :, d // 4, :, :].squeeze(), cmap='viridis')
        plt.subplot(236)
        plt.imshow(target[:, :, d // 2 + d // 4, :, :].squeeze(), cmap='viridis')
        plt.show()
        print('hello')
