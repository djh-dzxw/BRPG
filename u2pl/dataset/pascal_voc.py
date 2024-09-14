import copy
import math
import os
import os.path
import random
from .transform import *
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from . import augmentation as psp_trsform
from .base import BaseDataset


class voc_dset(BaseDataset):
    def __init__(
        self, data_root, data_list, trs_form, seed=0, n_sup=10582, split="val",
            acp=False,
            paste_trs=None,
            prob=0.5,
            flag=None
    ):
        super(voc_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        self.paste_trs = paste_trs
        self.acp = acp and split == "train"
        self.prob = prob
        self.flag = flag

        random.seed(seed)
        if len(self.list_sample) >= n_sup and split == "train":
            self.list_sample_new = random.sample(self.list_sample, n_sup)
        elif len(self.list_sample) < n_sup and split == "train":
            num_repeat = math.ceil(n_sup / len(self.list_sample))
            self.list_sample = self.list_sample * num_repeat

            self.list_sample_new = random.sample(self.list_sample, n_sup)
        else:
            self.list_sample_new = self.list_sample

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        image = self.img_loader(image_path, "RGB")
        label = self.img_loader(label_path, "L")

        # loader paste img and mask
        if self.acp:  
            if random.random() > self.prob:
                paste_idx = random.randint(0, self.__len__() - 1)
                paste_img_path = os.path.join(
                    self.data_root, self.list_sample_new[paste_idx][0]
                )
                paste_img = self.img_loader(paste_img_path, "RGB")
                paste_label_path = os.path.join(
                    self.data_root, self.list_sample_new[paste_idx][1]
                )
                paste_label = self.img_loader(paste_label_path, "L")
                paste_img, paste_label = self.paste_trs(paste_img, paste_label)
            else:
                paste_img, paste_label, instance_label = None, None, None

        if self.flag != 'unlabeled':
            image, label = self.transform(image, label)  

        if self.acp:  
            if paste_img is not None:
                return torch.cat((image[0], paste_img[0]), dim=0), torch.cat(
                    [label[0, 0].long(), paste_label[0, 0].long()], dim=0
                )
            else:
                h, w = image[0].shape[1], image[0].shape[2]
                paste_img = torch.zeros(3, h, w)
                paste_label = torch.zeros(h, w)
                return torch.cat((image[0], paste_img), dim=0), torch.cat(
                    [label[0, 0].long(), paste_label.long()], dim=0
                )

        if self.flag == 'unlabeled':  
            
            label = Image.fromarray(np.array(label))

            
            image, label = resize(image, label, (0.5, 2.0))  
            ignore_value = 254 if self.flag == 'unlabeled' else 255  
            image, label = crop(image, label, 513, ignore_value)  
            image, label = hflip(image, label, p=0.5)  

            
            image_w, image_s = deepcopy(image), deepcopy(image)

            if random.random() < 0.8:  
                image_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(image_s)
            image_s = transforms.RandomGrayscale(p=0.2)(image_s)  
            image_s = blur(image_s, p=0.5)

            
            last_transform = psp_trsform.Compose([psp_trsform.ToTensor(), psp_trsform.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])])
            label_tmp = deepcopy(label)
            image_w, label = last_transform(image_w, label)
            image_s, label_tmp = last_transform(image_s, label_tmp)

            return image_w[0], image_s[0], label[0, 0].long()

        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg, acp=False):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        if not acp:
            trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
        else:
            trs_form.append(psp_trsform.RandResize(cfg["acp"]["rand_resize"]))

    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    return psp_trsform.Compose(trs_form)


def build_vocloader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 10582)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = voc_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup)

    # build sampler
    sample = DistributedSampler(dset)

    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )
    return loader


def build_voc_semi_loader(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    acp = True if "acp" in cfg_dset.keys() else False

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = 10582 - cfg.get("n_sup", 10582)
    prob = cfg["acp"].get("prob", 0.5) if acp else 0


    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg)
    if acp:
        paste_trs = build_transfrom(cfg, acp=True)
    else:
        paste_trs = None

    dset = voc_dset(cfg["data_root"], cfg["data_list"], trs_form, seed, n_sup, split, acp=acp,
        paste_trs=paste_trs,
        prob=prob)

    if split == "val":
        # build sampler
        sample = DistributedSampler(dset)
        loader = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample,
            shuffle=False,
            pin_memory=True,
        )
        return loader

    else:
        # build sampler for unlabeled set
        data_list_unsup = cfg["data_list"].replace("labeled.txt", "unlabeled.txt")
        dset_unsup = voc_dset(
            cfg["data_root"], data_list_unsup, trs_form_unsup, seed, n_sup, split, flag='unlabeled'
        )

        sample_sup = DistributedSampler(dset)
        loader_sup = DataLoader(
            dset,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_sup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        return loader_sup, loader_unsup
