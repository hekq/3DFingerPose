from operator import index
import os
import sys
sys.path.append(".")
from torchvision.transforms.functional import scale
import torchvision.transforms.functional as TF
from utils import class_array_histogram_equalization,split_path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import pickle
from PIL import Image
import scipy
from scipy import ndimage
import skimage.exposure
import skimage.transform
import matplotlib.pyplot as plt

class fingerset(Dataset):
    def __init__(
            self, pkl, set_type, img_size, num_finger_cls, normalization="0-1"
        ):
        super(fingerset, self).__init__()
        self.pkl = pkl
        self.num_finger_cls = num_finger_cls
        self.set_type = set_type
        self.normalization = normalization
        if isinstance(pkl,str):
            with open(pkl, "rb") as f:
                tmp_data = pickle.load(f)
                self.items = tmp_data
                all_the_yaws = np.array([e["yaw"] for e in tmp_data])
        elif isinstance(pkl,list):
            self.items = []
            all_the_yaws = []
            for each_pkl in pkl:
                with open(each_pkl, "rb") as f:
                    tmp_data = pickle.load(f)
                    self.items += tmp_data
                    all_the_yaws += [e["yaw"] for e in tmp_data]
            all_the_yaws = np.array(all_the_yaws)
        else:
            raise Exception("unknown pkl",pkl)
        self.img_size = np.array(img_size)
        self.heq = class_array_histogram_equalization(all_the_yaws,180)

    def load_img(self, path):
        img = np.asarray(Image.open(path), dtype=np.float32)
        img = torch.from_numpy(img)
        return img

    # def rotate_scale_img(self, img, angle, translate, size, interp, fill):
    #     if interp=="nearest":
    #         interpolation=torchvision.transforms.functional.InterpolationMode.NEAREST
    #     else:
    #         interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR
    #     if isinstance(size, np.ndarray):
    #         size = size.tolist()
    #     img = torchvision.transforms.functional.affine(img.unsqueeze(0),
    #         angle,
    #         translate,
    #         1.0,
    #         [0,0],
    #         interpolation,
    #         fill = fill
    #     )
    #     img = torchvision.transforms.functional.resize(img, size, interpolation=interpolation)
    #     return img

    def rotate_scale_img_skimage(self, img, angle, translate, size, img_type):
        img = img.numpy()
        shift_y, shift_x = np.array(img.shape[:2]) / 2.
        tf_rotate = skimage.transform.SimilarityTransform(rotation=np.deg2rad(angle))
        tf_shift = skimage.transform.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = skimage.transform.SimilarityTransform(translation=[shift_x+translate[1], shift_y+translate[0]])
        # tf_shift_target = skimage.transform.SimilarityTransform(translation=translate)
        
        if img_type=="img":
            image_rotated = skimage.transform.warp(img, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, mode='reflect')
            resized = skimage.transform.resize(image_rotated,size,mode='reflect')
        else: # seg
            image_rotated = skimage.transform.warp(img, (tf_shift + (tf_rotate + tf_shift_inv)).inverse, mode='constant',order=0,cval=0)
            resized = skimage.transform.resize(image_rotated,size,mode='constant',cval=0,order=0)
        resized = torch.from_numpy(resized)[None]
        return resized

    def rotate_scale_ori(self, ori, angle, translate, size):
        ori = self.rotate_scale_img_skimage(ori,angle,translate,size,'ori')
        # rotation orientation:
        # very important for a volume, whose value itself is representing
        # something in spacial dim.
        ori = ori + angle
        ori[ori > 180] = ori[ori > 180] - 180
        ori[ori < 0] = ori[ori < 0] + 180
        # resize
        ratio = size / ori.shape[-2:]
        x = np.cos(ori / 180 * np.pi) * ratio[0]
        y = np.sin(ori / 180 * np.pi) * ratio[1]

        x_new = x / np.sqrt(x ** 2 + y ** 2)
        ori_new = np.arccos(x_new) / np.pi * 180
        return ori_new

    def resize_ori(self, ori, ratio):
        ori = ndimage.zoom(ori, ratio, mode="nearest")
        x = np.cos(ori / 180 * np.pi) * ratio[0]
        y = np.sin(ori / 180 * np.pi) * ratio[1]

        x_new = x / np.sqrt(x ** 2 + y ** 2)
        ori_new = np.arccos(x_new) / np.pi * 180
        return ori_new

    def rotate_img(self, img, angle):
        if angle == 0:
            return img
        rotated = ndimage.rotate(img, angle, mode="nearest", reshape=False)
        return rotated

    def rotate_ori(self, ori, angle):
        if angle == 0:
            return ori
        rotated = ndimage.rotate(ori, angle, mode="nearest", reshape=False)
        ori = (
            ori - angle
        )  # very important for a volume, whose value itself is representing something in spacial dim.
        h, w = np.where(ori > 180)
        ori[h, w] = ori[h, w] - 180
        h, w = np.where(ori < 0)
        ori[h, w] = ori[h, w] + 180
        return rotated

    def normalize_intensity(self, img, method="gaussian", m=None, M=None):
        if method=="gaussian":
            # print('using mean zero and std one')
            return self.normalize_intensity_gaussian(img)
        elif method=='local-heq':
            # print('using local Histogram_equalization')
            return self.local_heq(img)
        elif method=="predefine":
            _min = 132.823
            _max = 255.0
            _mean= 0.953
            _std = 0.145
            eps = 1e-6
            img = (img-_min)/(_max-_min)
            img = (img-_mean)/(_std+eps)
            return img
        elif method=="0-1":
            # print('using 0-1 linear proj')
            if m is None:
                m = img.min()
            if M is None:
                M = img.max()
            return (img - m) / (M - m)
        else:
            raise Exception("unknown normalization ",method)
    def convert_ori(self,ori):
        ori = ori/180*np.pi
        return torch.cat([torch.sin(2*ori) , torch.cos(2*ori)])

    def local_heq(self,img):
        img = img[0]
        arr = (img-img.min())/(img.max()-img.min())
        after_arr = skimage.exposure.equalize_adapthist(arr,16)
        after_arr = after_arr.astype(np.float32)
        return torch.from_numpy(after_arr).unsqueeze(0)

    def normalize_intensity_gaussian(self,img):
        img -= img.mean()
        img = img/(img.std()+1e-6)
        return img

    def process_fingertype(self, origin, num_cls):
        fingertype_ids = {"l0":0,"l1":1,"l2":2,"r0":3,"r1":4,"r2":5}
        if num_cls == 2:
            if origin in ['r0','l0']:
                return 0 
            else:
                return 1

        if num_cls == 4:
            if origin in ['l0']:
                return 0
            if origin in ['r0']:
                return 1
            if origin in ['l1','l2']:
                return 2 
            if origin in ['r1','r2']:
                return 3

        if num_cls == 6:
            return fingertype_ids[origin]

        raise Exception("error of finger type classes")

    def __getitem__(self, index):
        item = self.items[index]
        yaw = float(item["yaw"])
        roll = float(item["roll"])
        pitch = float(item["pitch"])
        finger_type = split_path(item['img'])[1]
        finger_type = self.process_fingertype(finger_type, self.num_finger_cls)
        ## process img
        img_path = item['img']
        seg_path = item['seg']

        img = self.load_img(img_path) # maybe ori
        seg = self.load_img(seg_path)

        original_img = img.clone().unsqueeze(0)
        original_seg = seg.clone().unsqueeze(0)

        new_yaw = self.heq(yaw) # augment the yaw
        if self.set_type == "train":
            #  -90 < new_yaw+tmp < 90
            tmp = np.random.uniform(low=-90 - new_yaw, high=90 - new_yaw)
            new_yaw = new_yaw + tmp
            assert new_yaw > -91 and new_yaw < 91, "yaw range error"
        img_rotation_angle = new_yaw - yaw # affine api

        # img = self.rotate_scale_img(original_img[0], img_rotation_angle, [translation_h,translation_w], self.img_size, "linear", img.max().item())
        img = self.rotate_scale_img_skimage(img, img_rotation_angle, [0,0], self.img_size,"img")
        seg = self.rotate_scale_img_skimage(seg, img_rotation_angle, [0,0], self.img_size,"seg")

        img = self.normalize_intensity(img,self.normalization)
        seg = self.normalize_intensity(seg,"0-1",0,255)
        return {
            "img": img,
            "seg": seg,
            "original_img": original_img,
            "original_seg": original_seg,
            "path": img_path,
            "yaw": new_yaw,
            "original_yaw": yaw,
            "roll": roll,
            "pitch": pitch,
            "finger_type": finger_type,
        }

    def __len__(
        self,
    ):
        return len(self.items)
