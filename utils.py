import numpy as np
from numpy.ma import isin
from scipy.ndimage.interpolation import affine_transform
import torch
import matplotlib.pyplot as plt
import os
from pathlib import Path
import glob
from torchvision.transforms import RandomAffine
import torchvision
import time
import random
import pickle
from PIL import Image
# import comet_ml
import io

class Profile(object):
    def __enter__(self):
        self.begin = time.time()
        return self
    def __exit__(self,type, value, traceback):
        self.period = time.time()-self.begin
    def __call__(self):
        return self.period

class averageArray():
    def __init__(self,dim=0,MIN=1000,MAX=-1000):
        self.dim = dim
        self.sum = np.zeros(dim)
        self.avg = np.zeros(dim)
        self.min = np.full(shape=(self.dim),fill_value=MIN)
        self.max = np.full(shape=(self.dim),fill_value=MAX)
        self.count = 0
        self.meta = []
        self.metaarray = np.empty((1,dim))

    def update(self,batch_data,need_abs=True):
        '''value is a batch of data'''
        assert isinstance(batch_data,np.ndarray) and batch_data.shape[1]==self.dim
        self.meta.append(batch_data)
        self.metaarray = np.concatenate(self.meta,axis=0)
        batches = batch_data.shape[0]
        if need_abs:
            self.sum += np.abs(batch_data).sum(axis=0)
        else:
            self.sum += batch_data.sum(axis=0)
        self.count += batches
        self.avg = self.sum*1.0/self.count
        self.val = batch_data
        self.min = np.where(batch_data.min(axis=0)<self.min,batch_data.min(axis=0),self.min)
        self.max = np.where(batch_data.max(axis=0)>self.max,batch_data.max(axis=0),self.max)
        self.std = np.std(self.metaarray,axis=0)

class averageScalar():
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self,value,bb):
        self.count += bb
        self.sum += value*bb
        self.avg = self.sum/self.count
        self.val = value

def draw_axis_on_img(img, yaw, pitch, roll, tdx=None, tdy=None, size = 50,thickness=(2,2,2)):
    import cv2
    """
    Function used to draw y (headpose label) on Input Image x.
    Implemented by: shamangary
    https://github.com/shamangary/FSA-Net/blob/master/demo/demo_FSANET.py
    Modified by: Omar Hassan
    """
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),thickness[0])
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),thickness[1])
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),thickness[2])

    return img

def save_model(model,optim,lr_scheduler,epoch,eval_error,path):
    if hasattr(model,"module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    if lr_scheduler is None:
        torch.save({"model":model_state,"optim":optim.state_dict(),"epoch":epoch,"eval_error":eval_error},path)
    else:
        torch.save({"model":model_state,"optim":optim.state_dict(),"lr_scheduler":lr_scheduler.state_dict(),"epoch":epoch,"eval_error":eval_error},path)

def load_model(model,ckp_path):
    def remove_module_string(k):
        items = k.split(".")
        items = items[0:1] + items[2:]
        return ".".join(items)
    if isinstance(ckp_path,str):
        ckp = torch.load(ckp_path,map_location = lambda storage,loc:storage)
        ckp_model_dict = ckp['model']
    else:
        ckp_model_dict = ckp_path

    example_key = list(ckp_model_dict.keys())[0]
    if "module" in example_key:
        ckp_model_dict = {remove_module_string(k):v for k,v in ckp_model_dict.items()}

    if hasattr(model,"module"):
        model.module.load_state_dict(ckp_model_dict)
    else:
        model.load_state_dict(ckp_model_dict)

def generate_random_seed():
    seed = hash(time.time()) % 10000
    return seed

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def my_load_state_dict_with_filter(model,ckp_path):
    ckp = torch.load(ckp_path,lambda storage,_:storage)
    ckp_state_dict = ckp['model']
    model_state_dict = model.state_dict()
    to_be_updated = {k:v for k,v in ckp_state_dict.items() if k in model_state_dict and v.shape==model_state_dict[k].shape}
    model_state_dict.update(to_be_updated)
    print("length of pretained weights: ",len(list(to_be_updated.keys())))
    model.load_state_dict(model_state_dict)

def string_for_loss(names,losses):
    str = ""
    for name,loss in zip(names,losses):
        if isinstance(loss,torch.Tensor):
            loss = loss.item()
        str = str+f"{name}: {loss:.2f}\t"
    return str

def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return torch.tensor(total_norm)

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def compute_rotated_position(abs_position,init_position,init_yaw,single=True):
    init_yaw = np.pi*init_yaw/180
    if single:
        x_hat = np.array([np.cos(init_yaw), -np.sin(init_yaw)])
        y_hat = np.array([np.cos(np.pi/2.0-init_yaw), np.sin(np.pi/2.0-init_yaw)])
    else:
        x_hat = np.stack([np.cos(init_yaw), -np.sin(init_yaw)],axis=1)
        y_hat = np.stack([np.cos(np.pi/2.0-init_yaw), np.sin(np.pi/2.0-init_yaw)],axis=1)

    x_new = np.sum((abs_position-init_position)*x_hat,-1)
    y_new = np.sum((abs_position-init_position)*y_hat,-1)
    if single:
        return np.array([x_new,y_new])
    else:
        return np.stack([x_new,y_new],axis=1)

def solve_fingerset_path(img_path):
    # /home/hk/lab_pose/raw/cwx/l0/100000_0.png
    pid = img_path.strip().split("/")[-3]
    fid = img_path.strip().split("/")[-2]
    seq = img_path.strip().split("/")[-1].split('.')[0]
    return pid,fid,seq

class time_tracking():
    def __init__(self):
        self.last_time = 0
    def __call__(self,name="",init=False):
        if init:
            self.last_time = time.time()
            return 
        now = time.time()
        print(f"Part {name} spent {round(now-self.last_time,2)} seconds")
        self.last_time = now

class IntervalRecorder():
    def __init__(self,count,_min,_max):
        self.count = count
        self.p_gt = np.zeros((count,)).astype(np.float32)
        self.p_pred = np.zeros((count,)).astype(np.float32)

        self._min = _min
        self._max = _max
        self.interval_length = float((_max - _min))/self.count
        # self.e = np.zeros((count,)).astype(np.float32)
        self.precision = np.zeros((count,)).astype(np.float32)

    def update(self,pred,gt):
        for e_pred,e_gt in zip(pred,gt):
            error = np.abs(e_gt-e_pred)
            index_gt = np.floor((e_gt-self._min)/self.interval_length).astype(int)
            index_gt = min(index_gt,self.count-1)
            self.p_gt[index_gt] += 1
            # self.e[index_gt] += error
            
            index_pred = np.floor((e_pred-self._min)/self.interval_length).astype(int)
            index_pred = min(index_pred,self.count-1)
            self.p_pred[index_pred] += 1
            
            if index_pred==index_gt:
                self.precision[index_pred] += 1

def array_histogram_equalization(array, number_bins=180):
    # get image histogram
    image_histogram, bins = np.histogram(array.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 180 * cdf / cdf[-1] - 90# normalize

    # use linear interpolation of cdf to find new pixel values
    def func(src):
        if isinstance(src,np.ndarray):
            image_equalized = np.interp(src.flatten(), bins[:-1], cdf)
            return image_equalized.reshape(src.shape)
        elif isinstance(src,list):
            src = np.array(src)
            image_equalized = np.interp(src.flatten(), bins[:-1], cdf)
            return image_equalized.reshape(src.shape)
        else:
            assert isinstance(src,(float,int))
            value = np.interp(np.array([src]), bins[:-1], cdf)
            return value[0]
    return func


class class_array_histogram_equalization():
    def __init__(self,array, number_bins=180):
        image_histogram, bins = np.histogram(array.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        self.cdf = 180 * cdf / cdf[-1] - 90# normalize
        self.bins = bins

    def __call__(self,src):
        if isinstance(src,np.ndarray):
            image_equalized = np.interp(src.flatten(), self.bins[:-1], self.cdf)
            return image_equalized.reshape(src.shape)
        elif isinstance(src,list):
            src = np.array(src)
            image_equalized = np.interp(src.flatten(), self.bins[:-1], self.cdf)
            return image_equalized.reshape(src.shape)
        else:
            assert isinstance(src,(float,int))
            value = np.interp(np.array([src]), self.bins[:-1], self.cdf)
            return value[0]

def split_path(path):
    return path.split("/")[-3:]

def unique_id(path):
    uid,fid,seq = split_path(path)
    return '-'.join([uid,fid,seq[:seq.find(".")]])

def count_parameters(model,human_readable=True):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if human_readable:
        count = count*1.0/1e6
        return f"{count}M"
    else:
        return f"{count}"

def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        top = 2 * np.sum(np.logical_and(vol1 == lab, vol2 == lab))
        bottom = np.sum(vol1 == lab) + np.sum(vol2 == lab)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)
