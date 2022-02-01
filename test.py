import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
import yaml
import pickle
import os
from operator import attrgetter
from build_models import UltraModel
from dataset import fingerset
from utils import averageScalar,string_for_loss
from utils import dice as Dice_coef
from yacs.config import CfgNode as CN

run = '2022-01-30-01-48-09'
with open(os.path.join('./runs',run,'config.yaml'),'r') as f:
    config = CN(yaml.safe_load(f))

model = UltraModel(1,config.MODEL.NUM_CLASSES,2,config.MODEL.BACKBONE).cuda()
best_ckp = torch.load(os.path.join('./models',run,'best.pth.tar'),map_location=lambda storage,_:storage)
model.load_state_dict(best_ckp["model"])
model.eval()


testset  = fingerset('./pkls/test.pkl', 'test', [64,64],2, config.DATA.NORMALIZATION)
dataloader = DataLoader(testset,128,False,num_workers=8)

yaw_error_scalar = averageScalar()
pitch_error_scalar = averageScalar()
roll_error_scalar = averageScalar()
finger_acc_scalar = averageScalar()
seg_dice_scalar = averageScalar()

def length_arange(min,max,length):
    return torch.arange(min,max,float(max-min)/length)
idx_tensor_yaw = length_arange(config.MODEL.YAW_MIN,config.MODEL.YAW_MAX,config.MODEL.NUM_CLASSES[0]).cuda()
idx_tensor_pitch = length_arange(config.MODEL.PITCH_MIN,config.MODEL.PITCH_MAX,config.MODEL.NUM_CLASSES[1]).cuda()
idx_tensor_roll = length_arange(config.MODEL.ROLL_MIN,config.MODEL.ROLL_MAX,config.MODEL.NUM_CLASSES[2]).cuda()

bad_predict = []
for iterx, item in enumerate(dataloader):
    with torch.no_grad():
        img = item["img"].cuda()
        yaw_gt   = item['yaw'].cuda().to(torch.float32)
        pitch_gt = item["pitch"].cuda().to(torch.float32)
        roll_gt  = item["roll"].cuda().to(torch.float32)
        gt = torch.stack([yaw_gt,pitch_gt,roll_gt],dim=-1)
        finger = item['finger_type'].cuda()
        path = item['path']
        seg = item['seg'].cuda()
        bb = img.shape[0]

        result = model(img)
        yaw_prob = result["yaw"]
        pitch_prob = result["pitch"]
        roll_prob = result['roll']
        finger_probs = result['finger']
        fcn_out = result['fcn']
        fcn_out = torch.sigmoid(fcn_out)
        fcn_out = (fcn_out>0.75).cpu().numpy().astype(np.int32)

        if config.TRAIN.CLS_LOSS=="bce":
            yaw_predict = torch.sum(torch.sigmoid(yaw_prob)*idx_tensor_yaw[None],dim=1)/config.MODEL.NUM_CLASSES[0]
            pitch_predict = torch.sum(torch.sigmoid(pitch_prob)*idx_tensor_pitch[None],dim=1)/config.MODEL.NUM_CLASSES[1]
            roll_predict = torch.sum(torch.sigmoid(roll_prob)*idx_tensor_roll[None],dim=1)/config.MODEL.NUM_CLASSES[2]
        else:
            yaw_predict = (F.softmax(yaw_prob,dim=1)*idx_tensor_yaw[None]).sum(dim=1)
            pitch_predict = (F.softmax(pitch_prob,dim=1)*idx_tensor_pitch[None]).sum(dim=1)
            roll_predict = (F.softmax(roll_prob,dim=1)*idx_tensor_roll[None]).sum(dim=1)

        yaw_error = (yaw_gt-yaw_predict).abs().mean(0).item()
        pitch_error = (pitch_gt-pitch_predict).abs().mean(0).item()
        roll_error = (roll_gt-roll_predict).abs().mean(0).item()

        finger_right = (finger_probs.argmax(dim=-1)==finger).sum().item()/bb

        seg_dice = Dice_coef(seg.cpu().numpy(),fcn_out,[1,])[0]

        yaw_error_scalar.update(yaw_error,bb)
        pitch_error_scalar.update(pitch_error,bb)
        roll_error_scalar.update(roll_error,bb)
        finger_acc_scalar.update(finger_right,bb)
        seg_dice_scalar.update(seg_dice,bb)

        print(f"Eval Evil: {iterx}/{len(dataloader)}|| ",string_for_loss(["yaw","pitch","roll","finger","seg"],
                                             [yaw_error,pitch_error,roll_error,finger_right,seg_dice]))
print(f"Average eval error: yaw: {yaw_error_scalar.avg:4f},pitch: {pitch_error_scalar.avg:4f},roll: {roll_error_scalar.avg:4f}")
