#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Subset
from torch.utils.tensorboard.writer import SummaryWriter
import torchvision
import torch.utils.model_zoo as model_zoo
import numpy as np
import os
import sys
import time
from datetime import datetime
import random
import string
import yaml
import shutil
import matplotlib.pyplot as plt
from lr_scheduler import build_scheduler

# ==== project
from build_models import UltraModel
from losses.focal_loss import FocalLoss
from losses.label_smooth_ce_loss import SmooothLabelCELoss
from losses.dice_loss import AnotherBinaryDice
from utils import dice as Dice_coef
from dataset import fingerset
from utils import save_model,averageArray,averageScalar,generate_random_seed,string_for_loss,fig2img
from config import get_config
from argparse import ArgumentParser

class Trainer():
    def __init__(self):
        args = self.parse_args()
        self.config = get_config(args)
        self.set_seed()
        self.make_dataset()
        self.make_model()
        self.interval_tensors()
        now = datetime.now()
        self.prefix = now.strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs(os.path.join('./models',self.prefix),exist_ok=True)
        os.makedirs(os.path.join('./runs',self.prefix),exist_ok=True)
        with open(os.path.join('./runs',self.prefix,'config.yaml'),"w") as f:
            self.config.dump(stream=f)
        self.writer = SummaryWriter(f'./runs/{self.prefix}')

    def parse_args(self,):
        parser = ArgumentParser()
        parser.add_argument("--cfg",type=str,default='',required=True)
        parser.add_argument("--opts",type=str,default=None,nargs="+",required=False)
        return parser.parse_args()

    def make_dataset(self):
        trainset = fingerset('./pkls/train.pkl','train',[64,64],2,self.config.DATA.NORMALIZATION)
        evalset  = fingerset('./pkls/eval.pkl', 'eval', [64,64],2,self.config.DATA.NORMALIZATION)
        testset  = fingerset('./pkls/test.pkl', 'test', [64,64],2,self.config.DATA.NORMALIZATION)
        # TODO
        debug = False
        if debug:
            trainset = Subset(trainset,list(range(0,100)))
            evalset = Subset(evalset,list(range(0,100)))
        print("len of train: ",len(trainset))
        print("len of val: ",len(evalset))
        print("len of test: ",len(testset))
        self.trainloader = DataLoader(dataset=trainset,batch_size=self.config.DATA.BATCH_SIZE,shuffle=True,num_workers=self.config.DATA.NUM_WORKERS,pin_memory=True)
        self.evalloader  = DataLoader(dataset=evalset,batch_size=self.config.DATA.BATCH_SIZE,shuffle=False,num_workers=self.config.DATA.NUM_WORKERS,pin_memory=True)
        self.testloader  = DataLoader(dataset=testset,batch_size=self.config.DATA.BATCH_SIZE,shuffle=False,num_workers=self.config.DATA.NUM_WORKERS,pin_memory=True)
        self.n_iter_per_epoch = len(self.trainloader)

    def make_model(self,):
        self.model = UltraModel(1,self.config.MODEL.NUM_CLASSES,2,self.config.MODEL.BACKBONE).cuda()
        # self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config.TRAIN.BASE_LR)
        self.optimizer = torch.optim.Adam([{'params':self.model.encoder.parameters(),'lr':5e-6},
            {'params':self.model.decoder.parameters(),'lr':1e-3},
            {'params':self.model.segmentation_head.parameters(),'lr':1e-3},
            {'params':self.model.yawHead.parameters(),'lr':self.config.TRAIN.BASE_LR},
            {'params':self.model.pitchHead.parameters(),'lr':self.config.TRAIN.BASE_LR},
            {'params':self.model.rollHead.parameters(),'lr':self.config.TRAIN.BASE_LR},
            {'params':self.model.fingerHead.parameters(),'lr':self.config.TRAIN.BASE_LR}],lr=self.config.TRAIN.BASE_LR)
        self.lr_scheduler = build_scheduler(self.config,self.optimizer,self.n_iter_per_epoch)
        self.model = nn.DataParallel(self.model,list(range(0,len(os.environ["CUDA_VISIBLE_DEVICES"].split(',')))))
        if self.config.TRAIN.CLS_LOSS=="focal":
            self.cls_criterion = FocalLoss(gamma=2.0).cuda()
        elif self.config.TRAIN.CLS_LOSS=="ce":
            self.cls_criterion = nn.CrossEntropyLoss().cuda()
        elif self.config.TRAIN.CLS_LOSS=="bce":
            self.cls_criterion = nn.BCEWithLogitsLoss().cuda()
        else: # use my label smooth loss
            print("using my label smooth loss")
            self.cls_criterion = SmooothLabelCELoss().cuda()

        self.finger_criterion = SmooothLabelCELoss().cuda()
        if self.config.TRAIN.FCN_LOSS=="dice":
            print("using dice loss")
            self.fcn_criterion = AnotherBinaryDice().cuda()
        else:
            print('using bce for seg')
            self.fcn_criterion = nn.BCEWithLogitsLoss().cuda()

        if self.config.TRAIN.REG_LOSS=="l1":
            self.reg_criterion = nn.L1Loss().cuda()
        else:
            self.reg_criterion = nn.MSELoss().cuda()

    def load_config_file(self,config_file):
        return yaml.safe_load(open(config_file,"r"))

    def set_seed(self,random_seed=False):
        if random_seed:
            seed = generate_random_seed()
        else:
            seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.seed = seed

    def run(self):
        # start_epoch = 0 if "start_epoch" not in self.__dict__ else self.start_epoch
        start_epoch = self.config.TRAIN.START_EPOCH
        best_error = np.inf
        for epoch in range(start_epoch,start_epoch+self.config.TRAIN.EPOCHS):
            avg_loss = self.train(epoch)
            eval_error = self.evaluate(epoch,"validation",self.evalloader)
            if epoch % 10==0:
                save_model(self.model.module,self.optimizer,None,epoch,eval_error,os.path.join(self.config.TRAIN.SAVE_DIR,self.prefix,f'{epoch}.pth.tar'))
            if eval_error<best_error:
                save_model(self.model.module,self.optimizer,None,epoch,eval_error,os.path.join(self.config.TRAIN.SAVE_DIR,self.prefix,"best.pth.tar"))
                best_error = eval_error

        self.evaluate(self.config.TRAIN.EPOCHS,"test",self.testloader)
        self.writer.close()

    def interval_tensors(self):
        def length_arange(min,max,length):
            return torch.arange(min,max,float(max-min)/length)
        self.idx_tensor_yaw = length_arange(self.config.MODEL.YAW_MIN,self.config.MODEL.YAW_MAX,self.config.MODEL.NUM_CLASSES[0]).cuda()
        self.idx_tensor_pitch = length_arange(self.config.MODEL.PITCH_MIN,self.config.MODEL.PITCH_MAX,self.config.MODEL.NUM_CLASSES[1]).cuda()
        self.idx_tensor_roll = length_arange(self.config.MODEL.ROLL_MIN,self.config.MODEL.ROLL_MAX,self.config.MODEL.NUM_CLASSES[2]).cuda()

    def train(self,epoch):
        self.model.train()
        loss_scalar = averageScalar()
        cls_loss_scalar = averageScalar()
        reg_loss_scalar = averageScalar()
        fcn_loss_scalar = averageScalar()
        finger_loss_scalar = averageScalar()

        self.optimizer.zero_grad()
        for iterx, item in enumerate(self.trainloader):
            img = item["img"].cuda()
            yaw_gt   = item['yaw'].cuda().to(torch.float32)
            pitch_gt = item["pitch"].cuda().to(torch.float32)
            roll_gt  = item["roll"].cuda().to(torch.float32)
            finger = item['finger_type'].cuda()
            gt = torch.stack([yaw_gt,pitch_gt,roll_gt],dim=-1)
            seg = item['seg'].cuda()
            bb = img.shape[0]

            yaw_label,pitch_label,roll_label = generate_label(yaw_gt,pitch_gt,roll_gt,self.config.MODEL.NUM_CLASSES,self.config)
            result = self.model(img)
            yaw_prob = result["yaw"]
            pitch_prob = result["pitch"]
            roll_prob = result["roll"]
            finger_probs = result['finger']
            fcn_out = result['fcn']

            # yaw_prob shape: [B,C]
            if self.config.TRAIN.CLS_LOSS=="bce":
                yaw_predict = torch.sum(torch.sigmoid(yaw_prob)*self.idx_tensor_yaw[None],dim=1)/self.config.TRAIN.NUM_CLASSES[0]
                pitch_predict = torch.sum(torch.sigmoid(pitch_prob)*self.idx_tensor_pitch[None],dim=1)/self.config.TRAIN.NUM_CLASSES[1]
                roll_predict = torch.sum(torch.sigmoid(roll_prob)*self.idx_tensor_roll[None],dim=1)/self.config.TRAIN.NUM_CLASSES[2]
            else:
                yaw_predict = (F.softmax(yaw_prob,dim=1)*self.idx_tensor_yaw[None]).sum(dim=1)
                pitch_predict = (F.softmax(pitch_prob,dim=1)*self.idx_tensor_pitch[None]).sum(dim=1)
                roll_predict = (F.softmax(roll_prob,dim=1)*self.idx_tensor_roll[None]).sum(dim=1)

            loss_cls = self.cls_criterion(yaw_prob,yaw_label) + \
                    self.cls_criterion(pitch_prob,pitch_label) + \
                    self.cls_criterion(roll_prob,roll_label)

            loss_reg = self.reg_criterion(yaw_predict,yaw_gt) + \
                    self.reg_criterion(pitch_predict,pitch_gt) + \
                    self.reg_criterion(roll_predict,roll_gt)

            loss_finger = self.finger_criterion(finger_probs,finger)
            loss_fcn = self.fcn_criterion(fcn_out,seg)

            loss = self.config.TRAIN.LOSS_RATIOS.SEG * loss_fcn + \
                self.config.TRAIN.LOSS_RATIOS.REG * loss_reg + \
                self.config.TRAIN.LOSS_RATIOS.CLS * loss_cls + \
                self.config.TRAIN.LOSS_RATIOS.FIN * loss_finger

            print(f"{iterx}(of {len(self.trainloader)})||{epoch}\t",string_for_loss(["loss","loss_cls","loss_reg","loss_finger","loss_fcn"],
                                  [loss,  loss_cls,   loss_reg,  loss_finger,  loss_fcn]))

            loss_scalar.update(loss.item(),bb)
            cls_loss_scalar.update(loss_cls.item(),bb)
            reg_loss_scalar.update(loss_reg.item(),bb)
            finger_loss_scalar.update(loss_finger.item(),bb)
            fcn_loss_scalar.update(loss_fcn.item(),bb)

            self.writer.add_scalar("train_loss_all",loss_scalar.val,epoch*self.n_iter_per_epoch+iterx)
            self.writer.add_scalar("train_loss_cls",cls_loss_scalar.val,epoch*self.n_iter_per_epoch+iterx)
            self.writer.add_scalar("train_loss_reg",reg_loss_scalar.val,epoch*self.n_iter_per_epoch+iterx)
            self.writer.add_scalar("train_loss_finger",finger_loss_scalar.val,epoch*self.n_iter_per_epoch+iterx)
            self.writer.add_scalar("train_loss_seg",fcn_loss_scalar.val,epoch*self.n_iter_per_epoch+iterx)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step_update(epoch*self.n_iter_per_epoch+iterx)
            # self.writer.add_scalar("train_lr", max(self.lr_scheduler.get_update_values(epoch*self.n_iter_per_epoch+iterx)), epoch*self.n_iter_per_epoch+iterx)
            self.writer.add_scalar("train_lr", self.optimizer.param_groups[-1]['lr'], epoch*self.n_iter_per_epoch+iterx)

        self.writer.add_scalar("epoch_loss_all",loss_scalar.avg,epoch)
        self.writer.add_scalar("epoch_loss_cls",cls_loss_scalar.avg,epoch)
        self.writer.add_scalar("epoch_loss_reg",reg_loss_scalar.avg,epoch)
        self.writer.add_scalar("epoch_loss_finger",finger_loss_scalar.avg,epoch)
        self.writer.add_scalar("epoch_loss_seg",fcn_loss_scalar.avg,epoch)
        return loss_scalar.avg

    def evaluate(self,epoch,context,dataloader):
        if context=="test":
            print("testing====")
            best_ckp = torch.load(os.path.join(self.config.TRAIN.SAVE_DIR,self.prefix,"best.pth.tar"),map_location=lambda storage,_:storage)
            self.model.module.load_state_dict(best_ckp["model"])

        self.model.eval()
        yaw_error_scalar = averageScalar()
        pitch_error_scalar = averageScalar()
        roll_error_scalar = averageScalar()
        finger_acc_scalar = averageScalar()
        seg_dice_scalar = averageScalar()

        for iterx, item in enumerate(dataloader):
            with torch.no_grad():
                img = item["img"].cuda()
                yaw_gt   = item['yaw'].cuda().to(torch.float32)
                pitch_gt = item["pitch"].cuda().to(torch.float32)
                roll_gt  = item["roll"].cuda().to(torch.float32)
                gt = torch.stack([yaw_gt,pitch_gt,roll_gt],dim=-1)
                finger = item['finger_type'].cuda()
                seg = item['seg'].cuda()
                bb = img.shape[0]

                result = self.model(img)
                yaw_prob = result["yaw"]
                pitch_prob = result["pitch"]
                roll_prob = result['roll']
                finger_probs = result['finger']
                fcn_out = result['fcn']
                fcn_out = torch.sigmoid(fcn_out)
                fcn_out = (fcn_out>0.75).cpu().numpy().astype(np.int32)

                if iterx % 10 ==0:
                    fig,axes = plt.subplots(1,3)
                    axes[0].imshow(img[0,0].cpu().numpy(),cmap='gray')
                    axes[1].imshow(seg[0,0].cpu().numpy(),cmap='gray')
                    axes[2].imshow(fcn_out[0,0],cmap='gray')
                    # fig = fig2img(fig) # convert to PIL image for logging

                    suffix = f"{context}_{epoch}_{iterx}"
                    self.writer.add_figure("segmentation_"+suffix,fig)

                if self.config.TRAIN.CLS_LOSS=="bce":
                    yaw_predict = torch.sum(torch.sigmoid(yaw_prob)*self.idx_tensor_yaw[None],dim=1)/self.config.MODEL.NUM_CLASSES[0]
                    pitch_predict = torch.sum(torch.sigmoid(pitch_prob)*self.idx_tensor_pitch[None],dim=1)/self.config.MODEL.NUM_CLASSES[1]
                    roll_predict = torch.sum(torch.sigmoid(roll_prob)*self.idx_tensor_roll[None],dim=1)/self.config.MODEL.NUM_CLASSES[2]
                else:
                    yaw_predict = (F.softmax(yaw_prob,dim=1)*self.idx_tensor_yaw[None]).sum(dim=1)
                    pitch_predict = (F.softmax(pitch_prob,dim=1)*self.idx_tensor_pitch[None]).sum(dim=1)
                    roll_predict = (F.softmax(roll_prob,dim=1)*self.idx_tensor_roll[None]).sum(dim=1)

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

                print(f"Eval Evil: {iterx}/{len(dataloader)}||{epoch} ",string_for_loss(["yaw","pitch","roll","finger","seg"],
                                                     [yaw_error,pitch_error,roll_error,finger_right,seg_dice]))
        print(f"Average eval error: {yaw_error_scalar.avg:4f},{pitch_error_scalar.avg:4f},{roll_error_scalar.avg:4f}")
        self.writer.add_scalar("eval_yaw_error",yaw_error_scalar.avg,epoch)
        self.writer.add_scalar("eval_pitch_error",pitch_error_scalar.avg,epoch)
        self.writer.add_scalar("eval_roll_error",roll_error_scalar.avg,epoch)
        self.writer.add_scalar("eval_finger_acc",finger_acc_scalar.avg,epoch)
        self.writer.add_scalar("eval_seg_dice",seg_dice_scalar.avg,epoch)
        return (yaw_error_scalar.avg+pitch_error_scalar.avg+roll_error_scalar.avg)/3.0

def get_frozen_parameters(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            # if 'bn' in module_name:
            #     module.eval()
            for name, param in module.named_parameters():
                yield param

def get_slow_updating_parameters(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            # if 'bn' in module_name:
            #     module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fcn_parameters(model):
    b = [model.upsample_model]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            # if 'bn' in module_name:
            #     module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    print("length of snapshot dict ",len(snapshot))
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict and model_dict[k].shape==v.shape}
    print("length of pretained resnet weights:\t",len(list(snapshot.keys())))
    print("length of model_dict ",len(model_dict))
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)

def generate_label(yaw,pitch,roll,num_intervals,config):
    one_hot = True if config.TRAIN.CLS_LOSS=="bce" else False
    dev = yaw.device
    bb = yaw.shape[0]
    yaw_id = torch.div((yaw - config.MODEL.YAW_MIN),((config.MODEL.YAW_MAX-config.MODEL.YAW_MIN)/num_intervals[0]),rounding_mode='floor')
    yaw_id = yaw_id.clamp(max=num_intervals[0]-1)
    pitch_id = torch.div((pitch - config.MODEL.PITCH_MIN), ((config.MODEL.PITCH_MAX-config.MODEL.PITCH_MIN)/num_intervals[1]), rounding_mode='floor')
    pitch_id = pitch_id.clamp(max=num_intervals[1]-1)
    roll_id = torch.div((roll - config.MODEL.ROLL_MIN), ((config.MODEL.ROLL_MAX-config.MODEL.ROLL_MIN)/num_intervals[2]), rounding_mode='floor')
    roll_id = roll_id.clamp(max=num_intervals[2]-1)
    if not one_hot:
        return yaw_id.to(torch.long).to(dev),pitch_id.to(torch.long).to(dev),roll_id.to(torch.long).to(dev)
    else:
        yaw_id_hot = torch.zeros(bb,num_intervals[0]).to(dev)
        yaw_id_hot.scatter_(1,yaw_id[:,None].to(torch.long),0.9)

        pitch_id_hot = torch.zeros(bb,num_intervals[1]).to(dev)
        pitch_id_hot.scatter_(1,pitch_id[:,None].to(torch.long),0.9)

        roll_id_hot = torch.zeros(bb,num_intervals[2]).to(dev)
        roll_id_hot.scatter_(1,roll_id[:,None].to(torch.long),0.9)
        return yaw_id_hot,pitch_id_hot,roll_id_hot

if __name__ == '__main__':
    t = Trainer()
    t.run()
