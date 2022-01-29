from matplotlib import use
import torch

from torch import Tensor
from torch import nn
from torch.nn import functional as F


class SmooothLabelCELoss(nn.Module):
    def __init__(self,smooth=0.1,use_uniform=False,reduction="mean"):
        super(SmooothLabelCELoss,self).__init__()
        self.smooth_coef = smooth
        self.smooth_std = 0.5
        self.reduction = reduction
        self.use_uniform = use_uniform
        self.intervals = 5
        self._loss = nn.KLDivLoss(reduction="batchmean")
        self.norm = nn.LogSoftmax(dim=1)

    def get_closest_number(self,y,C,num):
        # return: [left,right)
        assert num <= C
        half = num//2
        if y+half<C:
            left = max(y-half,0)
            right = left+num
        else:
            right = C
            left = right-num
        return left,right

    def center_smooth_label(self,target,C):
        B = target.shape[0]
        I = torch.empty(size=(B,self.intervals),device=target.device,dtype=torch.int64)
        for b in range(B):
            left,right = self.get_closest_number(target[b].item(),C,self.intervals)
            # print(f"target: {target[b]}",left,right)
            I[b,:] = torch.arange(left,right)
        softed = torch.zeros((B,C),dtype=torch.float32,device=target.device)
        use_uniform = self.use_uniform
        if use_uniform:
            softed.scatter_(1,I, self.smooth_coef*1.0/self.intervals)
            softed[torch.arange(0,B),target] = 1-self.smooth_coef+self.smooth_coef*1.0/self.intervals
        else:
            src = torch.exp(-self.smooth_std*(I - target.unsqueeze(dim=1))**2)
            softed.scatter_(1,I,src)
            softed = softed/softed.sum(1,keepdim=True)

        return softed

    def global_smooth_label(self,target,C):
        B = target.shape[0]
        if C==None:
            C = target.max()+1 # fallback
        out = torch.ones(B,C,device=target.device)*self.smooth_coef*1.0/C
        out[torch.arange(0,B),target] = 1-self.smooth_coef + self.smooth_coef*1.0/C
        return out


    def forward(self,output,label):
        # the same as CrossEntropy
        # output: is purely raw logits of the FC layer out
        output = self.norm(output) # raw logits -> probs -> log-probs
        C = output.shape[1]
        # soft_label = self.center_smooth_label(label,C)
        soft_label = self.global_smooth_label(label,C)
        loss = self._loss(output,soft_label)
        return loss
