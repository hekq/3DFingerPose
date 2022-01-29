import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import torchvision.models as models

YAW_MIN,YAW_MAX     = -90,90
PITCH_MIN,PITCH_MAX = -90,0
ROLL_MIN,ROLL_MAX   = -90,90

class reArangeUpsampling(nn.Module):
    def __init__(self,ratio):
        super(reArangeUpsampling,self).__init__()
        self.ratio = ratio
    def forward(self,x):
        upsample_ratio = self.ratio
        b,c,h,w = x.shape
        new_c = int(c/(upsample_ratio**2))
        new_h = int(h*upsample_ratio)
        new_w = int(w*upsample_ratio)
        y = torch.empty(b,new_c,new_h,new_w).to(x.device)
        for i in range(0,upsample_ratio):
            for j in range(0,upsample_ratio):
                start = (i*upsample_ratio+j)*new_c
                y[:, :, i::upsample_ratio, j::upsample_ratio] = x[:, start:start+new_c, :, :]
        return y

class upsample_block(nn.Module):
    def __init__(self,in_c,out_c,act_last=True):
        super(upsample_block,self).__init__()
        # self.conv1 = nn.ConvTranspose2d(in_c,out_c,3,2,1,output_padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_c,out_c,3,1,1)
        self.bn1 = nn.BatchNorm2d(out_c)
        if act_last:
            self.relu1 = nn.GELU()
            # self.relu1 = nn.ReLU()
            # self.relu1 = nn.LeakyReLU(inplace=True)
        else: 
            self.relu1 = None

    def forward(self,x):
        x = self.bn1(self.conv1(self.up(x)))
        if self.relu1 != None:
            return self.relu1(x)
        else:
            return x

class upsample_net(nn.Module):
    def __init__(self,in_c=2048,out_c=1):
        '''upsample 32 times'''
        super(upsample_net,self).__init__()
        self.block1 = upsample_block(2048,512)
        self.block2 = upsample_block(512+2048,128)
        self.block3 = upsample_block(128+1024,256)
        self.pool = nn.AdaptiveAvgPool2d(7)
        self.block4 = upsample_block(256+512,128)
        self.block5 = upsample_block(128+)

    def forward(self,x32,x16,x8,x4,x2,x):
        for y in [x32,x16,x8,x4,x2,x]:
            print(y.shape)
        x = x.unsqueeze(2).unsqueeze(2)
        x = self.block1(x) # 2
        x = self.block2(torch.cat([x,x32],dim=1)) # 4
        x = self.block3(torch.cat([x,x16],dim=1)) # 8
        x = self.pool(x) # 7
        x = self.block4(torch.cat([x,x8],dim=1)) # 16
        return x


class MtModel(nn.Module):
    def __init__(self, block, layers, in_class ,num_bins, final_size=1, num_finger_classes=4):
        self.inplanes = 64
        self.feature_dim = 512 * block.expansion*final_size**2
        self.cluster_dim = 512

        super(MtModel, self).__init__()
        self.conv1 = nn.Conv2d(in_class, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AdaptiveAvgPool2d(final_size) # 1 * 1
        self.upsample_model = upsample_net()

        self.fc_yaw   = nn.Sequential(nn.Linear(512*block.expansion*final_size**2,512,bias=False),nn.ReLU(),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,num_bins[0]))
        self.fc_pitch = nn.Sequential(nn.Linear(512*block.expansion*final_size**2,512,bias=False),nn.ReLU(),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,num_bins[1]))
        self.fc_roll  = nn.Sequential(nn.Linear(512*block.expansion*final_size**2,512,bias=False),nn.ReLU(),nn.Linear(512,128),nn.ReLU(),nn.Linear(128,num_bins[2]))

        ### the finger type part
        self.fc_finger = nn.Linear(512*block.expansion*final_size**2,num_finger_classes)

        idx_tensor_yaw = torch.arange(YAW_MIN,YAW_MAX,(YAW_MAX-YAW_MIN)/num_bins[0])
        idx_tensor_pitch = torch.arange(PITCH_MIN,PITCH_MAX,(PITCH_MAX-PITCH_MIN)/num_bins[1])
        idx_tensor_roll = torch.arange(ROLL_MIN,ROLL_MAX,(ROLL_MAX-ROLL_MIN)/num_bins[2])
        self.register_buffer("idx_tensor_yaw",idx_tensor_yaw)
        self.register_buffer("idx_tensor_pitch",idx_tensor_pitch)
        self.register_buffer("idx_tensor_roll",idx_tensor_roll)
        self.register_buffer("yaw_related",torch.Tensor([YAW_MIN,YAW_MAX,num_bins[0]]))
        self.register_buffer("pitch_related",torch.Tensor([PITCH_MIN,PITCH_MAX,num_bins[1]]))
        self.register_buffer("roll_related",torch.Tensor([ROLL_MIN,ROLL_MAX,num_bins[2]]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_feature(self,x):
        x = self.conv1(x) # 1/2
        x = self.bn1(x)
        x = self.relu(x)
        x2 = x
        x = self.maxpool(x) # 1/4
        x = self.layer1(x)
        x4 = x
        x = self.layer2(x) # 1/8
        x8 = x
        x = self.layer3(x) # 1/16
        x16 = x
        x32 = self.layer4(x) # 1/32
        return x32,x16,x8,x4,x2

    def forward(self, x):
        x32,x16,x8,x4,x2 = self.forward_feature(x) # [B, 2048, H/32, W/32]
        x = self.avgpool(x32) # 1 1 2048
        x = x.view(x.size(0),-1) # IMPORTANT: this is the the shared feature, it's basicly the same as the downsampled_x

        finger_out = self.fc_finger(x)
        yaw_probs = self.fc_yaw(x)
        pitch_probs = self.fc_pitch(x)
        roll_probs = self.fc_roll(x)

        yaw_reg = torch.sum(F.softmax(yaw_probs,-1)*self.idx_tensor_yaw[None],-1)
        pitch_reg = torch.sum(F.softmax(pitch_probs,-1)*self.idx_tensor_pitch[None],-1)
        roll_reg = torch.sum(F.softmax(roll_probs,-1)*self.idx_tensor_roll[None],-1)

        fcn_out = self.upsample_model(x32,x16,x8,x4,x2,x) # only 1/4 = 56 size
        # output: yaw,pitch,roll,yaw_raw_probs,pitch_raw_probs,roll_raw_probs
        # the orientation should not be used in this branch because of the privacy
        result = {
            "cls": [yaw_probs,pitch_probs,roll_probs],
            "reg": [yaw_reg,pitch_reg,roll_reg],
            "seg": fcn_out,
            "finger": finger_out
        }
        return result

class Encoder(nn.Module):
    def __init__(self, layers, in_class, final_size=1):
        super(Encoder,self).__init__()
        block = models.resnet.Bottleneck
        self.inplanes = 64
        self.feature_dim = 512 * block.expansion*final_size**2
        self.conv1 = nn.Conv2d(in_class, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.AdaptiveAvgPool2d(final_size) # 1 * 1

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x) # 1/2
        x = self.bn1(x)
        x2 = self.relu(x)
        x4 = self.maxpool(x2) # 1/4

        x = self.layer1(x4)
        x8 = self.layer2(x) # 1/8
        x16 = self.layer3(x8) # 1/16
        x32 = self.layer4(x16) # 1/32
        final_x = self.avgpool(x32)
        final_x = final_x.flatten(start_dim=1) # B C
        return {"32":x32,"16":x16,"8":x8,"4":x4,"2":x2,"flatten":final_x}


class ResNet(nn.Module):
    # ResNet for regression of 3 Euler angles.
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_angles = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_angles(x)
        return x

class AlexNet(nn.Module):
    # AlexNet laid out as a Hopenet - classify Euler angles in bins and
    # regress the expected value.
    def __init__(self, num_bins):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc_yaw = nn.Linear(4096, num_bins)
        self.fc_pitch = nn.Linear(4096, num_bins)
        self.fc_roll = nn.Linear(4096, num_bins)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        yaw = self.fc_yaw(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return yaw, pitch, roll
