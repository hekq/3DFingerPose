import segmentation_models_pytorch as smp
import torch.nn as nn
import torch

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=nn.LeakyReLU):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = nn.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear1 = nn.Linear(in_channels, int(in_channels//4), bias=True)
        activation1 = smp.base.modules.Activation(activation)

        linear2 = nn.Linear(int(in_channels//4), int(in_channels//16), bias=True)
        activation2 = smp.base.modules.Activation(activation)

        linear3 = nn.Linear(int(in_channels//16), classes, bias=True)
        super().__init__(pool, flatten, dropout, linear1, activation1, linear2, activation2, linear3)

class modulate_block(nn.Sequential):
    def __init__(self,in_c,out_c):
        # conv bn relu
        pool = nn.AdaptiveAvgPool2d(7)
        conv = nn.Conv2d(in_c,out_c,kernel_size=1+2,padding=1)
        bn = nn.BatchNorm2d(out_c)
        act = nn.ReLU()
        super().__init__(pool,conv,bn,act)


class UltraModel(nn.Module):
    def __init__(self,in_c,pose_c,finger_c,fcn_c=1):
        super(UltraModel,self).__init__()
        self.base = smp.Unet(
                encoder_name="mobilenet_v2",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                in_channels=in_c,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=fcn_c,                      # model output channels (number of classes in your dataset)
                )
        self.encoder = self.base.encoder
        self.decoder = self.base.decoder
        self.segmentation_head = self.base.segmentation_head
        # attention across different stages, uncomment if you want to try it
        # self.modulators = nn.ModuleList([modulate_block(64,512), modulate_block(128,512), modulate_block(256,512)])
        # yaw_attn = torch.tensor([0.1, 0.2, 0.4]).to(torch.float32)
        # pitch_attn = torch.tensor([0.1, 0.2, 0.4]).to(torch.float32)
        # roll_attn = torch.tensor([0.1, 0.2, 0.4]).to(torch.float32)
        # yaw_attn = nn.Parameter(yaw_attn,requires_grad=True)
        # pitch_attn = nn.Parameter(pitch_attn,requires_grad=True)
        # roll_attn = nn.Parameter(roll_attn,requires_grad=True)
        # self.register_parameter(name='yaw_attn', param=yaw_attn)
        # self.register_parameter(name='pitch_attn', param=pitch_attn)
        # self.register_parameter(name='roll_attn', param=roll_attn)

        print('pose_c',pose_c)
        self.yawHead = ClassificationHead(self.encoder.out_channels[-1], pose_c[0])
        self.pitchHead = ClassificationHead(self.encoder.out_channels[-1], pose_c[1])
        self.rollHead = ClassificationHead(self.encoder.out_channels[-1], pose_c[2])
        self.fingerHead = ClassificationHead(self.encoder.out_channels[-1], finger_c)

    def forward(self,x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        fcn_out = self.segmentation_head(decoder_output)
        # features: 6 layers, from 224,112,56,28,14,7
        # torch.Size([1, 1, 224, 224])
        # torch.Size([1, 64, 112, 112])
        # torch.Size[1, 64, 56, 56])
        # torch.Size([1, 128, 28, 28])
        # torch.Size([1, 256, 14, 14])
        # torch.Size([1, 512, 7, 7])

        # yaw_feature = 0
        # yaw_attn = torch.softmax(self.yaw_attn,0)
        # pitch_attn = torch.softmax(self.pitch_attn,0)
        # roll_attn = torch.softmax(self.roll_attn,0)
        # for stage in range(3):
        #     yaw_feature = yaw_feature + yaw_attn[stage] * self.modulators[stage](features[2+stage])
        # yaw_feature = yaw_feature + features[-1]
        #
        # pitch_feature = 0
        # for stage in range(3):
        #     pitch_feature = pitch_feature + pitch_attn[stage] * self.modulators[stage](features[2+stage])
        # pitch_feature = pitch_feature + features[-1]
        #
        # roll_feature = 0
        # for stage in range(3):
        #     roll_feature = roll_feature + roll_attn[stage] * self.modulators[stage](features[2+stage])
        # roll_feature = roll_feature + features[-1]

        yaw_probs = self.yawHead(features[-1])
        pitch_probs = self.pitchHead(features[-1])
        roll_probs = self.rollHead(features[-1])
        finger = self.fingerHead(features[-1])
        return {"fcn":fcn_out,"yaw":yaw_probs,"pitch":pitch_probs,"roll":roll_probs,"finger":finger}

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x

class ConvBlock(nn.Sequential):
    def __init__(self,in_c,out_c,stride):
        # kernel_size = 2*padding + stride
        # pro = nn.Tanh()

        conv1 = nn.Conv2d(in_c,in_c*4,kernel_size=4+stride,stride=stride,padding=2,bias=False)
        bn1 = nn.BatchNorm2d (in_c*4)
        ac1 = nn.LeakyReLU(0.2, inplace=True)
        d1 = nn.Dropout(0.1)
        pool = nn.AdaptiveAvgPool2d(1)
        flatten = nn.Flatten()

        linear1 = nn.Linear(in_c*4, in_c, bias=False)
        bnL1 = nn.BatchNorm1d(in_c)
        activation1 = nn.LeakyReLU(0.2, inplace=True)
        dL1 = nn.Dropout(0.1)

        linear2 = nn.Linear(in_c, in_c//4, bias=False)
        bnL2 = nn.BatchNorm1d(in_c//4)
        activation2 = nn.LeakyReLU(0.2, inplace=True)
        dL2 = nn.Dropout(0.1)

        linear3 = nn.Linear(int(in_c//4), out_c, bias=True)
        super().__init__(conv1, bn1, ac1, d1, pool, flatten, 
                linear1, bnL1, activation1,dL1,
                linear2, bnL2, activation2,dL2, linear3)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.disc1 = ConvBlock(64,1,8)
        self.disc2 = ConvBlock(64,1,4)
        self.disc3 = ConvBlock(128,1,2)
        self.disc4 = ConvBlock(256,1,1)
        self.disc5 = ClassificationHead(512,1)
        self.disc_all = [self.disc1,self.disc2,self.disc3,self.disc4,self.disc5]
    def forward(self,features):
        scores = []
        for jj,e in enumerate(features):
            scores.append(self.disc_all[jj](e))
        return scores

if __name__ == "__main__":
    dev = torch.device("cuda:0")
    input = torch.randn(4,1,64,64).to(dev)
    m = UltraModel(1,1).to(dev)
    pose = m(input)['pose']
    seg = m(input)['fcn_out']
    print(pose.shape)
    print(seg.min(),seg.max(),seg.shape)
    from utils import count_parameters
    print(count_parameters(m))
