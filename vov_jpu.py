import torch
import torch.nn.functional as F
import torch.nn as nn

class Conv_block(nn.Module):
    def __init__(self, inp_channel,oup_channel,kernel,str,pad):
        super(Conv_block, self).__init__()
        self.conv=nn.Conv2d(inp_channel,oup_channel,kernel,str,pad,bias=False)
        self.bn=nn.BatchNorm2d(oup_channel)
        self.relu=nn.ReLU(inplace=True)
    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        out=self.relu(x)
        return out


class OSA_block(nn.Module):
    def __init__(self, inp_channel, oup_channel, stage_channel, repeat_num=5):
        super(OSA_block, self).__init__()
        self.repeat_num=repeat_num
        self.osa_list=[]
        ori_channel=inp_channel
        for num in range(repeat_num):
            if num == 0:
                self.conv3x3=Conv_block(inp_channel,oup_channel,3,1,1)
            else:
                inp_channel=oup_channel
                self.conv3x3 = Conv_block(inp_channel, oup_channel, 3,1,1)
            self.osa_list.append(self.conv3x3)
        self.osa_list=nn.Sequential(*self.osa_list)
        self.feature1 = self.osa_list[0:1]
        self.feature2 = self.osa_list[1:2]
        self.feature3 = self.osa_list[2:3]
        self.feature4 = self.osa_list[3:4]
        self.feature5 = self.osa_list[4:]
        cat_channel = ori_channel+5*oup_channel
        self.conv1x1=Conv_block(cat_channel,stage_channel,1,1,0)

    def forward(self, x):
        x1=self.feature1(x)
        x2 = self.feature2(x1)
        x3 = self.feature3(x2)
        x4 = self.feature4(x3)
        x5 = self.feature5(x4)
        x_sum=torch.cat([x,x1,x2,x3,x4,x5],dim=1)
        out=self.conv1x1(x_sum)
        return out

class Vovnet39(nn.Module):

    def __init__(self):
        super(Vovnet39, self).__init__()
        # stage1
        self.conv3x3_1 = Conv_block(3, 64, 3, 2, 1)
        self.conv3x3_2 = Conv_block(64, 64, 3, 1, 1)
        self.conv3x3_3 = Conv_block(64, 128, 3, 1, 1)
        # stage2
        self.stage2=OSA_block(128, 64, 128)
        # stage3
        self.stage3=OSA_block(128, 80, 256)
        # stage4
        self.stage4=OSA_block(256, 96, 384)
        # stage5
        self.stage5=OSA_block(384, 112, 512)
        self.maxpool=nn.MaxPool2d(3,2,ceil_mode=True)

    def forward(self, x):
        x=self.conv3x3_1(x)
        x=self.conv3x3_2(x)
        x=self.conv3x3_3(x)# outstride=2
        x = self.maxpool(x)# outstride=4
        x=self.stage2(x)
        x=self.maxpool(x)# outstride=8
        out1=self.stage3(x)
        x = self.maxpool(out1)# outstride=16
        out2=self.stage4(x)
        x = self.maxpool(out2)# outstride=32
        out3=self.stage5(x)
        return out3,out2,out1

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = BatchNorm(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x


class JPU(nn.Module):
    def __init__(self, in_channels, width=512):
        super(JPU, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False),
                                       nn.BatchNorm2d(width),
                                       nn.ReLU(inplace=True))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='bilinear', align_corners=True)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode='bilinear', align_corners=True)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return feat


class VovJpu(nn.Module):
    def __init__(self):
        super(VovJpu, self).__init__()
        self.vovnet=Vovnet39()
        self.jpu=JPU((512,384,256))

        self.Conv_end1 = nn.Conv2d(2048,256,3,1,1,bias=False)
        self.bn_end1 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(0.1)
        self.conv_end2 = nn.Conv2d(256,3,(1,1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        input=x
        out1, out2, out3=self.vovnet(x)
        x = self.jpu(out1, out2, out3)
        x = self.Conv_end1(x)
        x = self.bn_end1(x)
        x = self.dropout(x)
        x = self.conv_end2(x)
        result = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return result

if __name__ == "__main__":
    test_inp=torch.randn((1,3,480,640)).to("cuda")
    model=VovJpu()
    model.cuda()
    out=model(test_inp)
    print(out.size())





