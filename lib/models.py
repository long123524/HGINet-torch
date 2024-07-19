import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

######
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

###
class LAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LAM, self).__init__()

        self.conv1 = BasicConv2d(in_channels//2, out_channels, 1)

        self.deconv1 = nn.Conv2d(
            in_channels, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels, in_channels // 8, (1, 9), padding=(0, 4)
        )

    def forward(self, x):

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.conv1(x)

        return x


    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)


###
# class GCN(nn.Module):
#     def __init__(self, adj_matrix, num_state, num_node, bias=False):
#         super(GCN, self).__init__()
#         self.adj_matrix = nn.Parameter(adj_matrix, requires_grad=False)  #
#         self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)
#
#     def forward(self, x):
#         # h = torch.matmul(self.adj_matrix, x)  #
#         h = self.conv1(x)
#         h = torch.matmul(self.adj_matrix, h)  #
#         h = h - x
#         h = self.relu(self.conv2(h))
#         return h


###referring to "Edge-aware Graph Representation Learning and Reasoning for Face Parsing"
class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x)
        h = h - x
        h = self.relu(self.conv2(h))
        return h


class TC(nn.Module):
    # ##
    def __init__(self, in_dim):
        super(TC, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels=in_dim// 2, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 2, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//2, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels=in_dim // 2, out_channels=in_dim, kernel_size=1)

        self.priors = nn.AdaptiveAvgPool2d(output_size=(6, 6))

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()
        q1 = self.priors(self.query_conv1(x1))[:, :, 1:-1, 1:-1].reshape(m_batchsize, self.chanel_in//2, -1).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width * height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)

        q2 = self.priors(self.query_conv2(x2))[:, :, 1:-1, 1:-1].reshape(m_batchsize, self.chanel_in//2, -1).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width * height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width * height)

        energy2 = torch.bmm(q1, k2)
        attention2 = self.softmax(energy2)
        out1 = torch.mul(v1, attention2)
        out1 = out1.view(m_batchsize, C//2, height, width)
        out1 = self.conv2(out1)

        energy1 = torch.bmm(q2, k1)
        attention1 = self.softmax(energy1)
        out2 = torch.mul(v2, attention1)
        out2 = out2.view(m_batchsize, C//2, height, width)
        out2 = self.conv1(out2)

        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2

class HGINet(nn.Module):
    def __init__(self, channel=64,num_classes=7, pretrained_path=r'D:\SCD\pretrained\pvt_v2_b2.pth', drop_rate = 0.4):
        super(HGINet, self).__init__()
        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        save_model = torch.load(pretrained_path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.channel = channel
        self.num_classes = num_classes
        self.TC_module = TC(32)

        self.Translayer2_1 = BasicConv2d(768, channel, 3,padding=1)
        self.Translayer3_1 = BasicConv2d(480, channel, 3,padding=1)
        self.Translayer4_1 = BasicConv2d(192, channel, 3,padding=1)

        self.Translayer5_1 = BasicConv2d(192, 64, 3,padding=1)
        self.Translayer6_1 = BasicConv2d(64, 32, 3,padding=1)

        self.Translayers_1 = BasicConv2d(96, 32, 3, padding=1)


        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        #
        self.out_feature = nn.Conv2d(32, num_classes, 1)
        self.out_feature2 = nn.Conv2d(32, num_classes, 1)
        self.out_feature1 = nn.Conv2d(32, 1, 1)

        self.LAM3 = LAM(512,256)
        self.LAM2 = LAM(320, 160)
        self.LAM1 = LAM(128, 64)

        self.C3 = CBAM(512)
        self.C2 = CBAM(320)
        self.C1 = CBAM(128)
        self.C0 = CBAM(64)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(32, 32 // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 // 16, 32, 1, bias=False)
        )

        # self.gcn = GCN(adj_matrix,32, 32)
        self.gcn = GCN(32, 32)

    def base_forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]
        x1 = self.drop(x1)
        x2 = pvt[1]
        x2 = self.drop(x2)
        x3 = pvt[2]
        x3 = self.drop(x3)
        x4 = pvt[3]
        x4 = self.drop(x4)
        ##level1

        cim_feature = self.C0(x1)
        ##level2, MSCFmodule
        x2_1 = self.LAM1(x2)
        x2_2 = self.C1(x2)
        x2_3 = torch.cat((x2_1,x2_2),1)
        x2_3 = self.Translayer4_1(x2_3)
        x2_4 = self.up1(x2_3)

        ###level3,MSCFmodule
        x3_1 = self.LAM2(x3)
        x3_2 = self.C2(x3)
        x3_3 = torch.cat((x3_1,x3_2),1)
        x3_3 = self.Translayer3_1(x3_3)
        x3_4 = self.up2(x3_3)

        ###level4,MSCFmodule
        x4_1 = self.LAM3(x4)
        x4_2 = self.C3(x4)
        x4_3 = torch.cat((x4_1,x4_2),1)
        x4_3 = self.Translayer2_1(x4_3)
        x4_4 = self.up3(x4_3)

        x_total= torch.cat((x2_4,x3_4,x4_4),1)
        ###SFSR module
        x_high = self.Translayer5_1(x_total)
        x_low = torch.nn.functional.softmax(cim_feature, dim=1)
        x_fuse = x_high + (x_high*x_low)
        x_fuse = self.Translayer6_1(x_fuse)
        return x_fuse

    def CD_forward(self, x1, x2):
        ###SDI module
        cz = torch.abs(x2 - x1)
        n,c,h,w = x1.size()
        x = torch.cat((x1,x2,cz),1)
        x = self.Translayers_1(x)
        x_out = x.view(n, 32, -1)
        map = self.gcn(x_out)
        maps = map.view(n, 32, *x.size()[2:])
        change = (torch.sigmoid(self.mlp(self.max_pool(x))))*maps
        change = self.out_feature1(change)
        return change

    def forward(self, t1, t2):

        t1_out = self.base_forward(t1)
        t2_out = self.base_forward(t2)
        t1_out, t2_out = self.TC_module(t1_out,t2_out)

        output1 = self.out_feature(t1_out)
        output2 = self.out_feature2(t2_out)
        change = self.CD_forward(t1_out,t2_out)

        ##output
        prediction1_1 = F.interpolate(change, scale_factor=4, mode='bilinear')
        prediction1_2 = F.interpolate(output1, scale_factor=4, mode='bilinear')
        prediction1_3 = F.interpolate(output2, scale_factor=4, mode='bilinear')

        return prediction1_1, prediction1_2, prediction1_3

