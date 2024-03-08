from led.utils.registry import ARCH_REGISTRY
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_base
from lite_isp import process




class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dwconv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.dwconv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class DownSample(nn.Module):
    def __init__(self, chan) -> None:
        super().__init__()
        self.downconv = nn.Conv2d(chan, 2*chan, 2, 2)

    def forward(self, x):
        return self.downconv(x)


class UpSample(nn.Module):
    def __init__(self, chan) -> None:
        super().__init__()
        self.upconv = nn.Conv2d(chan, chan * 2, 1, bias=False)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.pixel_shuffle(self.upconv(x))



@ARCH_REGISTRY.register()
class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],channels = 32):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                DownSample(chan)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                UpSample(chan)
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
        self.concatconv = nn.Conv2d(channels * 32, channels * 16, kernel_size=3, stride=1, padding=1)
        self.padder_size = 2 ** len(self.encoders)
        pretrained_convnext = convnext_base(pretrained=True)
        self.Convnext_branch = nn.Sequential(
            pretrained_convnext.features[0],
            pretrained_convnext.features[1],
            pretrained_convnext.features[2],
            pretrained_convnext.features[3],
            pretrained_convnext.features[4]
                            )
        for param in self.Convnext_branch.parameters():
            param.requires_grad = False

    def Rgb(self,lq,ccm,wb):
        return process(lq,wb,ccm,gamma=2.2)

    def piror_branch(self,x):
        # x1 = torch.rand(1,3,1024,1024)
        # y = self.Convnext_branch(x1)
        print(self.Convnext_branch)
        return(self.Convnext_branch(x))

    def forward(self, x):
        ccm = x[1]
        wb = x[2]
        rgb = self.Rgb(x[0],ccm,wb)
        rgb = self.piror_branch(rgb)
        # print(rgb.shape)
        inp = x[0]
        B, C, H, W = inp.shape
        # print(inp.shape)
        inp = self.check_image_size(inp)
        # print(inp.shape)
        x = self.intro(inp)
        #print(x.shape)
        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        t = torch.cat([x, rgb], 1)
        x = self.lrelu(self.concatconv(t))
        x = self.middle_blks(x)


        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]
    def lrelu(self, x):
        outt = torch.max(0.2 * x, x)
        return outt
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x






# pretrained_convnext = convnext_base(pretrained=True)
# #Convnext_branch = nn.Sequential(*list(pretrained_convnext.children())[:1])  # 以ConvNeXt的前5个层为例
# Convnext_branch = nn.Sequential(
#     pretrained_convnext.features[0],
#     pretrained_convnext.features[1],
#     pretrained_convnext.features[2],
#     pretrained_convnext.features[3]


# )
# print(Convnext_branch)
# x = torch.rand(1,3,512,512)
# y = Convnext_branch(x)
# print(y.shape)
