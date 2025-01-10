import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as vmodels
import sys
sys.path.append("../../")
from workbench_utils import init_weights


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channel_num):
#         super(ResidualBlock, self).__init__()

#         conv_block = [  nn.ReflectionPad2d(1),
#                         nn.Conv2d(in_channel_num, in_channel_num, 3),
#                         nn.InstanceNorm2d(in_channel_num),
#                         nn.ReLU(inplace=True),
#                         nn.ReflectionPad2d(1),
#                         nn.Conv2d(in_channel_num, in_channel_num, 3),
#                         nn.InstanceNorm2d(in_channel_num)  ]

#         self.conv_block = nn.Sequential(*conv_block)

#     def forward(self, x):
#         return x + self.conv_block(x)
    

# class ConvGenerator(nn.Module):
#     """
#     使用的DefenseGAN的celeba_generator架构, 微调了kernel_size等一些参数
#     """
#     def __init__(self, latent_dim=256):
#         super(ConvGenerator, self).__init__()
#         self.base_size = 7
#         # self.base_channel_num = 96
#         # self.base_channel_num = 64
#         self.base_channel_num = 48
#         # self.base_channel_num = 32
#         # self.base_channel_num = 16
#         output_nc = 3

#         # Upsampling
#         fc = [
#             nn.Linear(
#                 in_features=latent_dim,
#                 out_features=16*self.base_channel_num*self.base_size*self.base_size
#             ),
#             nn.InstanceNorm1d(16*self.base_channel_num*self.base_size*self.base_size),
#             nn.ReLU(inplace=True)
#         ]
#         self.fc = nn.Sequential(*fc)

#         model = []
#         in_channel_num = 16*self.base_channel_num
#         out_channel_num = in_channel_num // 2
#         # Upsampling, 7 -> 224, 5次
#         for _ in range(5):
#             model += [  
#                 nn.ConvTranspose2d(in_channel_num, out_channel_num, 3, stride=2, padding=1, output_padding=1),
#                 nn.InstanceNorm2d(out_channel_num),
#                 nn.ReLU(inplace=True) 
#             ]
#             in_channel_num = out_channel_num
#             out_channel_num = in_channel_num//2
#             # 每个up层后面跟随3个ResidualBlock
#             model += [
#                 ResidualBlock(in_channel_num),
#                 ResidualBlock(in_channel_num),
#                 ResidualBlock(in_channel_num)
#             ]

#         # Output layer
#         model += [  
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(in_channel_num, output_nc, 7),
#             # nn.Tanh() 
#         ]

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         x = self.fc(x)
#         x = x.reshape(x.size(0), 16*self.base_channel_num, self.base_size, self.base_size)
#         # print(x.shape)
#         x = self.model(x)
#         return x
    

# class ConvDiscriminator(nn.Module):
#     """
#     使用的DefenseGAN的celeba_discriminator架构, 微调了一些参数
#     """
#     def __init__(self):
#         super(ConvDiscriminator, self).__init__()
#         # base_channel_num = 96
#         # base_channel_num = 64
#         # base_channel_num = 48
#         base_channel_num = 32
#         # base_channel_num = 16
#         input_nc = 3

#         def get_extra_conv(out_dim1):
#             return [  
#                 nn.Conv2d(out_dim1, out_dim1, 3, stride=1, padding=1),
#                 nn.InstanceNorm2d(out_dim1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Conv2d(out_dim1, out_dim1, 3, stride=1, padding=1),
#                 nn.InstanceNorm2d(out_dim1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 # nn.Conv2d(out_dim1, out_dim1, 3, stride=1, padding=1),
#                 # nn.InstanceNorm2d(out_dim1),
#                 # nn.LeakyReLU(0.2, inplace=True),
#                 # nn.Conv2d(out_dim1, out_dim1, 3, stride=1, padding=1),
#                 # nn.InstanceNorm2d(out_dim1),
#                 # nn.LeakyReLU(0.2, inplace=True) 
#             ]

#         # A bunch of convolutions one after another
#         # layer1
#         layer1 = [   
#             nn.Conv2d(input_nc, base_channel_num, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(base_channel_num),
#             nn.LeakyReLU(0.2, inplace=True) 
#         ]
#         layer1 += get_extra_conv(base_channel_num)
#         self.layer1 = nn.Sequential(*layer1)

#         # layer2
#         in_dim = base_channel_num
#         out_dim = base_channel_num*2
#         layer2 = [  
#             nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(out_dim),
#             nn.LeakyReLU(0.2, inplace=True) 
#         ]
#         layer2 += get_extra_conv(out_dim)
#         self.layer2 = nn.Sequential(*layer2)
    
#         # layer3
#         in_dim = out_dim
#         out_dim = out_dim*2
#         layer3 = [ 
#             nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(out_dim),
#             nn.LeakyReLU(0.2, inplace=True) 
#         ]
#         layer3 += get_extra_conv(out_dim)
#         self.layer3 = nn.Sequential(*layer3)
        
#         # layer4
#         in_dim = out_dim
#         out_dim = out_dim*2
#         layer4 = [  
#             nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(out_dim),
#             nn.LeakyReLU(0.2, inplace=True) 
#         ]
#         layer4 += get_extra_conv(out_dim)
#         self.layer4 = nn.Sequential(*layer4)

#         # layer5
#         in_dim = out_dim
#         out_dim = out_dim*2
#         layer5 = [  
#             nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
#             nn.InstanceNorm2d(base_channel_num*8),
#             nn.LeakyReLU(0.2, inplace=True) 
#         ]
#         layer5 += get_extra_conv(out_dim)
#         self.layer5 = nn.Sequential(*layer5)

#         # FCN classification layer
#         # last_layer = [nn.Conv2d(base_channel_num*16, 1, 4, padding=1)]
#         last_layer = [nn.Conv2d(base_channel_num*4, 1, 4, padding=1)]
#         # self.fc = nn.Linear(base_channel_num*16 * 7**2, 1)


#         self.last_layer = nn.Sequential(*last_layer)
#         self.apply(init_weights)

#     def forward(self, x):
#         x =  self.layer1(x)
#         x =  self.layer2(x)
#         x =  self.layer3(x)
#         # x =  self.layer4(x)
#         # x =  self.layer5(x)
#         x =  self.last_layer(x)
#         # print(f"x的shape: {x.shape}")
#         x = F.avg_pool2d(input=x, kernel_size=x.size()[2:])
#         x = x.view(x.size()[0], -1)
#         # x = self.fc(x)
#         return x


# def get_extra_conv_g(in_dim):
#     seq = nn.Sequential(
#         ResidualBlock(in_dim),
#         ResidualBlock(in_dim),
#         # ResidualBlock(in_dim)
#     )
#     return seq


# def get_extra_conv_d(in_dim):
#     seq = nn.Sequential(
#         ResidualBlock(in_dim),
#         ResidualBlock(in_dim),
#         ResidualBlock(in_dim)
#     )
#     return seq


class ConvGenerator(nn.Module):
    """
    使用的DefenseGAN的celeba_generator架构, 微调了kernel_size等一些参数
    """
    def __init__(self, latent_dim=256):
        super(ConvGenerator, self).__init__()
        # self.net_dim = 256
        # self.net_dim = 128
        self.net_dim = 96
        # self.net_dim = 64
        # self.net_dim = 48
        # self.net_dim = 32
        self.init_size = 4
        # norm_fuction_1 = nn.InstanceNorm1d
        norm_fuction_2 = nn.InstanceNorm2d
        # norm_fuction_1 = nn.BatchNorm1d
        # norm_fuction_2 = nn.BatchNorm2d

        # layer0
        self.fc = nn.Linear(latent_dim, 4 * self.net_dim * self.init_size ** 2)
        self.norm = norm_fuction_2(4 * self.net_dim)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.up = nn.Upsample(scale_factor=2)

        # layer1
        self.conv1 = nn.Conv2d(
            in_channels=4 * self.net_dim,
            out_channels=4 * self.net_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = norm_fuction_2(4*self.net_dim)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.up1 = nn.Upsample(scale_factor=2)

        # layer2
        self.conv2 = nn.Conv2d(
            in_channels=4 * self.net_dim,
            out_channels=2 * self.net_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = norm_fuction_2(2 * self.net_dim)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.up2 = nn.Upsample(scale_factor=2)

        # layer3
        self.conv3 = nn.Conv2d(
            in_channels=2 * self.net_dim,
            out_channels=self.net_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm3 = norm_fuction_2(self.net_dim)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.up3 = nn.Upsample(scale_factor=2)

        # layer4
        self.conv4 = nn.Conv2d(
            in_channels=self.net_dim,
            out_channels=3,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # self.out = nn.Sigmoid()
        self.out = nn.Tanh()

    def forward(self, noise):
        x = self.fc(noise)
        x = x.reshape(-1, 4*self.net_dim, self.init_size, self.init_size)
        x = self.norm(x)
        x = self.relu(x)
        x = self.up(x)
        # [B, 4*self.net_dim, init_size, init_size]
        # print(x.shape)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.up1(x)
        # [B, 4*self.net_dim, 2*init_size, 2*init_size]
        # print(x.shape)

        x = self.conv2(x)  
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.up2(x)
        # [B, 2*self.net_dim, 4*init_size, 4*init_size]
        # print(x.shape)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.up3(x)
        # [B, self.net_dim, 8*init_size, 8*init_size]
        # print(x.shape)

        x = self.conv4(x)
        # [B, 3, 16*init_size, 16*init_size]
        # print(x.shape)

        x = self.out(x)
        return x


class ConvDiscriminator(nn.Module):
    """
    使用的DefenseGAN的celeba_discriminator架构, 微调了一些参数
    """
    def __init__(self):
        super(ConvDiscriminator, self).__init__()
        # WGAN的判别器不能使用batch norm
        # self.net_dim = 256
        # self.net_dim = 128
        self.net_dim = 96
        # self.net_dim = 64
        self.img_size = 64
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)

        def discriminator_block(in_filters, out_filters, norm_type="none"):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if norm_type == "bn":
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            elif norm_type == "in":
                block.append(nn.InstanceNorm2d(out_filters))
            return block

        self.layer1 = nn.Sequential(
            *discriminator_block(3, self.net_dim, norm_type="in")
        )
        self.layer2 = nn.Sequential(
            *discriminator_block(self.net_dim, 2*self.net_dim, norm_type="in")
        )
        self.layer3 = nn.Sequential(
            *discriminator_block(2*self.net_dim, 4*self.net_dim, norm_type="in")
        )
        self.layer4 = nn.Sequential(
            *discriminator_block(4*self.net_dim, 8*self.net_dim, norm_type="in")
        )

        self.last_feats = 8*self.net_dim * ((self.img_size // (2**4)) ** 2)
        self.linear = nn.Linear(self.last_feats, 1)

        self.out = nn.Sigmoid()   # 是否移除?

    def forward(self, x):
        # print(x.shape)  
        x = self.layer1(x)
        # [2, net_dim, img_size//2, img_size//2]
        # print(x.shape)

        x = self.layer2(x)
        # [2, 2*net_dim, img_size//4, img_size//4]
        # print(x.shape)

        x = self.layer3(x)
        # [2, 4*net_dim, img_size//8, img_size//8]
        # print(x.shape)

        x = self.layer4(x)
        # [2, 8*net_dim, img_size//16, img_size//16]
        # print(x.shape)

        x = x.reshape(-1, self.last_feats)
        x = self.linear(x)
        x = self.out(x)
        return x


# from torchinfo import summary
# g = ConvGenerator()
# a = torch.randn((2, 256))
# summary(
#     g,
#     input_data=a
# )

# d = ConvDiscriminator()
# # a = torch.randn((2, 3, 224, 224))
# a = torch.randn((2, 3, 64, 64))
# # b = d(a)
# # print(b.shape)
# summary(
#     d,
#     input_data=a
# )
