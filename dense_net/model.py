import os#, psutil
from torch import nn
import torch

class unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(unet, self).__init__()
        # collection
        # self.conv_in = self.single_conv_in(1, 16, 3)
        # self.convs1_L = self.conv_block(16, 16 ,3)
        # self.down_conv1 = self.down_conv(16, 32, 3, 3)
        # self.convs2_L = self.conv_block(32, 32, 3)
        # self.down_conv2 = self.down_conv(32, 64, 3, 3)
        # self.convs3_L = self.conv_block(64, 64, 3, padding=[(1,1), (1,2)])
        # self.down_conv3 = self.down_conv(64, 128, 3, 3)
        # self.convs4_L = self.conv_block(128, 128, 3, padding=[(1,1),(1,1)])

        # self.down_conv_bottom = self.down_conv(128, 256, 3, 3)
        # self.convs_bottom = self.conv_block(256, 256, 3, padding=[(1,1),(1,1)])
        # self.up_conv_bottom = self.up_conv(256, 128, 3, 3)

        # self.convs4_R = self.conv_block(128*2, 128, 3)
        # self.up_conv1 = self.up_conv(128, 64, 3, 3, output_padding=(0,1))
        # self.convs3_R = self.conv_block(64*2, 64, 3, padding=[(1,1),(1,0)])
        # self.up_conv2 = self.up_conv(64, 32, 3, 3, output_padding=(2,1))
        # self.convs2_R = self.conv_block(32*2, 32, 3)
        # self.up_conv3 = self.up_conv(32, 16, 3, 3)
        # self.convs1_R = self.conv_block(16*2, 16, 3)
        # self.conv_out = self.single_conv_out(16, 1, 3)

        #induction
        self.conv_in = self.single_conv_in(1, 16, 3)
        self.convs1_L = self.conv_block(16, 16 ,3)
        self.down_conv1 = self.down_conv(16, 32, 3, 3)
        self.convs2_L = self.conv_block(32, 32, 3)
        self.down_conv2 = self.down_conv(32, 64, 3, 3)
        self.convs3_L = self.conv_block(64, 64, 3, padding=[(1,0), (1,0)])
        self.down_conv3 = self.down_conv(64, 128, 3, 3)
        self.convs4_L = self.conv_block(128, 128, 3, padding=[(1,0),(1,0)])

        self.down_conv_bottom = self.down_conv(128, 256, 3, 3)
        self.convs_bottom = self.conv_block(256, 256, 3, padding=[(1,2),(1,0)])
        self.up_conv_bottom = self.up_conv(256, 128, 3, 3, output_padding=(0,1))

        self.convs4_R = self.conv_block(128*2, 128, 3, padding=[(1,2),(1,2)])
        self.up_conv1 = self.up_conv(128, 64, 3, 3, output_padding=(0,0))
        self.convs3_R = self.conv_block(64*2, 64, 3, padding=[(1,2),(1,2)])
        self.up_conv2 = self.up_conv(64, 32, 3, 3, output_padding=(2,1))
        self.convs2_R = self.conv_block(32*2, 32, 3)
        self.up_conv3 = self.up_conv(32, 16, 3, 3, output_padding=(0,2))
        self.convs1_R = self.conv_block(16*2, 16, 3)
        self.conv_out = self.single_conv_out(16, 1, 3)

    def forward(self, conv1):        
        # conv1_L = self.conv_in(x)
        # conv1_L = self.convs1_L(conv_in)
        # # del conv_in
        # print(process.memory_info()[0])
        # conv2_L = self.down_conv1(conv1_L)
        # conv2_L = self.convs2_L(conv2_L)
        # print(process.memory_info()[0])
        # conv3_L = self.down_conv2(conv2_L)
        # conv3_L = self.convs3_L(conv3_L)
        # print(process.memory_info()[0])
        # conv4_L = self.down_conv3(conv3_L)
        # conv4_L = self.convs4_L(conv4_L)
        # print(process.memory_info()[0])

        # conv_bottom = self.down_conv_bottom(conv4_L)
        # conv_bottom = self.convs_bottom(conv_bottom)
        # conv_bottom = self.up_conv_bottom(conv_bottom)

        # print(process.memory_info()[0])
        # conv4_R = self.convs4_R(torch.cat([conv_bottom, conv4_L], 1))
        # # del conv_bottom, conv4_L
        # print(process.memory_info()[0])
        # conv4_R = self.up_conv1(conv4_R)
        # conv3_R = self.convs3_R(torch.cat([conv4_R, conv3_L], 1))
        # # del conv4_R, conv3_L
        # print(process.memory_info()[0])
        # conv3_R = self.up_conv2(conv3_R)
        # conv2_R = self.convs2_R(torch.cat([conv3_R, conv2_L], 1))
        # # del conv3_R, conv2_L
        # print(process.memory_info()[0])
        # conv2_R = self.up_conv3(conv2_R)
        # conv1_R = self.convs1_R(torch.cat([conv2_R, conv1_L], 1))
        # del conv2_R, conv1_L
        # print(process.memory_info()[0])
        # conv_out = self.conv_out(conv1_R)
        # del conv1_R
        # print(process.memory_info()[0])

        #process = psutil.Process(os.getpid())

        conv1 = self.conv_in(conv1)
        conv1 = self.convs1_L(conv1)
        conv2 = self.down_conv1(conv1)
        conv2 = self.convs2_L(conv2)
        conv3 = self.down_conv2(conv2)
        conv3 = self.convs3_L(conv3)
        conv4 = self.down_conv3(conv3)
        conv4 = self.convs4_L(conv4)

        conv_bottom = self.down_conv_bottom(conv4)
        conv_bottom = self.convs_bottom(conv_bottom)
        conv_bottom = self.up_conv_bottom(conv_bottom)

        conv4 = self.convs4_R(torch.cat([conv_bottom, conv4], 1))
        conv4 = self.up_conv1(conv4)
        conv3 = self.convs3_R(torch.cat([conv4, conv3], 1))
        conv3 = self.up_conv2(conv3)
        conv2 = self.convs2_R(torch.cat([conv3, conv2], 1))
        conv2 = self.up_conv3(conv2)
        conv1 = self.convs1_R(torch.cat([conv2, conv1], 1))
        conv1 = self.conv_out(conv1)
        #print(process.memory_info()[0])

        return conv1

    def single_conv_in(self, in_channels, out_channels, kernel_size, padding=1):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        
        return conv

    def single_conv_out(self, in_channels, out_channels, kernel_size, padding=1):
        conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        
        return conv

    def conv_block(self, in_channels, out_channels, kernel_size, padding=[(1,1),(1,1)]):
        conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding[1]))

        return conv_block

    def down_conv(self, in_channels, out_channels, kernel_size, stride):
        down_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride))

        return down_conv

    def up_conv(self, in_channels, out_channels, kernel_size, stride, output_padding=0):
        up_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, output_padding=output_padding))
        
        return up_conv


class unet_small(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(unet_small, self).__init__()

        #induction
        self.conv_in = self.single_conv_in(1, 4, 3)
        self.convs1_L = self.conv_block(4, 4 ,3)
        self.down_conv1 = self.down_conv(4, 8, 3, 3)
        self.convs2_L = self.conv_block(8, 8, 3)
        self.down_conv2 = self.down_conv(8, 16, 3, 3)
        self.convs3_L = self.conv_block(16, 16, 3, padding=[(0,0), (0,0)])
        self.down_conv3 = self.down_conv(16, 32, 3, 3)
        self.convs4_L = self.conv_block(32, 32, 3, padding=[(0,0),(0,0)])

        self.down_conv_bottom = self.down_conv(32, 64, 3, 3)
        self.convs_bottom = self.conv_block(64, 64, 3, padding=[(1,1),(1,1)])
        self.up_conv_bottom = self.up_conv(64, 32, 3, 3, output_padding=(0,0))

        self.convs4_R = self.conv_block(32*2, 32, 3, padding=[(2,2),(2,2)])
        self.up_conv1 = self.up_conv(32, 16, 3, 3, output_padding=(2,0))
        self.convs3_R = self.conv_block(16*2, 16, 3, padding=[(2,2),(2,2)])
        self.up_conv2 = self.up_conv(16, 8, 3, 3, output_padding=(2,2))
        self.convs2_R = self.conv_block(8*2, 8, 3)
        self.up_conv3 = self.up_conv(8, 4, 3, 3, output_padding=(0,2))
        self.convs1_R = self.conv_block(4*2, 4, 3)
        self.conv_out = self.single_conv_out(4, 1, 3)

    def forward(self, conv1):        
        conv1 = self.conv_in(conv1)
        conv1 = self.convs1_L(conv1)
        conv2 = self.down_conv1(conv1)
        conv2 = self.convs2_L(conv2)
        conv3 = self.down_conv2(conv2)
        conv3 = self.convs3_L(conv3)
        conv4 = self.down_conv3(conv3)
        conv4 = self.convs4_L(conv4)

        conv_bottom = self.down_conv_bottom(conv4)
        conv_bottom = self.convs_bottom(conv_bottom)
        conv_bottom = self.up_conv_bottom(conv_bottom)

        conv4 = self.convs4_R(torch.cat([conv_bottom, conv4], 1))
        conv4 = self.up_conv1(conv4)
        conv3 = self.convs3_R(torch.cat([conv4, conv3], 1))
        conv3 = self.up_conv2(conv3)
        conv2 = self.convs2_R(torch.cat([conv3, conv2], 1))
        conv2 = self.up_conv3(conv2)
        conv1 = self.convs1_R(torch.cat([conv2, conv1], 1))
        conv1 = self.conv_out(conv1)
        # print(process.memory_info()[0])

        return conv1

    def single_conv_in(self, in_channels, out_channels, kernel_size, padding=1):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        
        return conv

    def single_conv_out(self, in_channels, out_channels, kernel_size, padding=1):
        conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        
        return conv

    def conv_block(self, in_channels, out_channels, kernel_size, padding=[(1,1),(1,1)]):
        conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding[1]))

        return conv_block

    def down_conv(self, in_channels, out_channels, kernel_size, stride):
        down_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride))

        return down_conv

    def up_conv(self, in_channels, out_channels, kernel_size, stride, output_padding=0):
        up_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, output_padding=output_padding))
        
        return up_conv


class unet_small_collect(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(unet_small_collect, self).__init__()

        self.conv_in = self.single_conv_in(1, 4, 3)
        self.convs1_L = self.conv_block(4, 4 ,3)
        self.down_conv1 = self.down_conv(4, 8, 3, 3)
        self.convs2_L = self.conv_block(8, 8, 3)
        self.down_conv2 = self.down_conv(8, 16, 3, 3)
        self.convs3_L = self.conv_block(16, 16, 3, padding=[(0,0), (0,0)])
        self.down_conv3 = self.down_conv(16, 32, 3, 3)
        self.convs4_L = self.conv_block(32, 32, 3, padding=[(0,0),(0,0)])

        self.down_conv_bottom = self.down_conv(32, 64, 3, 3)
        self.convs_bottom = self.conv_block(64, 64, 3, padding=[(1,1),(1,1)])
        self.up_conv_bottom = self.up_conv(64, 32, 3, 3, output_padding=(0,0))

        self.convs4_R = self.conv_block(32*2, 32, 3, padding=[(2,2),(2,2)])
        self.up_conv1 = self.up_conv(32, 16, 3, 3, output_padding=(2,1))
        self.convs3_R = self.conv_block(16*2, 16, 3, padding=[(2,2),(2,2)])
        self.up_conv2 = self.up_conv(16, 8, 3, 3, output_padding=(2,1))
        self.convs2_R = self.conv_block(8*2, 8, 3)
        self.up_conv3 = self.up_conv(8, 4, 3, 3, output_padding=(0,0))
        self.convs1_R = self.conv_block(4*2, 4, 3)
        self.conv_out = self.single_conv_out(4, 1, 3)

    def forward(self, conv1):        

        conv1 = self.conv_in(conv1)
        conv1 = self.convs1_L(conv1)
        conv2 = self.down_conv1(conv1)
        conv2 = self.convs2_L(conv2)
        conv3 = self.down_conv2(conv2)
        conv3 = self.convs3_L(conv3)
        conv4 = self.down_conv3(conv3)
        conv4 = self.convs4_L(conv4)

        conv_bottom = self.down_conv_bottom(conv4)
        conv_bottom = self.convs_bottom(conv_bottom)
        conv_bottom = self.up_conv_bottom(conv_bottom)

        conv4 = self.convs4_R(torch.cat([conv_bottom, conv4], 1))
        conv4 = self.up_conv1(conv4)
        conv3 = self.convs3_R(torch.cat([conv4, conv3], 1))
        conv3 = self.up_conv2(conv3)
        conv2 = self.convs2_R(torch.cat([conv3, conv2], 1))
        conv2 = self.up_conv3(conv2)
        conv1 = self.convs1_R(torch.cat([conv2, conv1], 1))
        conv1 = self.conv_out(conv1)
        # print(process.memory_info()[0])

        return conv1

    def single_conv_in(self, in_channels, out_channels, kernel_size, padding=1):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        
        return conv

    def single_conv_out(self, in_channels, out_channels, kernel_size, padding=1):
        conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        
        return conv

    def conv_block(self, in_channels, out_channels, kernel_size, padding=[(1,1),(1,1)]):
        conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding[1]))

        return conv_block

    def down_conv(self, in_channels, out_channels, kernel_size, stride):
        down_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride))

        return down_conv

    def up_conv(self, in_channels, out_channels, kernel_size, stride, output_padding=0):
        up_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, output_padding=output_padding))
        
        return 
        
class unet_small_shallow_induction(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(unet_small_shallow_induction, self).__init__()

        self.conv_in = self.single_conv_in(1, 4, 3)
        self.convs1_L = self.conv_block(4, 4 ,3)
        self.down_conv1 = self.down_conv(4, 8, 3, 3)
        self.convs2_L = self.conv_block(8, 8, 3)
        self.down_conv2 = self.down_conv(8, 16, 3, 3)
        self.convs3_L = self.conv_block(16, 16, 3, padding=[(0,0), (0,0)])
        # self.down_conv3 = self.down_conv(16, 32, 3, 3)
        # self.convs4_L = self.conv_block(32, 32, 3, padding=[(0,0),(0,0)])

        self.down_conv_bottom = self.down_conv(16, 32, 3, 3)
        self.convs_bottom = self.conv_block(32, 32, 3, padding=[(1,1),(1,1)])
        self.up_conv_bottom = self.up_conv(32, 16, 3, 3, output_padding=(2,0))

        # self.convs4_R = self.conv_block(32*2, 32, 3, padding=[(2,2),(2,2)])
        # self.up_conv1 = self.up_conv(32, 16, 3, 3, output_padding=(2,1))
        self.convs3_R = self.conv_block(16*2, 16, 3, padding=[(2,2),(2,2)])
        self.up_conv2 = self.up_conv(16, 8, 3, 3, output_padding=(2,1))
        self.convs2_R = self.conv_block(8*2, 8, 3)
        self.up_conv3 = self.up_conv(8, 4, 3, 3, output_padding=(0,2))
        self.convs1_R = self.conv_block(4*2, 4, 3)
        self.conv_out = self.single_conv_out(4, 1, 3)

    def forward(self, conv1):        

        conv1 = self.conv_in(conv1)
        conv1 = self.convs1_L(conv1)
        conv2 = self.down_conv1(conv1)
        conv2 = self.convs2_L(conv2)
        conv3 = self.down_conv2(conv2)
        conv3 = self.convs3_L(conv3)
        # conv4 = self.down_conv3(conv3)
        # conv4 = self.convs4_L(conv4)

        conv_bottom = self.down_conv_bottom(conv3)
        conv_bottom = self.convs_bottom(conv_bottom)
        conv_bottom = self.up_conv_bottom(conv_bottom)

        # conv4 = self.convs4_R(torch.cat([conv_bottom, conv4], 1))
        # conv4 = self.up_conv1(conv4)
        conv3 = self.convs3_R(torch.cat([conv_bottom, conv3], 1))
        conv3 = self.up_conv2(conv3)
        conv2 = self.convs2_R(torch.cat([conv3, conv2], 1))
        conv2 = self.up_conv3(conv2)
        conv1 = self.convs1_R(torch.cat([conv2, conv1], 1))
        conv1 = self.conv_out(conv1)
        # print(process.memory_info()[0])

        return conv1

    def single_conv_in(self, in_channels, out_channels, kernel_size, padding=1):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        
        return conv

    def single_conv_out(self, in_channels, out_channels, kernel_size, padding=1):
        conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
        
        return conv

    def conv_block(self, in_channels, out_channels, kernel_size, padding=[(1,1),(1,1)]):
        conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding[0]),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding[1]))

        return conv_block

    def down_conv(self, in_channels, out_channels, kernel_size, stride):
        down_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride))

        return down_conv

    def up_conv(self, in_channels, out_channels, kernel_size, stride, output_padding=0):
        up_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, output_padding=output_padding))
        
        return up_conv