import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DoubleConv, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class U_Net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(U_Net, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv1 = DoubleConv(in_ch=self.in_ch, out_ch=64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = DoubleConv(512, 1024)
        
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=self.out_ch, kernel_size=1),
            nn.Sigmoid()
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        
        conv5 = self.conv5(pool4)
        
        up1 = self.up1(conv5)
        cat1 = torch.cat([conv4, up1], dim=1)
        conv6 = self.conv6(cat1)

        up2 = self.up2(conv6)
        cat2 = torch.cat([conv3, up2], dim=1)
        conv7 = self.conv7(cat2)
        
        up3 = self.up3(conv7)
        cat3 = torch.cat([conv2, up3], dim=1)
        conv8 = self.conv8(cat3)
        
        up4 = self.up4(conv8)
        cat4 = torch.cat([conv1, up4], dim=1)
        conv9 = self.conv9(cat4)
        
        conv_out = self.conv_out(conv9)
        
        return conv_out


if __name__ == "__main__":
    model = U_Net(in_ch=3, out_ch=1)
    input_size = torch.randn(1, 3, 256, 256)
    out_size = model(input_size)
    print(out_size.data.shape)

