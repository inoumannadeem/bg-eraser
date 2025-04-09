import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()
        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):
        return self.relu_s1(self.bn_s1(self.conv_s1(x)))

class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU7,self).__init__()
        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)
        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):
        hx = x
        hxin = self.rebnconvin(hx)
        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)
        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)
        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)
        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)
        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)
        hx6 = self.rebnconv6(hx)
        hx7 = self.rebnconv7(hx6)
        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1))
        hx6dup = F.interpolate(hx6d,size=hx5.shape[2:],mode='bilinear',align_corners=True)
        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1))
        hx5dup = F.interpolate(hx5d,size=hx4.shape[2:],mode='bilinear',align_corners=True)
        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = F.interpolate(hx4d,size=hx3.shape[2:],mode='bilinear',align_corners=True)
        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = F.interpolate(hx3d,size=hx2.shape[2:],mode='bilinear',align_corners=True)
        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = F.interpolate(hx2d,size=hx1.shape[2:],mode='bilinear',align_corners=True)
        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        return hx1d + hxin

class U2NET(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(U2NET,self).__init__()
        self.stage1 = RSU7(in_ch,32,64)
        self.pool12 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage2 = RSU7(64,32,128)
        self.pool23 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage3 = RSU7(128,64,256)
        self.pool34 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage4 = RSU7(256,128,512)
        self.pool45 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage5 = RSU7(512,256,512)
        self.pool56 = nn.MaxPool2d(2,stride=2,ceil_mode=True)
        self.stage6 = RSU7(512,256,512)
        self.stage5d = RSU7(1024,256,512)
        self.stage4d = RSU7(1024,128,256)
        self.stage3d = RSU7(512,64,128)
        self.stage2d = RSU7(256,32,64)
        self.stage1d = RSU7(128,16,64)
        self.side1 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1)
        self.side3 = nn.Conv2d(128,out_ch,3,padding=1)
        self.side4 = nn.Conv2d(256,out_ch,3,padding=1)
        self.side5 = nn.Conv2d(512,out_ch,3,padding=1)
        self.side6 = nn.Conv2d(512,out_ch,3,padding=1)
        self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self,x):
        hx = x
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)
        hx6 = self.stage6(hx)
        hx6up = F.interpolate(hx6,size=hx5.shape[2:],mode='bilinear',align_corners=True)
        hx5d = self.stage5d(torch.cat((hx6up,hx5),1))
        hx5dup = F.interpolate(hx5d,size=hx4.shape[2:],mode='bilinear',align_corners=True)
        hx4d = self.stage4d(torch.cat((hx5dup,hx4),1))
        hx4dup = F.interpolate(hx4d,size=hx3.shape[2:],mode='bilinear',align_corners=True)
        hx3d = self.stage3d(torch.cat((hx4dup,hx3),1))
        hx3dup = F.interpolate(hx3d,size=hx2.shape[2:],mode='bilinear',align_corners=True)
        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1))
        hx2dup = F.interpolate(hx2d,size=hx1.shape[2:],mode='bilinear',align_corners=True)
        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1))
        d1 = self.side1(hx1d)
        d2 = self.side2(hx2d)
        d2 = F.interpolate(d2,size=d1.shape[2:],mode='bilinear',align_corners=True)
        d3 = self.side3(hx3d)
        d3 = F.interpolate(d3,size=d1.shape[2:],mode='bilinear',align_corners=True)
        d4 = self.side4(hx4d)
        d4 = F.interpolate(d4,size=d1.shape[2:],mode='bilinear',align_corners=True)
        d5 = self.side5(hx5d)
        d5 = F.interpolate(d5,size=d1.shape[2:],mode='bilinear',align_corners=True)
        d6 = self.side6(hx6)
        d6 = F.interpolate(d6,size=d1.shape[2:],mode='bilinear',align_corners=True)
        d0 = self.outconv(torch.cat((d1,d2,d3,d4,d5,d6),1))
        return F.sigmoid(d0), F.sigmoid(d1), F.sigmoid(d2), F.sigmoid(d3), F.sigmoid(d4), F.sigmoid(d5), F.sigmoid(d6)

def normalize(img):
    return (img - img.min()) / (img.max() - img.min())

def remove_background(model, input_image):
    # Convert PIL Image to tensor
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(input_image).unsqueeze(0)
    
    # Move to same device as model
    image_tensor = image_tensor.to(next(model.parameters()).device)
    
    # Inference
    with torch.no_grad():
        d0, *_ = model(image_tensor)
    
    # Convert mask to PIL Image
    pred = d0.cpu().squeeze().numpy()
    mask = (normalize(pred) * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask)
    
    # Resize mask to original image size
    mask_img = mask_img.resize(input_image.size, Image.LANCZOS)
    
    # Apply mask to original image
    result = Image.new('RGBA', input_image.size, (0, 0, 0, 0))
    result.paste(input_image, mask=mask_img)
    
    return result
