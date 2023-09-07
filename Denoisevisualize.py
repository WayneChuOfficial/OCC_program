from NAFNet import NAFNet
from NAFLocal import NAFNetLocal
from memnet import MemNet
from torchvision import transforms
import torch
import matplotlib.pylab as plt
from Denoisedataset import MyDataset
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as f
import cv2
residual=False
def calculate_psnr(imgc,recon):
        scale = 10/np.log(10)
        max = 1
        return scale*torch.log(max**2/torch.nn.functional.mse_loss(recon,imgc))

#model = NAFNet(3,32,12,[2, 2, 4, 8],[2, 2, 2, 2])
model = NAFNetLocal(3,32,12,[2, 2, 4, 8],[2, 2, 2, 2])
if residual:
        model.load_state_dict(torch.load('NAFNet4denoise_32_residual_mixture_night.pth',map_location='cpu'))
else:
        model.load_state_dict(torch.load('NAFNet4denoise_0069.pth',map_location='cpu'))

transform = transforms.Compose([
        transforms.ToTensor(),
        ])
train_dataset = MyDataset('CCPD2020/ccpd_green/night','./Dataset/new/test', transform=transform,makenoise=False,residual=residual,mixture=True,FFT=False,random_crop=False,crop_size=256)
#train_dataset = Coredataset('dataBlue.txt',transform = transform)
#train_dataset = MyDataset('CCPD2020/ccpd_green/night','./occ_noise/ISO640', transform=transform,makenoise=False,residual=residual,mixture=False,FFT=False,random_crop=False,crop_size=256)
id = [1720,562,4166]

i = np.random.randint(0,len(train_dataset))
print(i)
imgn, imgc = train_dataset[i]
with torch.no_grad():
        recon = model(imgn.unsqueeze(0))
recon = recon.squeeze(0)
if residual:
        recon = imgn/(recon+1e-12)
        recon = torch.clip(recon,0,1)
        imgc = imgn/(imgc+1e-12)
        imgc = torch.clip(imgc,0,1)
        print(torch.isnan (imgc).any())
        print(torch.isnan (recon).any())
        psnr = calculate_psnr(imgc,recon)
else:
        recon = torch.clip(recon,0,1)
        psnr = calculate_psnr(imgc,recon)
recon = recon.permute(1,2,0).detach().numpy()
imgn = imgn.permute(1,2,0).numpy()
imgc = imgc.permute(1,2,0).numpy()


print('psnr = ',psnr.item())
plt.figure()
plt.axis('off')
plt.subplot(1,3,1)
plt.imshow(imgc)
#plt.savefig('result_imgc',bbox_inches='tight')
plt.subplot(1,3,2)
#plt.figure(dpi=500)
plt.axis('off')
plt.imshow(imgn)
#plt.savefig('result_imgn',bbox_inches='tight')
plt.subplot(1,3,3)
#plt.figure(dpi=500)
plt.axis('off')
plt.imshow(recon)
#plt.savefig('result_recon',bbox_inches='tight')
plt.show()
'''
plt.figure(dpi=500)
for i in range(len(id)):
        imgn, imgc = train_dataset[id[i]]

        with torch.no_grad():
                recon = model(imgn.unsqueeze(0))
        recon = recon.squeeze(0)
        if residual:
                recon = imgn/recon
                imgc = imgn/imgc
                imgc = torch.clip(imgc,0,1)
                recon = torch.clip(recon,0,1)
                psnr = calculate_psnr(imgc,recon)
        else:
                recon = torch.clip(recon,0,1)
                psnr = calculate_psnr(imgc,recon)
        recon = recon.permute(1,2,0).detach().numpy()
        imgn = imgn.permute(1,2,0).numpy()
        imgc = imgc.permute(1,2,0).numpy()

        print('psnr = ',psnr.item())
        
        plt.subplot(1,3,i+1)
        plt.axis('off')
        plt.imshow(recon)
#plt.show()
plt.savefig('recon_realnoise',bbox_inches='tight')
'''

