import os
from cv2 import resize
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import random
import matplotlib.pylab as plt
import cv2
import torchvision.transforms.functional as F
#matplotlib.use('TkAgg')

def default_loader(path):
    return Image.open(path).convert('RGB')  #转换RGB通道

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root1,root2, transform=None,makenoise = True,mixture=False,FFT = False,
    random_crop = False,crop_size = 256,resize = None,residual=False):
        self.dataset1 = []
        self.dataset2 = []
        self.root1 = root1
        self.root2 = root2
        self.transform = transform
        self.makenoise = makenoise
        self.random_crop = random_crop
        self.crop_size = crop_size
        self.resize = resize
        self.residual = residual
        self.mixture = mixture
        self.FFT = FFT
        for image_name in os.listdir(root1):
            self.dataset1.append((f'{root1}/{image_name}'))
        if mixture:
            for list_name in os.listdir(root2):
                for image_name in os.listdir(os.path.join(root2,list_name)):
                    self.dataset2.append((f'{root2}/{list_name}/{image_name}'))
        else:
            for image_name in os.listdir(root2):
                self.dataset2.append((f'{root2}/{image_name}'))

    def __getitem__(self, index):
        imgc = self.dataset1[index]
        #imgc = self.loader(imgc)
        
        imgc = cv2.imread(imgc)[:,:,::-1].copy() #h w c
        #imgc = F.adjust_contrast(torch.tensor(imgc).permute(2,0,1),2).permute(1,2,0).numpy()
        if self.resize is not None:
            imgc = cv2.resize(imgc, (self.resize[1],self.resize[0]), interpolation=cv2.INTER_AREA)
        
        if self.makenoise:
            noise_id = np.random.choice([0,1],size=(imgc.shape[1])) # w
            noise_id = np.nonzero(noise_id)[0]
            imgn = imgc.copy()
            imgn[:,noise_id,:] = np.tile(np.random.randint(0,50,size=noise_id.shape[0]),[imgc.shape[0],1,imgc.shape[2]]).reshape(imgc.shape[0],-1,imgc.shape[2])
        else:
            noise = self.dataset2[np.random.randint(len(self.dataset2))]
            noise = cv2.imread(noise)[:,:,::-1].copy()
            '''
            m = np.max(noise)
            noise = np.uint8(np.float64(noise) * 255 / m)
            '''
            noise = cv2.resize(noise,(imgc.shape[1],imgc.shape[0]),interpolation=cv2.INTER_AREA)
            #noise = np.where(noise == 0,noise+1,noise)
            imgn = np.uint8(noise/255 *imgc)
            #imgc = np.uint8(imgc*np.average(imgn)/np.average(imgc))
            if self.FFT:
                for i in range(imgn.shape[2]):
                    #dft = cv2.dft(np.float32(imgn[:,:,i]),flags = cv2.DFT_COMPLEX_OUTPUT)
                    dft = np.fft.fft2(np.float32(imgn[:,:,i]))
                    dft_shift = np.fft.fftshift(dft)
                    rows, cols = imgn[:,:,i].shape
                    crow,ccol = rows//2 , cols//2
                    # 首先创建一个掩码
                    mask = np.ones((rows,cols),np.uint8)
                    rowsize = 5
                    colsize = 5
                    mask[crow-rowsize:crow+rowsize, :ccol-colsize] = mask[crow-rowsize:crow+rowsize,ccol+colsize:] = 0
                    # 应用掩码和逆DFT
                    fshift = dft_shift*mask
                    f_ishift = np.fft.ifftshift(fshift)
                    #img_back = cv2.idft(f_ishift)
                    #img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
                    img_back = np.fft.ifft2(f_ishift)
                    img_back = np.abs(img_back)
                    imgn[:,:,i] = np.uint8(img_back)
                    #imgn[:,:,i] = np.uint8(img_back/np.max(img_back)*255)
                #imgn = np.uint8(imgn*1.2)
        if self.transform is not None:
            imgc = self.transform(imgc)#c,h,w   to tensor
            imgn = self.transform(imgn)#c,h,w   to tensor
            if self.residual:
                noise = self.transform(noise)
            #imgn = transforms.functional.adjust_brightness(imgn, 0.7)

        #return imgn,imgc,label
        if self.random_crop:
            x = random.randint(0,imgc.shape[1] - self.crop_size)
            y = random.randint(0,imgc.shape[2] - self.crop_size)
            imgc = imgc[:,x:x+self.crop_size,y:y+self.crop_size]
            imgn = imgn[:,x:x+self.crop_size,y:y+self.crop_size]
            if self.residual:
                noise = noise[:,x:x+self.crop_size,y:y+self.crop_size]
        if self.residual:
            return imgn,noise
        else:
            return imgn,imgc

    def __len__(self):
        return len(self.dataset1)


if __name__ == '__main__':
    transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            #transforms.Normalize([0.5 for _ in range(3)],[0.5 for _ in range(3)],)
            ])
    train_dataset = MyDataset('CCPD2020/ccpd_green/night','./Dataset/new', transform=transform,mixture=True,
    makenoise=False,random_crop=False,crop_size=256,residual=False,FFT=False)
    i = np.random.randint(0,len(train_dataset))
    print(i)#16 188 317 233 200 186 392 285
    imgn,imgc = train_dataset[i]
    #imgn,imgc = transforms.RandomEqualize(p=1)(torch.tensor(imgn).permute(2,0,1))/255,transforms.RandomEqualize(p=1)(torch.tensor(imgc).permute(2,0,1))/255
    print(torch.nn.functional.mse_loss(imgn.unsqueeze(0),imgc.unsqueeze(0)))
    print(torch.mean(torch.sqrt((imgn - imgc)**2 + 1e-12)))
    print('psnr=',10/np.log(10)*torch.log(1/torch.nn.functional.mse_loss(imgn.unsqueeze(0),imgc.unsqueeze(0))))
    imgn,imgc = imgn.permute(1,2,0).numpy(),imgc.permute(1,2,0).numpy()
    #imgn = imgn / np.max(imgn)
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.imshow(np.clip(imgn/imgc,0,1))
    #plt.savefig('build_dataset_imgc',bbox_inches='tight')
    
    #plt.figure(dpi=500)
    plt.subplot(1,3,2)
    plt.axis('off')
    plt.imshow(imgc)
    #plt.savefig('build_dataset_clean',bbox_inches='tight')

    #plt.figure(dpi=500)
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.imshow(imgn)
    #plt.savefig('build dataset_imgn',bbox_inches='tight')
    plt.show()
    '''
    num = 3
    id = [274,116,337]
    plt.figure(dpi=500)
    for i in range(num):
        imgn,imgc = train_dataset[id[i]]
        imgn,imgc = imgn.permute(1,2,0).numpy(),imgc.permute(1,2,0).numpy()
        plt.subplot(1,num,i+1)
        plt.axis('off')
        plt.imshow(imgn)
    plt.savefig('imgn',bbox_inches='tight')
    '''
    
