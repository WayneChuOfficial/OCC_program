from NAFNet import NAFNet
from torchvision import transforms
import torch
import matplotlib.pylab as plt
from Denoisedataset import MyDataset
import cv2
import numpy as np
from PIL import Image
from NAFLocal import NAFNetLocal
import torchvision.transforms.functional as f
from DetectVisualize import DetectTest
from OCRVisualize import OCRTest
from Licensedataset import DetectDataset
import matplotlib
from torchvision import transforms
import os
def seed_all(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
#seed_all(41)
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
dectetroot = 'WpodNet_ccpd2020_night_95.pth'
#dectetroot = 'wpod_net.pt'
ocrroot = 'OcrNet.pth'
makenoise = False
residual = False
dexplorer = DetectTest(dectetroot)
'''
noiseroot = './Dataset/new/test/4k_speed4000_iso3200'
device = torch.device('cuda')
if not makenoise:
    dataset = []
    for image_name in os.listdir(noiseroot):
            dataset.append((f'{noiseroot}/{image_name}'))

train_dataset = DetectDataset('CCPD2020/ccpd_green/night')
#train_dataset = DetectDataset('E:/CCPD2019.tar/CCPD2019/CCPD2019/ccpd_base')
#image_path, points = test_dataset.dataset[50]
i = np.random.randint(0,len(train_dataset))
print(i)
image_path, points = train_dataset.dataset[i]
image = cv2.imread(image_path)#读取通道顺序为B、G、R  h,w,c
imgc = image[:, :, ::-1].copy()#RGB  h,w,c

if makenoise:
    noise_id = np.random.choice([0,1],size=(imgc.shape[1]))
    noise_id = np.nonzero(noise_id)[0]
    imgn = imgc.copy()
    #imgn[:,noise_id,:] = 0
    imgn[:,noise_id,:] = np.tile(np.random.randint(0,50,size=noise_id.shape[0]),[imgc.shape[0],1,imgc.shape[2]]).reshape(imgc.shape[0],-1,imgc.shape[2])
else:
    noise = dataset[np.random.randint(len(dataset))]
    noise = cv2.imread(noise)[:,:,::-1].copy()
    noise = cv2.resize(noise,(imgc.shape[1],imgc.shape[0]),interpolation=cv2.INTER_AREA)
    
    imgn = np.uint8(noise/255 *imgc)

imgc = torch.tensor(imgc).permute(2,0,1)/255 #c h w
imgn = torch.tensor(imgn).permute(2,0,1)/255 #c h w
'''
root = 'realscene/distance3m_angle30'

dataset = []
for image_name in os.listdir(root):
        dataset.append((f'{root}/{image_name}'))
i = np.random.randint(len(dataset))
print(i)
imgn = cv2.imread(dataset[44])[200:3000,600:2600,::-1].copy()#RGB
imgn = cv2.resize(imgn, (720,1160), interpolation=cv2.INTER_AREA)
imgn = torch.tensor(imgn).permute(2,0,1)/255 #c h w
#model = NAFNet(3,32,12,[2, 2, 4, 8],[2, 2, 2, 2])
model = NAFNetLocal(3,32,12,[2, 2, 4, 8],[2, 2, 2, 2])

model.load_state_dict(torch.load('NAFNet4denoise_0069.pth',map_location='cpu'))
model.eval()

with torch.no_grad():
    recon = model(imgn.unsqueeze(0))
recon = torch.clip(recon,0,1)
#recon = np.array(recon.squeeze(0).permute(1,2,0).detach().numpy()*255,dtype=np.uint8) # h w c
#recon = recon.squeeze(0).permute(1,2,0).detach().numpy()
recon = np.array((recon.squeeze(0).permute(1,2,0)* 255).byte().data) # h w c
h, w, c = recon.shape
image = recon
#image = np.array((imgn.permute(1,2,0)* 255).byte().data)
image = image[:,:,::-1].copy() #B G R
#print(torch.nn.functional.mse_loss(recon,image))
pre = dexplorer(image)
print(pre)
pre,confidence = pre[0]


'''标记框'''
'''
[x1, y1, x2, y2, x4, y4, x3, y3] = points
points = np.array([x1, x2, x3, x4, y1, y2, y3, y4])
[x1, x2, x3, x4, y1, y2, y3, y4] = points
label = np.array([x1, x2, x3, x4, x1, y1, y2, y3, y4 ,y1]).reshape(2,5)
'''
'''预测框 防止标记框偏小 把每个顶点向外拉bias个像素'''
bias = 0
x1, x2, x3, x4, y1, y2, y3, y4 = pre.reshape(-1)
x1, x2, x3, x4 = x1 * w-bias, x2 * w+bias, x3 * w+bias, x4 * w-bias
y1, y2, y3, y4 = y1 * h-bias, y2 * h-bias, y3 * h+bias, y4 * h+bias
box = np.array([x1, x2, x3, x4, x1, y1, y2, y3, y4 ,y1]).reshape(2,5)

image = image[:,:,::-1].copy() #RGB

'''从原图中扣出车牌'''
oexplorer = OCRTest(ocrroot)
ROTATED_SIZE = [48,144]
#透视变换前坐标
pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
#变换后矩阵位置
pts2 = np.float32([[0, 0],[ROTATED_SIZE[1],0],[ROTATED_SIZE[1], ROTATED_SIZE[0]],[0,ROTATED_SIZE[0]]])
M = cv2.getPerspectiveTransform(pts1, pts2)
crop_img = cv2.warpPerspective(image, M, (ROTATED_SIZE[1],ROTATED_SIZE[0]))
predict = oexplorer(crop_img)
print(predict)

imgn = imgn.permute(1,2,0).numpy()
recon = recon / 255
plt.figure()
plt.subplot(1,2,1)
plt.imshow(imgn)
plt.subplot(1,2,2)
plt.imshow(recon)
plt.plot(box[0],box[1],color='r')
plt.text(x1,y1-20,predict,c = 'r')
plt.show()
