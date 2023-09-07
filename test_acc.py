import cv2
import torch
from DetectVisualize import DetectTest
from OCRVisualize import OCRTest
import matplotlib.pyplot as plt
import numpy as np
from Licensedataset import cleandataset
import os
from NAFLocal import NAFNetLocal
import torchvision.transforms.functional as F
from torchvision.transforms import transforms
def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)

#dectetroot = 'wpod_net.pt'
dectetroot = 'WpodNet_ccpd2020_night_95.pth'
ocrroot = 'OcrNet.pth'
noiseroot = './originalnoise'
dexplorer = DetectTest(dectetroot)
oexplorer = OCRTest(ocrroot)
'''
model = NAFNetLocal(3,32,12,[2, 2, 4, 8],[2, 2, 2, 2])
residual=True
if residual:
    model.load_state_dict(torch.load('NAFNet4denoise_residual_crop256.pth',map_location='cpu'))
else:
    model.load_state_dict(torch.load('NAFNet4denoise_crop256_light.pth',map_location='cpu'))
'''
train_dataset = cleandataset('CCPD2020/ccpd_green/train')
#train_dataset = cleandataset('E:/CCPD2019.tar/CCPD2019/CCPD2019/ccpd_db')

acc = 0
iou_acc = 0
list = []
total = len(train_dataset)
sequence = np.random.choice(len(train_dataset),size=len(train_dataset),replace=False)
for i in range(total):
    #imgn,imgc, points,license = train_dataset[i]
    #c, h, w = imgn.shape
    image_path, points,license = train_dataset.dataset[sequence[i]]
    #image_path, points,license = train_dataset.dataset[233]
    
    print('label=',license)
    '''
    with torch.no_grad():
        recon = model(imgn.unsqueeze(0))
    recon = recon.squeeze(0)
    if residual:
            recon = imgn/recon
            recon = torch.clip(recon,0,1)
    else:
            recon = torch.clip(recon,0,1)
    recon = np.array((recon.squeeze(0).permute(1,2,0)* 255).byte().data)
    
    plt.subplot(1,3,1)
    plt.imshow(imgn.permute(1,2,0).numpy())
    plt.subplot(1,3,2)
    plt.imshow(imgc.permute(1,2,0).numpy())
    plt.subplot(1,3,3)
    plt.imshow(recon)
    plt.show()
    
    image = recon
    image = image[:,:,::-1].copy()#BGR
    '''
    image = cv2.imread(image_path)
    #image = F.adjust_contrast(torch.tensor(image).permute(2,0,1),1.2).permute(1,2,0).numpy()
    h, w, c = image.shape
    pre = dexplorer(image)
    #print(pre)
    if len(pre) > 0:
        pre,confidence = pre[0]

        '''标记框'''
        [x1, y1, x2, y2, x4, y4, x3, y3] = points
        points = np.array([x1, x2, x3, x4, y1, y2, y3, y4])
        #points[0:4] = points[0:4]*_w/w
        #points[4:] = points[4:]*_h/h
        [x1, x2, x3, x4, y1, y2, y3, y4] = points
        label = np.array([x1, x2, x3, x4, x1, y1, y2, y3, y4 ,y1]).reshape(2,5)
        src1 = [x1,y1,x3,y3]
        '''预测框'''
        bias = 0
        
        x1, x2, x3, x4, y1, y2, y3, y4 = pre.reshape(-1)
        x1, x2, x3, x4 = x1 * w-bias, x2 * w+bias, x3 * w+bias, x4 * w-bias
        y1, y2, y3, y4 = y1 * h-bias, y2 * h-bias, y3 * h+bias, y4 * h+bias
        
        box = np.array([x1, x2, x3, x4, x1, y1, y2, y3, y4 ,y1]).reshape(2,5)
        src2 = [x1,y1,x3,y3]
        
        iou = compute_IOU(src1,src2)
        if iou > 0.7:
            iou_acc += 1
        print('iou acc=',iou_acc,'/',i+1,'=',round(iou_acc/(i+1),4))
        
        image = image[:, :, ::-1].copy()#RGB
        
        ROTATED_SIZE = [48,144]
        #透视变换前坐标
        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        #变换后矩阵位置
        pts2 = np.float32([[0, 0],[ROTATED_SIZE[1],0],[ROTATED_SIZE[1], ROTATED_SIZE[0]],[0,ROTATED_SIZE[0]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        crop_img = cv2.warpPerspective(image, M, (ROTATED_SIZE[1],ROTATED_SIZE[0]))
        #crop_img = F.adjust_brightness(torch.tensor(crop_img).permute(2,0,1),2).permute(1,2,0).numpy()
        #crop_img = F.adjust_contrast(torch.tensor(crop_img).permute(2,0,1),2).permute(1,2,0).numpy()
        predict = oexplorer(crop_img)
        if len(predict)>8:
            predict = predict[:8]
        print('predict=',predict)
        
        if predict[1:] == license[1:]:
            acc += 1
            list.append(sequence[i])
        print('acc=',acc,'/',i+1,'=',round(acc/(i+1),4))
        
print('acc=',acc,'/',total,'=',acc/total)
print('iou acc=',iou_acc,'/',total,'=',iou_acc/total)
print(list)