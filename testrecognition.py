import cv2
import torch
from DetectVisualize import DetectTest
from OCRVisualize import OCRTest
import matplotlib.pyplot as plt
import numpy as np
from Licensedataset import DetectDataset
import matplotlib,os

matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
#dectetroot = 'wpod_net.pt'
dectetroot = 'WpodNet_ccpd2020_night_95.pth'
ocrroot = 'OcrNet.pth'
dexplorer = DetectTest(dectetroot)

image = cv2.imread('./CCPD2020/ccpd_green/night/029748563218390804-90_253-132&494_435&584-435&584_155&581_132&496_420&494-0_0_3_25_30_30_30_30-128-20.jpg')#B、G、R
image = cv2.resize(image,(720,1160),interpolation=cv2.INTER_AREA)
h, w, c = image.shape


pre = dexplorer(image)
print(pre)
if len(pre) > 0:
    pre,confidence = pre[0]

if len(pre) > 0:
    '''预测框 '''
    bias = 0
    x1, x2, x3, x4, y1, y2, y3, y4 = pre.reshape(-1)
    x1, x2, x3, x4 = x1 * w-bias, x2 * w+bias, x3 * w+bias, x4 * w-bias
    y1, y2, y3, y4 = y1 * h-bias, y2 * h-bias, y3 * h+bias, y4 * h+bias
    box = np.array([x1, x2, x3, x4, x1, y1, y2, y3, y4 ,y1]).reshape(2,5)

    image = image[:, :, ::-1].copy()
    '''从原图中扣出车牌'''
    oexplorer = OCRTest(ocrroot)
    ROTATED_SIZE = [48,144]
    #透视变换前坐标
    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    #变换后矩阵位置
    pts2 = np.float32([[0, 0],[ROTATED_SIZE[1],0],[ROTATED_SIZE[1], ROTATED_SIZE[0]],[0,ROTATED_SIZE[0]]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    crop_img = cv2.warpPerspective(image, M, (ROTATED_SIZE[1],ROTATED_SIZE[0]))
    plt.figure(dpi=200)
    plt.imshow(crop_img)
    plt.axis('off')
    plt.savefig('crop_img',bbox_inches='tight')
    predict = oexplorer(crop_img)
    print(predict)

plt.figure()
plt.imshow(image)
if len(pre) > 0:
    plt.plot(box[0],box[1],color='r')
    plt.text(x1,y1-20,predict,c = 'r')
plt.show()