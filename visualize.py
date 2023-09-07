import cv2
import torch
from DetectVisualize import DetectTest
from OCRVisualize import OCRTest
import matplotlib.pyplot as plt
import numpy as np
from Licensedataset import DetectDataset
import matplotlib,os
import torchvision.transforms.functional as F

matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
dectetroot = 'WpodNet_ccpd2020_night_95.pth'
ocrroot = 'OcrNet.pth'
dexplorer = DetectTest(dectetroot)
makenoise = False
noiseroot = './occ_noise/ISO400'
if not makenoise:
    dataset = []
    for image_name in os.listdir(noiseroot):
            dataset.append((f'{noiseroot}/{image_name}'))
train_dataset = DetectDataset('CCPD2020/ccpd_green/night')
#train_dataset = DetectDataset('E:/CCPD2019.tar/CCPD2019/CCPD2019/ccpd_base')
i = np.random.randint(0,len(train_dataset))
print(i)
'''
list1 =[52, 183, 58, 126, 374, 178, 301, 375, 139, 382, 266, 49, 93, 388, 343, 99, 203, 218, 132, 73, 87, 260, 148, 3, 116, 75, 28, 357, 151, 390, 216, 264, 184, 1, 82, 41, 364, 273, 242, 253, 156, 13, 212, 391, 246, 330, 225, 351, 265, 354, 144, 78, 63, 274, 169, 110, 276, 202, 174, 314, 288, 84, 275, 193, 283, 43, 226, 161, 196, 258, 317, 180, 24, 127, 395, 335, 29, 34, 115, 336, 77, 79, 118, 389, 83, 333, 7, 113, 236, 299, 256, 285, 25, 270, 241, 277, 367, 51, 214, 298, 54, 119, 101, 62, 171, 150, 229, 338, 55, 91, 123, 105, 353, 120, 189, 60, 94, 67, 365, 141, 11, 164, 21, 349, 155, 61, 42, 45, 315, 220, 170, 371, 373, 134, 249, 204, 179, 281, 306, 76, 192, 31, 33, 145, 168, 136, 30, 344, 162, 32, 206, 291, 269, 208, 85, 370, 56, 103, 250, 233, 111, 175, 381, 259, 237, 44, 92, 297, 108, 278, 173, 88, 352, 117, 350, 39, 341, 358, 48, 362, 369, 90, 66, 121, 154, 366, 199, 35, 223, 248, 53, 97, 9, 304, 131, 348, 268, 197, 15, 312, 272, 177, 234, 157, 129, 27, 287, 262, 300, 165, 255, 16, 114, 368, 176, 332, 240, 235, 130, 394, 158, 384, 172, 187, 252, 337, 284, 257, 59, 231, 205, 356, 137, 342, 200, 143, 149, 109, 22, 346, 339, 135, 12, 228, 160, 243, 247, 153, 282, 
6, 104, 293, 26, 19, 327, 379, 194, 294, 46, 0, 201, 377, 133, 64, 142, 140, 311, 98, 8, 10, 5, 209, 20, 295, 398, 122, 95, 40, 224, 163, 96, 72, 80, 198, 210, 47, 4, 303, 232, 396, 302, 267, 378, 17, 146, 188, 219, 221, 100, 147, 38, 37, 107, 251, 227, 207, 217, 68, 279, 286, 23, 244, 14, 215, 65, 290, 263, 70, 2, 347, 74, 112, 81, 271, 191, 71, 128, 124, 57, 36, 125, 321]
list2=[196, 129, 226, 23, 20, 367, 76, 147, 17, 163, 12, 174, 335, 120, 158, 131, 115, 206, 288, 311, 87, 214, 209, 30, 155, 272, 332, 259, 84, 92, 256, 295, 337, 16, 145, 82, 35, 260, 37, 19, 144, 128, 59, 141, 357, 302, 225, 154, 234, 148, 166, 83, 110, 78, 228, 157, 369, 38, 264, 266, 382, 257, 205, 0, 299, 189, 374, 65, 221, 252, 306, 99, 14, 314, 278, 100, 33, 79, 284, 341, 91, 351, 352, 81, 175, 108, 350, 194, 371, 304, 235, 117, 269, 29, 193, 244, 261, 111, 282, 389, 275, 187, 41, 94, 265, 31, 80, 283, 249, 360, 368, 
95, 49, 380, 199, 339, 354, 73, 45, 395, 58, 28, 274, 291, 315, 113, 250, 294, 287, 388, 231, 224, 62, 361, 9, 363, 390, 279, 303, 
168, 40, 365, 336, 96, 126, 290, 297, 353, 216, 198, 207, 137, 123, 4, 162, 347, 68, 396, 200, 2, 384, 268, 229, 348, 149, 151, 338, 112, 127, 237, 109, 312, 184, 8, 66, 165, 223, 173, 258, 321, 391, 24, 26, 203, 51, 394, 180, 179, 202, 130, 142, 188, 114, 70, 201, 132, 7, 21, 285, 104, 344, 220, 381, 56, 171, 227, 52, 71, 243, 25, 139, 270, 362, 242, 204, 122, 32, 378, 39, 97, 42, 55, 135, 356, 15, 292, 48, 57, 262, 11, 85, 69, 125, 146, 72, 103, 373, 273, 232, 150, 170, 61, 197, 93, 43, 377, 124, 379, 271, 398, 116, 
364, 327, 240, 330, 349, 212, 355, 176, 53, 277, 248, 121, 241, 133, 101, 67, 281, 317, 370, 301, 210, 156, 143, 119, 153, 236, 5, 
169, 218, 18, 164, 375, 27, 160, 215, 36, 63, 217, 178, 263, 140, 54, 22, 172, 300, 46, 134, 1, 358, 255, 105, 60, 345, 276, 47, 298, 233, 98, 75, 34, 293, 74, 346, 253, 342, 107, 161, 3, 219, 88, 192, 77, 44, 10, 191, 343, 90, 366, 13, 247, 6, 183, 208, 333]
with open('test.txt','w',encoding='utf-8') as f:    
    for i in range(len(train_dataset)):
        if i not in list1 and i not in list2:
            print(i)
            image_path, points = train_dataset.dataset[i]
            f.write(image_path[26:]+'\n')
'''
image_path, points = train_dataset.dataset[16]
image = cv2.imread(image_path)#B、G、R       
#image = F.adjust_contrast(torch.tensor(image).permute(2,0,1),2).permute(1,2,0).numpy() 
#image = F.adjust_brightness(torch.tensor(image).permute(2,0,1),2).permute(1,2,0).numpy() 
h, w, c = image.shape

imgc = image.copy()

if makenoise:
    noise_id = np.random.choice([0,1],size=(imgc.shape[1]))
    noise_id = np.nonzero(noise_id)[0]
    imgn = imgc.copy()
    imgn[:,noise_id,:] = 0
else:
    imgn = dataset[np.random.randint(len(dataset))]
    imgn = cv2.imread(imgn)[:,:,::-1].copy()
    imgn = cv2.resize(imgn,(imgc.shape[1],imgc.shape[0]),interpolation=cv2.INTER_AREA)

    imgn = np.array(imgn/255 *imgc,dtype=np.uint8)
    for i in range(imgn.shape[2]):
        #imgn = cv2.cvtColor(imgn,cv2.COLOR_RGB2GRAY)
        dft = cv2.dft(np.float32(imgn[:,:,i]),flags = cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        rows, cols = imgn[:,:,i].shape
        crow,ccol = rows//2 , cols//2
        # 首先创建一个掩码，
        mask = np.ones((rows,cols,2),np.uint8)
        rowsize = 5
        colsize = 5
        mask[crow-rowsize:crow+rowsize, :ccol-colsize] = mask[crow-rowsize:crow+rowsize,ccol+colsize:] = 0
        # 应用掩码和逆DFT
        fshift = dft_shift*mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
        #imgn[:,:,i] = img_back
        imgn[:,:,i] = np.uint8(img_back/np.max(img_back)*255)                    


image = imgc
pre = dexplorer(image)
print(pre)
if len(pre) > 0:
    pre,confidence = pre[0]


'''标记框'''
[x1, y1, x2, y2, x4, y4, x3, y3] = points
points = np.array([x1, x2, x3, x4, y1, y2, y3, y4])
[x1, x2, x3, x4, y1, y2, y3, y4] = points
label = np.array([x1, x2, x3, x4, x1, y1, y2, y3, y4 ,y1]).reshape(2,5)

if len(pre) > 0:
    '''预测框'''
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
    predict = oexplorer(crop_img)
    print(predict)

plt.figure()
plt.imshow(image)
plt.plot(label[0],label[1],color='b')
if len(pre) > 0:
    plt.plot(box[0],box[1],color='r')
    plt.text(x1,y1-20,predict,c = 'r')
plt.show()