import torch
import cv2
import numpy as np
import os
from OCRNet import OcrNet
from fakelicense import Draw
import matplotlib.pylab as plt

class OCRTest:
    def __init__(self, root,device=torch.device('cpu')):
        self.net = OcrNet(70)
        self.class_name = ['*', "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", 
    "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", 
    "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",'港','澳','A', 'B', 'C', 'D', 'E',
     'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        self.device = device
        self.net.load_state_dict(torch.load(root, map_location='cpu'))
        self.net.to(device)
        self.net.eval()

    def __call__(self, image):
        with torch.no_grad():
            image = torch.from_numpy(image).permute(2, 0, 1) / 255
            image = image.unsqueeze(0)
            out = self.net(image.to(self.device)).reshape(-1, 70)
            out = torch.argmax(out, dim=1).cpu().numpy().tolist()
            c = ''
            for i in out:
                c += self.class_name[i]
            return self.deduplication(c)

    def deduplication(self, c):
        '''符号去重'''
        temp = ''
        new = ''
        for i in c:
            if i == temp:
                continue
            else:
                if i == '*':
                    temp = i
                    continue
                new += i
                temp = i
        return new

