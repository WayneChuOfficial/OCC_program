from tkinter import Image
from charset_normalizer import detect
from torch.utils.data import Dataset
from torch import nn
import os
from torchvision.transforms import transforms
from einops import rearrange
import random
import cv2
import enhance,make_label
import numpy
import torch
import re
from fakelicense import Draw
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F

class OcrDataSet(Dataset):
    def __init__(self,class_name=['*', "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", 
    "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", 
    "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学",'港','澳','A', 'B', 'C', 'D', 'E',
     'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']):
        super(OcrDataSet, self).__init__()
        self.dataset = []
        self.draw = Draw()
        self.class_name = class_name
        for i in range(10000):
            self.dataset.append(1)
        self.smudge = enhance.Smudge()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        plate, label = self.draw()
        target = []
        for i in label:
            target.append(self.class_name.index(i))
        plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)

        '''数据增强'''
        plate = self.data_to_enhance(plate)

        # cv2.imshow('a', plate)
        # cv2.waitKey()

        image = torch.from_numpy(plate).permute(2, 0, 1) / 255
        # image = self.transformer(image)
        # print(image.shape)
        target_length = torch.tensor(len(target)).long()
        target = torch.tensor(target).reshape(-1).long()
        _target = torch.full(size=(15,), fill_value=0, dtype=torch.long)
        _target[:len(target)] = target

        return image, _target, target_length

    def data_to_enhance(self, plate):
        '''随机污损'''
        plate = self.smudge(plate)
        '''高斯模糊'''
        plate = enhance.gauss_blur(plate)
        '''高斯噪声'''
        plate = enhance.gauss_noise(plate)
        '''增广数据'''
        plate, pts = enhance.augment_sample(plate)
        '''抠出车牌'''
        plate = enhance.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate


class DetectDataset(Dataset):

    def __init__(self,root):
        super(DetectDataset, self).__init__()
        self.dataset = []
        self.draw = Draw()
        self.smudge = enhance.Smudge()
        self.root = root
        for image_name in os.listdir(root):
            box = self.get_box(image_name)
            x3, y3, x4, y4, x1, y1, x2, y2 = box#右下、左下、左上、右上
            box = [x1, y1, x2, y2, x4, y4, x3, y3]
            self.dataset.append((f'{root}/{image_name}', box))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, points = self.dataset[item]
        image = cv2.imread(image_path)
        #image =  transforms.ColorJitter(0.5,0.5,0.5,0.5)(torch.tensor(image).permute(2,0,1)).permute(1,2,0).numpy()
        #更换假车牌
        '''
        if random.random() < 0.5:
            plate, _ = self.draw()
            plate = cv2.cvtColor(plate, cv2.COLOR_RGB2BGR)
            plate = self.smudge(plate)
            image = enhance.apply_plate(image, points, plate)
        '''
        [x1, y1, x2, y2, x4, y4, x3, y3] = points
        points = [x1, x2, x3, x4, y1, y2, y3, y4]
        image, points = enhance.augment_detect(image, points, 208)
        
        # cv2.imshow('a',image)
        # cv2.waitKey()
        image_tensor = torch.from_numpy(image)/255
        image_tensor = rearrange(image_tensor, 'h w c -> c h w')
        label = make_label.object_label(points,208,16)
        label = torch.from_numpy(label).float()
        return image_tensor,label

    def up_background(self, image):
        '''高斯模糊'''
        image = enhance.gauss_blur(image)
        '''高斯噪声'''
        image = enhance.gauss_noise(image)
        '''随机剪裁'''
        image = enhance.random_cut(image, (208, 208))
        return image

    def data_to_enhance(self, plate):
        '''随机污损'''
        plate = self.smudge(plate)
        '''高斯模糊'''
        plate = enhance.gauss_blur(plate)
        '''高斯噪声'''
        plate = enhance.gauss_noise(plate)
        '''增广数据'''
        plate, pts = enhance.augment_sample(plate)
        '''抠出车牌'''
        plate = enhance.reconstruct_plates(plate, [numpy.array(pts).reshape((2, 4))])[0]
        return plate

    def get_box(self, name):
        # print(name)
        name = re.split('[.&_-]', name)[7:15]
        # print(name)
        # exit()
        name = [int(i) for i in name]
        return name

class cleandataset(Dataset):
    def __init__(self,root):
        super(cleandataset, self).__init__()
        self.dataset = []
        self.provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
        self.alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
        self.ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
        self.root = root
        for image_name in os.listdir(root):
            box,label = self.get_label(image_name)
            x3, y3, x4, y4, x1, y1, x2, y2 = box#右下、左下、左上、右上
            box = [x1, y1, x2, y2, x4, y4, x3, y3]
            self.dataset.append((f'{root}/{image_name}', box,label))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, points ,label= self.dataset[item]
        image = cv2.imread(image_path)

        [x1, y1, x2, y2, x4, y4, x3, y3] = points
        points = [x1, x2, x3, x4, y1, y2, y3, y4]
        image, points = enhance.augment_detect(image, points, 208)
        
        image_tensor = torch.from_numpy(image)/255
        image_tensor = rearrange(image_tensor, 'h w c -> c h w')
        label = make_label.object_label(points,208,16)
        label = torch.from_numpy(label).float()
        return image_tensor,label

    
    def get_label(self, name):
        # print(name)
        box = re.split('[.&_-]', name)[7:15]
        label = re.split('[.&_-]', name)[15:-3]
        # print(name)
        # exit()
        box = [int(i) for i in box]
        label = [int(i) for i in label]
        c = ''
        c += self.provinces[label[0]]
        c += self.alphabets[label[1]]
        for i in label[2:]:
            c += self.ads[i]
        return box,c

