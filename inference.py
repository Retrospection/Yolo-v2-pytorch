# coding: utf-8

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from src.yolo_net import YoloTest, Yolo
import torch
import cv2
import numpy as np


def read_image(path):
    img1 = cv2.imread(path)
    img1 = cv2.resize(img1, (224, 224))
    img1 = img1.transpose((2, 0, 1))
    img1 = img1[np.newaxis, :, :, :]
    return torch.Tensor(img1)


class FeatureExtractor(object):

    def __init__(self):
        self.net = YoloTest()
        state_dict = torch.load('trained_models\\only_params_trained_yolo_coco')
        del state_dict['stage3_conv2.weight']
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def get_feature(self, image_path):
        image = read_image(image_path)
        return self.net(image).reshape((1024 * 7 * 7)).detach().numpy()








# net = Yolo(10177)
# state_dict = torch.load('trained_models\\only_params_trained_yolo_coco')
# net.load_state_dict(state_dict)
# net.eval()




# img10 = readImage('D:\\dev\\dataset\\CASIA-WebFace\\0000045\\001.jpg')
# img11 = readImage('D:\\dev\\dataset\\CASIA-WebFace\\0000045\\002.jpg')
# img21 = readImage('D:\\dev\\dataset\\CASIA-WebFace\\0000099\\001.jpg')
#
# logits = net(img10)
# print(logits.view(1, 5, -1, 49).shape)
# output10 = net(img10).reshape((1024*7*7,)).detach().numpy()
# output11 = net(img11).reshape((1024*7*7,)).detach().numpy()
# output21 = net(img21).reshape((1024*7*7,)).detach().numpy()


# dis11 = np.linalg.norm(output10 - output11)
# dis21 = np.linalg.norm(output10 - output21)
#
# print(dis11)
# print(dis21)
#
#
# def cosdis(vec1, vec2):
#     return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*(np.linalg.norm(vec2)))
#
# cosdis11 = cosdis(output10, output11)
# cosdis21 = cosdis(output10, output21)
# print(cosdis11)
# print(cosdis21)