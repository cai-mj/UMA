# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 18:57:38 2018

@author: cai-mj
"""
import numpy as np
import cv2

rgb_mean = {"EGTEA" : (0.1989, 0.3445, 0.5153),
            "UTG" : (0.4326, 0.4757, 0.4900),
            "Yale_Human_Grasp" : (0.3668, 0.3697, 0.3794),
            "EDSH" : (0.4625, 0.4356, 0.4099),
            "EDSH-K" : (0.4602, 0.4372, 0.4359),
            "GTEA" : (0.55, 0.4275, 0.1791),
            "Egohands" : (0.4745, 0.4415, 0.4044)
        }
rgb_std = {"EGTEA" : (0.1354, 0.1790, 0.2280),
           "UTG" : (0.2296, 0.2468, 0.2444),
           "Yale_Human_Grasp" : (0.2729, 0.2783, 0.2765),
           "Egohands" : (0.2284, 0.2154, 0.2172)
        }
lab_mean = {"EGTEA" : (104.0547, 142.2113, 155.6068),
            "Yale_Human_Grasp" : (99.1253, 129.2906, 128.9494)
        }
save_code = {"Yale_Human_Grasp" : "egtea2yale",
             "Yale_M" : "egtea2yaleM",
             "Yale_H" : "egtea2yaleH",
             "EDSH" : "egtea2edsh",
             "EDSH-K" : "egtea2edshk",
             "UTG" : "egtea2utg",
             "GTEA" : "egtea2gtea",
             "Egohands" : "egtea2egoh"}

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
        
def whiteBalance_rgb(img):
    """
    white balancing with grayworld assumption
    """
    img = img.astype(np.float32) / 255.0
    avg = [np.average(img[...,0]), np.average(img[...,1]), np.average(img[...,2])]
    
    avg_target = 0.5
    channels = cv2.split(img)
    out_channels = []
    for c, channel in enumerate(channels):
        assert len(channel.shape) == 2
        channel = channel * avg_target / avg[c]
        out_channels.append(channel)
    out = cv2.merge(out_channels) * 255
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

def prob2seg(prob):
    """
    convert probability map to 0/255 segmentation mask
    prob: probability map [0-255]
    """
    # smooth and thresholding
    prob = cv2.GaussianBlur(prob, (5, 5), 0)
    ret, mask = cv2.threshold(prob,75,1,cv2.THRESH_BINARY) # would remove the single-channel dimension
    
    # remove holes and spots
    kernel = np.ones((5,5),np.uint8)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    
    # filter out small area
    contours, hierarchy = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_ratio = 0.002
    max_ratio = 0.2
    area_img = prob.shape[0] * prob.shape[1]
    mask_close = mask_close * 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > area_img*min_ratio and area < area_img*max_ratio:
            cv2.drawContours(mask_close, [contours[i]], -1, 1, -1)

    return mask_close * 255

