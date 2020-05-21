#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:41:46 2019

@author: arthur
"""

import argparse
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from net_pix2pix import define_G
from refinenet import rf101
from utils import AverageMeter, prob2seg, save_code

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
opt = parser.parse_args()
opt.cuda = True
opt.lr = 1e-5
wd = 0 #1e-4 # weight decay for L2 penalty

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.deterministic = True
cudnn.benchmark = False

np.random.seed(0)

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    gpu_ids = [opt.gpu]
else:
    gpu_ids = []

transform_list = [transforms.ToTensor(),
                  transforms.Normalize([0.5], [0.5])]
transform = transforms.Compose(transform_list)
transform1 = transforms.Compose([transforms.ToTensor()])


class HandBayesianDataset(Dataset):
    def __init__(self, dataset, phase, pseudoDir):
        super(HandBayesianDataset, self).__init__()
        self.imgs = []
        self.masks = []
        self.pmaps = []
        self.umaps = []
        maskDir = "data/%s/%sannot" % (dataset, phase)
        filenames = [k for k in os.listdir(maskDir) if ".png" in k]
        filenames.sort()
        self.imgs.extend([pseudoDir + '/' + filename for filename in filenames])
        self.masks.extend([maskDir + '/' + filename for filename in filenames])
        self.pmaps.extend([pseudoDir + '/' + filename[:-4] + "_p.png" for filename in filenames])
        self.umaps.extend([pseudoDir + '/' + filename[:-4] + "_u.png" for filename in filenames])
        img_sample = cv2.imread(self.imgs[0])
        self.img_size = (img_sample.shape[1], img_sample.shape[0])
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        # get input image
        imgname = self.imgs[index]
        img = cv2.imread(imgname)
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) 
        img_tensor = transform(img)
        # get hand mask image
        maskname = self.masks[index]                
        mask = cv2.imread(maskname, 0)          
        ret, mask_thres = cv2.threshold(mask,0,1,cv2.THRESH_BINARY)
        mask_thres = cv2.resize(mask_thres, (256, 256), interpolation = cv2.INTER_CUBIC)  
        mask_tensor = torch.from_numpy(mask_thres)            
        mask_tensor = mask_tensor.unsqueeze(0)
        # get hand prediction map
        pname = self.pmaps[index]
        pred = cv2.imread(pname, 0)
        ret, pred_thres = cv2.threshold(pred, 128, 1, cv2.THRESH_BINARY)
        pred_thres = cv2.resize(pred_thres, (256, 256), interpolation = cv2.INTER_CUBIC)
        pred_tensor = torch.from_numpy(pred_thres)
        pred_tensor = pred_tensor.unsqueeze(0)
        # get uncertainty map and compute spatial weight
        uname = self.umaps[index]
        uncertainty = cv2.imread(uname, 0)
        weight = cv2.resize(255-uncertainty, (256, 256), interpolation = cv2.INTER_CUBIC)
        #ret, weight = cv2.threshold(weight, 127, 255, cv2.THRESH_TOZERO) # binary confidence mask
        weight = np.expand_dims(weight, axis=2)
        weight_tensor = transform1(weight)
        
        sample = {'img': img_tensor, 'mask': mask_tensor, 'pred': pred_tensor, 'weight': weight_tensor, 'index': index}
        return sample


print('===> Building model')
# simpler pix2pix generator
#netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', True, gpu_ids) # set dropout layer as True for Bayesian CNN
# RefineNet
netG = rf101(opt.output_nc, pretrained=False, gpu_ids=gpu_ids, use_dropout=True) # set dropout layer as True for Bayesian CNN

criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()
criterionBCE_weight = nn.BCELoss(reduction='none')

# setup optimizer
optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr, weight_decay=wd)

if opt.cuda:
    criterionL1 = criterionL1.cuda(device=opt.gpu)
    criterionMSE = criterionMSE.cuda(device=opt.gpu)
    criterionBCE_weight = criterionBCE_weight.cuda(device=opt.gpu)


def train_update(epoch, dataDir, uct_ratio):
    """
    train with updated psudo training data (predition from previous epoch)
    dataDir: filefolder saving intermediate prediction results
    uct_ratio: uncertainty ratio w.r.t mean uncertainty of training samples
    """
    losses = AverageMeter()
    trainset = HandBayesianDataset(opt.dataset, "test", dataDir)
    data_loader = DataLoader(dataset=trainset, batch_size=opt.batchSize, shuffle=True)
    netG.train()
    for iteration, sample in enumerate(data_loader):
        img = sample['img']
        mask = sample['pred']  # Note that prediction of previous iteration is used as mask (or pseudo-labels)
        weight = sample['weight']
        
        if opt.cuda:
            img = img.cuda(device=opt.gpu)
            mask = mask.float().cuda(device=opt.gpu)
            weight = weight.cuda(device=opt.gpu)
        
        # forward
        pred_hand = netG(img)

        optimizerG.zero_grad()
        loss = criterionBCE_weight(pred_hand, mask)  
        # compute spatial weighted loss
        loss = loss * weight
        
        # compute sample weighted loss (not used in the paper)
#        uct = torch.from_numpy(uct_ratio[sample['index']])
#        if opt.cuda:
#            uct = uct.cuda(device=opt.gpu)
#        for b in range(len(uct)):
#            loss[b] = loss[b] * uct[b]
        
        loss = torch.mean(loss)
        loss.backward()
        optimizerG.step()
        
        losses.update(loss.data.item(), opt.batchSize)
        
        if (iteration+1) % 10 ==0 or iteration==len(data_loader)-1:
            print('Epoch of adaptation: [{0}][{1}/{2}]\t'
              'Loss {loss_g.val:.4f} ({loss_g.avg:.4f})\t'.format(epoch, iteration+1, len(data_loader),\
              loss_g=losses))


def test_update_m(imgDir, maskDir, saveDir, num_sampling=10):
    """
    apply a hand segmentation model to a directory
    generate intermediate results (prediction, uncertainty map) for next round of training
    uncertainty is computed based on standard variance
    imgDir: filefolder of original images
    """
    iou = AverageMeter()
    f1 = AverageMeter()
    uct_avg = AverageMeter()
    uct_max = AverageMeter()
    netG.train() # set as train mode to enable dropout
    
    if not os.path.exists(saveDir):
        os.system("mkdir %s" % saveDir)
    filenames = [k for k in os.listdir(imgDir) if ".png" in k]
    filenames.sort()

    uct_a = np.zeros(len(filenames))
    img = cv2.imread(imgDir + "/" + filenames[0])
    img_size = (img.shape[1], img.shape[0])
    for iteration in range(0, len(filenames), opt.testBatchSize):
        img_b = []
        img_tensor_b = []
        mask_b = []
        # construct a list of training data
        indices = range(iteration, min(len(filenames), iteration+opt.testBatchSize))
        for index in indices:
            filename = filenames[index]
            # get hand mask (only used for evaluation)
            mask = cv2.imread(maskDir + "/" + filename, 0)
            mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_CUBIC)
            mask = np.expand_dims(mask, axis=2)
            mask_b.append(mask)
            # get input image and transfer to torch tensor
            img = cv2.imread(imgDir + "/" + filename)
            img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
            img_tensor = transform(img)
            img_tensor_b.append(img_tensor)
            img_b.append(img)
        # construct mini-batch tensor from a list of tensors
        img_tensor_b = torch.stack(img_tensor_b)
        # move to GPU
        if opt.cuda:
            img_tensor_b = img_tensor_b.cuda(device=opt.gpu)
        
        
        # multiple stochastic forward
        predicts = []
        with torch.no_grad():   
            for s in range(num_sampling):
                predict = netG(img_tensor_b)
                predicts.append(predict.cpu().numpy())            
        predicts_a = np.array(predicts) # dimension of predicts_a should be [num_sampling, batchsize, channel, width, height]
        
        
        # compute mean and std from multiple stochastic forward
        predict_mean_b = np.mean(predicts_a, axis=0)       
        predict_std_b = np.std(predicts_a, axis=0) # dimension of predict_std should be [batchsize, channel, width, height]        
        
        for b in range(len(indices)):
            predict_mean = predict_mean_b[b]
            predict_std = predict_std_b[b]
            
            std_mean = np.mean(predict_std)            
            std_max = np.amax(predict_std)
            uct_a[iteration+b] = std_mean
            
            uct_max.update(std_max, 1)
            uct_avg.update(std_mean, 1)
    
            predict_mean = np.transpose(predict_mean, (1,2,0)) * 255
            #predict_mean = cv2.GaussianBlur(predict_mean, (5, 5), 0)
            predict_mean = np.clip(predict_mean, 0, 255)
            predict_mean = predict_mean.astype(np.uint8)
            
            # compute segmentation map from probability map
            predict_seg = prob2seg(predict_mean)
            predict_seg = np.expand_dims(predict_seg, axis=2)
            
            # compute IoU
            mask = mask_b[b]
            mask = mask.astype(np.uint8)
            union = np.logical_or(mask>128, predict_mean>128)
            intersection = np.logical_and(mask>128, predict_mean>128)
            M = np.count_nonzero(mask>128)
            P = np.count_nonzero(predict_mean>128)
            #print(filename, 'intersect:', intersection.shape, np.count_nonzero(intersection), 'union:', union.shape, np.count_nonzero(union))
            if np.count_nonzero(union) > 0:
                iou.update(np.count_nonzero(intersection)*1.0/np.count_nonzero(union), 1)
                recall = np.count_nonzero(intersection)*1.0/(M+0.00001)
                precision = np.count_nonzero(intersection)*1.0/(P+0.00001)
                f1.update(2*recall*precision/(recall+precision+0.00001), 1)
            
            # save mean prediction, segmentation
            predict_mean = cv2.resize(predict_mean, img_size, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("%s/%s_p.png" % (saveDir, filenames[iteration+b][:-4]), predict_mean)  
            predict_seg = cv2.resize(predict_seg, img_size, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("%s/%s_s.png" % (saveDir, filenames[iteration+b][:-4]), predict_seg)
            # save uncertainty map and original image
            predict_std = np.transpose(predict_std, (1,2,0)) / std_max*255#0.4291 * 255
            predict_std = np.clip(predict_std, 0, 255)
            predict_std = predict_std.astype(np.uint8)
            #predict_std = cv2.GaussianBlur(predict_std, (5, 5), 0)
            predict_std = cv2.resize(predict_std, img_size, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("%s/%s_u.png" % (saveDir, filenames[iteration+b][:-4]), predict_std)        
            img = cv2.resize(img_b[b], img_size, interpolation = cv2.INTER_CUBIC)
            cv2.imwrite("%s/%s" % (saveDir, filenames[iteration+b]), img)
            
            
    print('Test-m: [{0}]\t'
      'IoU.avg {iou.avg:.4f} ({f1.avg:.4f})\t'
      'Uct.max {uct_max.avg:.4f} \t'
      'Uct.avg {uct_avg.avg:.4f}'.format(len(filenames), iou=iou, f1=f1, uct_max=uct_max, uct_avg=uct_avg))
    return uct_a


def test_update_1(imgDir, maskDir, saveDir):
    """
    apply a hand segmentation model to a directory
    generate prediction map for next round of training
    no use of average prediction from multiple dropout
    """
    iou = AverageMeter()
    f1 = AverageMeter()
    netG.eval() # set as train mode to enable dropout
    
    if not os.path.exists(saveDir):
        os.system("mkdir %s" % saveDir)
    filenames = [k for k in os.listdir(imgDir) if ".png" in k]
    filenames.sort()
    for i, filename in enumerate(filenames):        
        mask = cv2.imread(maskDir + "/" + filename, 0)
        mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_CUBIC)
        mask = np.expand_dims(mask, axis=2)
        img = cv2.imread(imgDir + "/" + filename)
        img_size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC)
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        if opt.cuda:
            img_tensor = img_tensor.cuda(device=opt.gpu)
            
        with torch.no_grad():   
            predict = netG(img_tensor)
            predict = predict.cpu().numpy()[0,:,:,:]
         

        predict = np.transpose(predict, (1,2,0)) * 255
        predict = np.clip(predict, 0, 255)
        predict = predict.astype(np.uint8)
        
        # compute segmentation map from probability map
        predict_seg = prob2seg(predict)
        predict_seg = np.expand_dims(predict_seg, axis=2)
        
        # compute IoU
        union = np.logical_or(mask>128, predict>128)
        intersection = np.logical_and(mask>128, predict>128)
        M = np.count_nonzero(mask>128)
        P = np.count_nonzero(predict>128)
        #print(filename, 'intersect:', intersection.shape, np.count_nonzero(intersection), 'union:', union.shape, np.count_nonzero(union))
        if np.count_nonzero(union) > 0:
            iou.update(np.count_nonzero(intersection)*1.0/np.count_nonzero(union), 1)
            recall = np.count_nonzero(intersection)*1.0/(M+0.0001)
            precision = np.count_nonzero(intersection)*1.0/(P+0.00001)
            f1.update(2*recall*precision/(recall+precision+0.00001), 1)
        
        # save mean prediction, uncertainty map and original image
        predict = cv2.resize(predict, img_size, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("%s/%s_p.png" % (saveDir, filename[:-4]), predict)  
        predict_seg = cv2.resize(predict_seg, img_size, interpolation = cv2.INTER_CUBIC)
        cv2.imwrite("%s/%s_s.png" % (saveDir, filename[:-4]), predict_seg)
        
    print('Test-1: [{0}]\t'
      'IoU.avg {iou.avg:.4f} ({f1.avg:.4f})\t'.format(len(filenames), iou=iou, f1=f1))
            
    
def save_checkpoint(state, filename, save_path):
    torch.save(state, os.path.join(save_path, filename))

def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, epoch)
    torch.save(netG, net_g_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

save_path = '/mnt/hdd3/pytorch-hand/save'   # you can change to your own save path
if not os.path.exists(save_path):
    os.makedirs(save_path)

    
def train_val_update():
    """
    update intermediate pseudo training data after each epoch of train-val
    """    
    iter_start = 1
    iter_end = 11
    
    if iter_start > 1:
        trained_model = "%s/bayes_%s_round%02d.pth.tar" % (save_path, save_code[opt.dataset], iter_start-1)
    else:
        trained_model = "%s/bayes_rf101_egtea_00040.pth.tar" % (save_path) # for pre-trained RefineNet
        #trained_model = "%s/bayes_egtea_00080.pth.tar" % (save_path) # for simpler pix2pix generator (doesn't work well)
    pretrained_dict = torch.load(trained_model)
    pretrained_dict = pretrained_dict['state_dict']    
    model_dict = netG.state_dict()
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)
    
    imgDir = "data/%s/test" % (opt.dataset)
    maskDir = "data/%s/testannot" % (opt.dataset)
    pseudoDir = "result/%s_round%02d" % (save_code[opt.dataset], iter_start-1)
    #test_update_1(imgDir, maskDir, pseudoDir)
    uct_init = test_update_m(imgDir, maskDir, pseudoDir, num_sampling=10)   
    prev_uct_avg = np.mean(uct_init)    
    uct_ratio = uct_init / (np.amax(uct_init)+0.0001) # uct_ratio is not used in the paper, neither here
    uct_ratio = np.array([v+0.0001 for v in uct_ratio])
    for epoch in tqdm(range(iter_start, iter_end)):
        print('dataDir:', pseudoDir)
        torch.manual_seed(opt.seed)  # fixing random seed could make adaptation more stable
        if opt.cuda:
            torch.cuda.manual_seed(opt.seed)
        train_update(epoch, pseudoDir, uct_ratio)
        pseudoDir = "result/%s_round%02d" % (save_code[opt.dataset], epoch)
        #test_update_1(imgDir, maskDir, pseudoDir)        
        uct_a = test_update_m(imgDir, maskDir, pseudoDir, num_sampling=10)
        if np.abs(np.mean(uct_a) - prev_uct_avg) < 0.1 * prev_uct_avg:
            print("Uncertainty-guided model adaptation converges.")
            break
        prev_uct_avg = np.mean(uct_a) 
        uct_ratio = uct_a / (np.amax(uct_a)+0.0001)
        uct_ratio = np.array([v+0.0001 for v in uct_ratio])
        netG_name = "bayes_%s_round%02d.pth.tar" % (save_code[opt.dataset], epoch)
        save_checkpoint({'epoch': epoch, 'arch': 'rgb', 'state_dict': netG.state_dict(), 'optimizer': optimizerG.state_dict(),},
                                netG_name, save_path)
    print('done!')
    
def main(argv):
    """
    """
    train_val_update()
    
if __name__ == "__main__":
    main(sys.argv[1:])
    