# -*- coding: utf-8 -*-
"""
Created on Mon Feb 5 10:50:48 2018

@author: cai-mj
"""

#from __future__ import print_function
import argparse
import os
import time
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn

from net_pix2pix import define_G, GANLoss, define_D_global
from refinenet import rf101
from utils import AverageMeter

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=80, help='number of epochs to train for')
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
print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
    gpu_ids = [opt.gpu]
else:
    gpu_ids = []
        
class HandDataset(Dataset):
    def __init__(self, img_path, mask_path):
        super(HandDataset, self).__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.imgs = [k for k in os.listdir(self.img_path) if ".png" in k]
        self.imgs.sort()
        img_sample = cv2.imread(self.img_path+ '/' + self.imgs[0])
        self.img_size = (img_sample.shape[1], img_sample.shape[0])
                
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.transform = transforms.Compose(transform_list)        

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # get input image
        imgname = self.img_path + '/' + self.imgs[index]
        img = cv2.imread(imgname)
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) 
        img_tensor = self.transform(img)
        # get hand mask image
        maskname = self.mask_path + '/' + self.imgs[index]                
        mask = cv2.imread(maskname, 0)               
        mask = cv2.resize(mask, (256, 256), interpolation = cv2.INTER_CUBIC)  
        mask_tensor = torch.from_numpy(mask)            
        mask_tensor = mask_tensor.unsqueeze(0)
        sample = {'img': img_tensor, 'mask': mask_tensor}
        return sample

class HandMultiDataset(Dataset):
    def __init__(self, datasets, phase):
        super(HandMultiDataset, self).__init__()
        self.imgs = []
        self.masks = []
        self.datasets = datasets
        for dataset in datasets:
            imgDir = "data/%s/%s" % (dataset, phase)
            maskDir = "data/%s/%sannot" % (dataset, phase)
            filenames = [k for k in os.listdir(maskDir) if ".png" in k]
            filenames.sort()
            self.imgs.extend([imgDir + '/' + filename for filename in filenames])
            self.masks.extend([maskDir + '/' + filename for filename in filenames])
        img_sample = cv2.imread(self.imgs[0])
        self.img_size = (img_sample.shape[1], img_sample.shape[0])
                
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize([0.5], [0.5])]
        self.transform = transforms.Compose(transform_list)
        
    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        # get input image
        imgname = self.imgs[index]
        img = cv2.imread(imgname)
        img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) 
        img_tensor = self.transform(img)
        # get hand mask image
        maskname = self.masks[index]                
        mask = cv2.imread(maskname, 0)          
        ret, mask_thres = cv2.threshold(mask,0,1,cv2.THRESH_BINARY)
        mask_thres = cv2.resize(mask_thres, (256, 256), interpolation = cv2.INTER_CUBIC)  
        mask_tensor = torch.from_numpy(mask_thres)            
        mask_tensor = mask_tensor.unsqueeze(0)
        sample = {'img': img_tensor, 'mask': mask_tensor}
        return sample

print('===> Loading datasets')
#train_set = HandDataset("data/"+opt.dataset+"/train", "data/"+opt.dataset+"/trainannot")
#test_set = HandDataset("data/"+opt.dataset+"/test", "data/"+opt.dataset+"/testannot")
datasets = opt.dataset.split("+")
train_set = HandMultiDataset(datasets, "train")
test_set = HandMultiDataset(datasets, "test")
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
# simpler pix2pix generator
#netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', True, gpu_ids) # set dropout layer as True for Bayesian CNN
# refinenet
netG = rf101(opt.output_nc, gpu_ids=gpu_ids, use_dropout=True) # set dropout layer as True for Bayesian CNN
# discriminator
#netD = define_D_global(4, gpu_ids=gpu_ids)#define_D(4, 64, 'batch', False, [0])

criterionL1 = nn.L1Loss()
criterionBCE = nn.BCELoss()
criterionGAN = GANLoss(use_lsgan=True) # BCE loss when use_lsgan=False

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# setup optimizer for GAN since Adam is not suitable for unstable gradient
#optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lr)
#optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lr)

if opt.cuda:
    criterionL1 = criterionL1.cuda(device=opt.gpu)
    criterionBCE = criterionBCE.cuda(device=opt.gpu)
    criterionGAN = criterionGAN.cuda(device=opt.gpu)
        
resume_G = False
if resume_G:
    trained_model = "save/bayes_rf101_egtea_00040.pth.tar"
    #trained_model = "model/00050_netG_egtea.pth.tar"
    pretrained_dict = torch.load(trained_model)
    pretrained_dict = pretrained_dict['state_dict']    
    model_dict = netG.state_dict()
    #pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)
    
    
resume_D = False
if resume_D:
    trained_model = "model/00050_netD_egtea_wb.pth.tar"
    pretrained_dict = torch.load(trained_model)
    pretrained_dict = pretrained_dict['state_dict']    
    model_dict = netD.state_dict()
    #pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netD.load_state_dict(model_dict)

def train(epoch):
    losses = AverageMeter()
    for iteration, sample in enumerate(training_data_loader):
        img = sample['img']
        mask = sample['mask']
        
        if opt.cuda:
            img = img.cuda(device=opt.gpu)
            mask = mask.float().cuda(device=opt.gpu)
        
        # forward
        pred_hand = netG(img)

        optimizerG.zero_grad()
        loss = criterionBCE(pred_hand, mask)   
        loss.backward()
        optimizerG.step()
        
        losses.update(loss.data.item(), opt.batchSize)
        
        if (iteration+1) % 100 ==0:
            print('Epoch: [{0}][{1}/{2}]\t'
              'Loss {loss_g.val:.4f} ({loss_g.avg:.4f})\t'.format(epoch, iteration+1, len(training_data_loader)+1,\
              loss_g=losses))


def train_Gan(epoch):
    losses_g = AverageMeter()
    losses_d = AverageMeter()
    netD.train()
    netG.train()
    for iteration, sample in enumerate(training_data_loader):
        img = sample['img']
        mask = sample['mask']
        
        if opt.cuda:
            img = img.cuda(device=opt.gpu)
            mask = mask.float().cuda(device=opt.gpu)
        
        # forward
        pred_hand = netG(img)
        
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################     
        # clamp parameters to a range
#        for p in netD.parameters():
#            p.data.clamp_(-0.01, 0.01)
        # train with fake
        fake_ab = torch.cat((img, pred_hand), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False, opt.gpu)
        # train with real
        real_ab = torch.cat((img, mask), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True, opt.gpu)        
        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        if losses_d.avg > 0.2:
            optimizerD.zero_grad()
            loss_d.backward()       
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################        
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((img, pred_hand), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True, opt.gpu)
        # Second, G(A) = B
        loss_g_bce = criterionBCE(pred_hand, mask)       
        loss_g = loss_g_gan*0.1 + loss_g_bce
        if losses_d.avg < 0.5:
            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()
        #raw_input()

        losses_g.update(loss_g.data.item(), opt.batchSize)
        losses_d.update(loss_d.data.item(), opt.batchSize)
        if (iteration+1) % 100 ==0:
            print("loss_g_gan: %.4f loss_g_bce: %.4f" % (loss_g_gan.data.item(), loss_g_bce.data.item()*10))
            print('Epoch: [{0}][{1}/{2}]\t'
          'Loss_D {loss_d.val:.4f} ({loss_d.avg:.4f})\t'
          'Loss_G {loss_g.val:.4f} ({loss_g.avg:.4f})\t'.format(epoch, iteration+1, len(training_data_loader)+1,\
          loss_d=losses_d, loss_g=losses_g))


def test(epoch):
    """
    evaluate hand segmentation network
    """
    losses = AverageMeter()
    iou = AverageMeter()
    f1 = AverageMeter()
    netG.eval()
    for i, sample in enumerate(testing_data_loader):
        img = sample['img']
        mask = sample['mask']

        if opt.cuda:
            img = img.cuda(device=opt.gpu)
            mask = mask.float().cuda(device=opt.gpu)
        
        with torch.no_grad():
            prediction = netG(img)
        loss = criterionBCE(prediction, mask)
        losses.update(loss.data.item(), opt.testBatchSize)
        
        prediction = prediction.cpu().numpy()
        mask = mask.cpu().numpy()
        for b in range(prediction.shape[0]):
            outim = prediction[b,:,:,:]
            outim = np.transpose(outim, (1,2,0))
            outim = outim*255
            outim = np.clip(outim, 0, 255)
            outim = outim.astype(np.uint8)        
            gt = mask[b,:,:,:]
            gt = np.transpose(gt, (1,2,0))
            gt = gt*255
            gt = np.clip(gt, 0, 255)
            gt = gt.astype(np.uint8)
            union = np.logical_or(gt>128, outim>128)
            intersection = np.logical_and(gt>128, outim>128)
            M = np.count_nonzero(gt>128)
            P = np.count_nonzero(outim>128)
            if np.count_nonzero(union) > 0:
                iou.update(np.count_nonzero(intersection)*1.0/np.count_nonzero(union), 1)
                recall = np.count_nonzero(intersection)*1.0/(M+0.0001)
                precision = np.count_nonzero(intersection)*1.0/(P+0.00001)
                f1.update(2*recall*precision/(recall+precision+0.00001), 1)
        
    print('Test: [{0}/{1}]\t'
      'Loss.avg {loss.avg:.4f} \t'
      'IoU.avg {iou.avg:.4f} ({f1.avg:.4f})\t'.format(len(testing_data_loader), len(testing_data_loader),\
          loss=losses, iou=iou, f1=f1))

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

save_path = 'save' # you can change to your own save path
if not os.path.exists(save_path):
    os.makedirs(save_path)
    
for epoch in tqdm(range(1, opt.nEpochs+1)):
    train(epoch)
    test(epoch)
    netG_name = "%s_%05d.pth.tar" % ("netG", epoch)
    if epoch % 10 == 0:
        save_checkpoint({'epoch': epoch, 'arch': 'rgb', 'state_dict': netG.state_dict(), 'optimizer': optimizerG.state_dict(),},
                            netG_name, save_path)
#    netD_name = "%s_%05d.pth.tar" % ("netD", epoch)
#    if epoch % 1 == 0:
#        save_checkpoint({'epoch': epoch, 'arch': 'rgb', 'state_dict': netD.state_dict(), 'optimizer': optimizerD.state_dict(),},
#                            netD_name, save_path)
print('done!')
