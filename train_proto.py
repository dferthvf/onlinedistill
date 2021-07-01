from __future__ import print_function

import os
import argparse
import socket
import time
import sys
import random

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import scipy
from scipy.stats import t
from sklearn import metrics

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter
from eval.meta_eval import meta_test
from eval.cls_eval import validate

import kornia
from kornia import augmentation as augs
from kornia import filters, color

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from policy import Policy

from itertools import chain


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p
    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm)
    return out


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h


def random_rotate(imgs):
    batch_size = imgs.size(0)
    rand = torch.rand(batch_size, 4, device=imgs.device)
    labels = torch.argmax(rand, dim=1) 
    angles = labels * 90
    imgs = kornia.geometry.transform.rotate(imgs, labels)
    return imgs, labels


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0 and np.random.rand(1) < 0.5:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    x_a = x
    y_a, y_b = y, y[index]
    mixed_x = x.clone()
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = mixed_x[index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return mixed_x, y_a, y_b, lam, index


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


@torch.no_grad()
def momentum_update(model, model_ema, ema=0.999):
    for param, param_ema in zip(model.parameters(), model_ema.parameters()):
        param_ema.data = param_ema.data * ema + param.data * (1. - ema)
    for module, module_ema in zip(model.modules(), model_ema.modules()):
        if isinstance(module, nn.BatchNorm2d):
            module_ema.running_mean = module_ema.running_mean * ema + module.running_mean * (1. - ema)
            module_ema.running_var = module_ema.running_var * ema + module.running_var * (1. - ema)


def KL(y_s, y_t, T):
    p_s = F.log_softmax(y_s/T, dim=1)
    p_t = F.softmax(y_t/T, dim=1)
    loss = F.kl_div(p_s, p_t, size_average=False) * (T**2) / y_s.shape[0]
    return loss


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,80', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')

    # dataset
    parser.add_argument('--model', type=str, default='resnet12', choices=model_pool)
    parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet',
                                                                                'CIFAR-FS', 'FC100'])
    parser.add_argument('--transform', type=str, default='A', choices=transforms_list)
    parser.add_argument('--use_trainval', action='store_true', help='use trainval set')

    # cosine annealing
    parser.add_argument('--cosine', action='store_true', help='using cosine annealing')

    # specify folder
    parser.add_argument('--model_path', type=str, default='', help='path to save model')
    parser.add_argument('--tb_path', type=str, default='', help='path to tensorboard')
    parser.add_argument('--data_root', type=str, default='', help='path to data root')

    # meta setting
    parser.add_argument('--n_train_runs', type=int, default=200, metavar='N',
                        help='Number of train runs')
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=1, type=int,
                        help='The number of augmented samples for each meta test sample')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='test_batch_size',
                        help='Size of test batch)')

    parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

    opt = parser.parse_args()

    if opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        opt.transform = 'D'

    if opt.use_trainval:
        opt.trial = opt.trial + '_trainval'

    # set the path according to the environment
    if not opt.model_path:
        opt.model_path = './models_pretrained'
    if not opt.tb_path:
        opt.tb_path = './tensorboard'
    if not opt.data_root:
        opt.data_root = './data/{}'.format(opt.dataset)
    else:
        opt.data_root = '{}/{}'.format(opt.data_root, opt.dataset)
    opt.data_aug = True

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trans_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.transform)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.adam:
        opt.model_name = '{}_useAdam'.format(opt.model_name)

    opt.model_name = '{}_trial_{}'.format(opt.model_name, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.n_gpu = torch.cuda.device_count()

    return opt


def main():

    opt = parse_option()

    # dataloader
    train_partition = 'trainval' if opt.use_trainval else 'train'
    if opt.dataset == 'miniImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        meta_trainloader = DataLoader(MetaImageNet(args=opt, partition='train',
                                                   train_transform=train_trans,
                                                   test_transform=test_trans),
                                      batch_size=opt.batch_size, shuffle=False, drop_last=False,
                                      num_workers=opt.num_workers)

        meta_testloader = DataLoader(MetaImageNet(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaImageNet(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            n_cls = 64
    elif opt.dataset == 'tieredImageNet':
        train_trans, test_trans = transforms_options[opt.transform]
        train_loader = DataLoader(TieredImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(TieredImageNet(args=opt, partition='train_phase_val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaTieredImageNet(args=opt, partition='test',
                                                        train_transform=train_trans,
                                                        test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaTieredImageNet(args=opt, partition='val',
                                                       train_transform=train_trans,
                                                       test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 448
        else:
            n_cls = 351
    elif opt.dataset == 'CIFAR-FS' or opt.dataset == 'FC100':
        train_trans, test_trans = transforms_options['D']

        train_loader = DataLoader(CIFAR100(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(CIFAR100(args=opt, partition='train', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
        meta_testloader = DataLoader(MetaCIFAR100(args=opt, partition='test',
                                                  train_transform=train_trans,
                                                  test_transform=test_trans),
                                     batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                     num_workers=opt.num_workers)
        meta_valloader = DataLoader(MetaCIFAR100(args=opt, partition='val',
                                                 train_transform=train_trans,
                                                 test_transform=test_trans),
                                    batch_size=opt.test_batch_size, shuffle=False, drop_last=False,
                                    num_workers=opt.num_workers)
        if opt.use_trainval:
            n_cls = 80
        else:
            if opt.dataset == 'CIFAR-FS':
                n_cls = 64
            elif opt.dataset == 'FC100':
                n_cls = 60
            else:
                raise NotImplementedError('dataset not supported: {}'.format(opt.dataset))
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = create_model(opt.model, n_cls, opt.dataset)
    # ckpt = torch.load('/media/hhd1/yue/rfs/checkpoint/resnet12_miniImageNet_lr_0.05_decay_0.0005_trans_A_cosine_trial_od_999_200_cutmix_1_weighted_kl_cosine_double_ema/ckpt_epoch_200.pth')
    ckpt = torch.load('/media/hhd1/yue/rfs/checkpoint/mini_simple.pth')
    model.load_state_dict(ckpt['model'], strict=False)

    model_ema = create_model(opt.model, n_cls, opt.dataset)
    model_ema.load_state_dict(ckpt['model'], strict=False)

    # optimizer
    if opt.adam:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=opt.learning_rate,
                                     weight_decay=opt.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), 
                              lr=opt.learning_rate,
                              momentum=opt.momentum,
                              weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
            model_ema = nn.DataParallel(model_ema)
        model = model.cuda()
        model_ema = model_ema.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # set cosine annealing scheduler
    if opt.cosine:
        eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    # routine: proto
    for epoch in range(1, opt.epochs + 1):
        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, meta_trainloader, model, criterion, optimizer, opt, model_ema)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_std = meta_test(model, meta_testloader, use_logit=False, is_norm=True, classifier='Cosine', opt=None)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_std', test_std, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


def train(epoch, train_loader, model, criterion, optimizer, opt, model_ema):
    """One epoch training"""
    model.train()
    model_ema.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    acc = []

    end = time.time()
    for idx, data in enumerate(train_loader):
        support_xs, support_ys, query_xs, query_ys = data
        data_time.update(time.time() - end)
        batch_size, num_support, num_dims, height, width = support_xs.size()
        
        if torch.cuda.is_available():
            support_xs = support_xs.cuda()
            support_ys = support_ys.cuda()
            query_xs = query_xs.cuda()
            query_ys = query_ys.cuda()

        support_xs = support_xs.view(-1, num_dims, height, width)
        query_xs = query_xs.view(-1, num_dims, height, width)

        query_ys = query_ys.view(-1)
        input_mixed, target_a, target_b, lam, index = cutmix_data(query_xs, query_ys)

        support_feat, support_logit = model(support_xs, is_feat=True)
        support_feat = support_feat[-1]

        query_feat = model(input_mixed, is_feat=True)[0][-1]

        # with torch.no_grad():
            # support_logit_ema = model_ema(support_xs, is_feat=True)[-1]

        support_feat = normalize(support_feat)
        query_feat = normalize(query_feat)

        num_dims = support_feat.size(1)
        support_feat = support_feat.view(batch_size, -1, num_dims)
        query_feat = query_feat.view(batch_size, -1, num_dims)

        logit = torch.matmul(query_feat, support_feat.transpose(2, 1)) * model.temp
        
        pred = logit.argmax(axis=-1)

        # loss = criterion(logit.view(-1, 5), query_ys.view(-1)) # + KL(support_logit, support_logit_ema, 4.0)
        loss = mixup_criterion(criterion, logit.view(-1, 5), target_a, target_b, lam)

        losses.update(loss.item(), batch_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        acc.append(metrics.accuracy_score(query_ys.view(-1).detach().cpu().numpy(), pred.view(-1).detach().cpu().numpy()))
        accuracy, std = mean_confidence_interval(acc)
        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {accuracy:.3f} ({std:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, accuracy=accuracy, std=std))
            sys.stdout.flush()

    print(' * Acc@1 {accuracy:.3f} ({std:.3f})'
          .format(accuracy=accuracy, std=std))

    print(model.temp)
    return accuracy, losses.avg


if __name__ == '__main__':
    main()
