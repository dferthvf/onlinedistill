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

from models import model_pool
from models.util import create_model

from dataset.mini_imagenet import ImageNet, MetaImageNet
from dataset.tiered_imagenet import TieredImageNet, MetaTieredImageNet
from dataset.cifar import CIFAR100, MetaCIFAR100
from dataset.transform_cfg import transforms_options, transforms_list

from util import adjust_learning_rate, accuracy, AverageMeter
from eval.meta_eval import meta_test
from eval.cls_eval import validate

import numpy as np
from PIL import Image
import torchvision.transforms as transforms

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


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--eval_freq', type=int, default=10, help='meta-eval frequency')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
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
    parser.add_argument('--n_test_runs', type=int, default=600, metavar='N',
                        help='Number of test runs')
    parser.add_argument('--n_ways', type=int, default=5, metavar='N',
                        help='Number of classes for doing each classification run')
    parser.add_argument('--n_shots', type=int, default=1, metavar='N',
                        help='Number of shots in test')
    parser.add_argument('--n_queries', type=int, default=15, metavar='N',
                        help='Number of query in test')
    parser.add_argument('--n_aug_support_samples', default=5, type=int,
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
        train_loader = DataLoader(ImageNet(args=opt, partition=train_partition, transform=train_trans),
                                  batch_size=opt.batch_size, shuffle=True, drop_last=True,
                                  num_workers=opt.num_workers)
        val_loader = DataLoader(ImageNet(args=opt, partition='val', transform=test_trans),
                                batch_size=opt.batch_size // 2, shuffle=False, drop_last=False,
                                num_workers=opt.num_workers // 2)
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
        val_loader = DataLoader(TieredImageNet(args=opt, partition='val', transform=test_trans),
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
    model_ema = create_model(opt.model, n_cls, opt.dataset)


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

    # routine: supervised pre-training
    for epoch in range(1, opt.epochs + 1):
        if opt.cosine:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        kl_weight = epoch * 1.0 / opt.epochs
        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt, model_ema, policy=None, kl_weight=kl_weight)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_acc_top5', test_acc_top5, epoch)
        logger.log_value('test_loss', test_loss, epoch)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
                'model_ema': model_ema.state_dict() if opt.n_gpu <= 1 else model_ema.module.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # save the last model
    state = {
        'opt': opt,
        'model': model.state_dict() if opt.n_gpu <= 1 else model.module.state_dict(),
        'model_ema': model_ema.state_dict() if opt.n_gpu <= 1 else model_ema.module.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)


def train(epoch, train_loader, model, criterion, optimizer, opt, model_ema, policy, kl_weight=1.0):
    """One epoch training"""
    model.train()
    model_ema.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kl_losses = AverageMeter()
    d_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, (input_view1, input_view2, input_jigsaw, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        batch_size = input_view1.size(0)

        input_view1 = input_view1.float()
        input_view2 = input_view2.float()

        if torch.cuda.is_available():
            input_view1 = input_view1.cuda()
            input_view2 = input_view2.cuda()
            target = target.cuda()

        output = model(input_view1)

        with torch.no_grad():
            output_ema = model_ema(input_view1).detach()            

        # ===================mixed up loss===============
        ce_loss = criterion(output, target)
        kl_loss = KL(output, output_ema, 4.0) 

        loss = ce_loss * (1-kl_weight) + (kl_loss) * kl_weight 

        acc1, acc5 = accuracy(output_ema, target, topk=(1, 5))
        losses.update(loss.item(), input_view1.size(0))
        kl_losses.update(kl_loss.item(), input_view1.size(0))
        top1.update(acc1[0], input_view1.size(0))
        top5.update(acc5[0], input_view1.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():   
            momentum_update(model, model_ema)

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'KL_Loss {kl_loss.val:.4f} ({kl_loss.avg:.4f})\t'
                  'D_Loss {d_loss.val:.4f} ({d_loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, kl_loss=kl_losses, d_loss=d_losses, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


if __name__ == '__main__':
    main()
