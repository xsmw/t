#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import sys
import os

import shutil
import time
#import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
#from torchvision.utils import make_grid
import skimage
import dataloader
import lossfuns
import models
from trainer import StereoTrainer
import gather

import logging
logger = logging.getLogger(__name__)


visualize_disp = False # True # 

model_by_name = models.model_by_name
dataloader_by_name = dataloader.dataloader_by_name
preprocess = dataloader.preprocess
Ldict = gather.Ldict
SmoothGather = gather.SmoothGather
AverageGather = gather.AverageGather


def train_val(args):

    # ------------- trainer( model, lossfun) and optimzer ------------------------------------ #
    lossfun = lossfuns.StereoLoss(args)
    model = model_by_name(args)
    log_info_model(model, args.dir_save)

    trainer = StereoTrainer(model, lossfun)
    if args.cuda: # carry model to cuda
        trainer = nn.DataParallel(trainer).cuda() # , [0]
        model = trainer.module.model
    
    param_groups = model.get_parameters(args.lr, args.weight_decay)
    optimizer = torch.optim.Adam(param_groups, lr=args.lr, betas=args.beta)

    # --------------- DataLoader ------------------------------------------------------------ #
    tn = 3 if lossfun.flag_supervised else 2
    TrainImgLoader = dataloader_by_name(
                        args.datas_train, args.dir_datas_train, args.bn, training=True, 
                        crop_size=args.crop_size, n=tn)
    ValImgLoader = dataloader_by_name(
                        args.datas_val, args.dir_datas_val, args.bn, training=False, 
                        crop_size=[0, 0], n=3)

    # --------------- load pretrained weight ------------------------------------------------ #
    if os.path.exists(str(args.loadmodel)):
        
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        msg = 'Load pretrained weight successfully ! \nLoaded weight file: %s\n' % str(args.loadmodel)
        logger.info(msg)

    elif('None' != str(args.loadmodel).strip()):
        
        msg = 'No available weight ! \nPlease check weight file[ %s ]\n\n' % str(args.loadmodel)
        logger.warning(msg)

    # ---------------- recover from last interrupt -------------------------------------------- #
    dir_save = args.dir_save # dirpath for saving weight
    if(not os.path.isdir(dir_save)):
        os.makedirs(dir_save)
    
    # ---- recover validation meters of last interrupt
    path_training_info = os.path.join(dir_save, 'training_info.pkl')
    epoch0_train, epoch0_val = 0, 0
    meters_all = Ldict()
    meters_epoch = {}
    epe_best = 10000000.0
    
    if(os.path.isfile(path_training_info)):
        
        info_dict = torch.load(path_training_info)
        meters_all.extend(info_dict)
        epoch0_val = meters_all.data['epoch'][-1]
        epe_best = meters_all.data['epe_best'][-1]
    
    # ---- recover state data of last interrupt
    path_checkpoint = os.path.join(dir_save, 'checkpoint.pkl')

    if(os.path.isfile(path_checkpoint)):
        
        data = torch.load(path_checkpoint)
        epoch0_train = data['epoch']
        meters_epoch['train_loss'] = data['train_loss']
        model.load_state_dict(data['state_dict']) # recover model state
        optimizer.load_state_dict(data['state_dict_optim']) # recover optimizer state
        torch.random.set_rng_state(data['random_state']) # recover random state

    # ---- log current process
    msg = 'Trained %d epoch | Valed %d epoch \n' % (epoch0_train, epoch0_val)
    logger.info(msg)
    
    # --------- training ---------------------------------------------------------------- #
    start_full_time = time.time()
    for epoch in range(epoch0_val+1, args.epochs+1):
        
        # --------- adjust learing rate ------------------------------ #
        lr = lr_adjust(optimizer, epoch, args)
        #lr=0.001
        msg = 'This is %d-th epoch, lr=%f' % (epoch, lr)
        logger.info(msg)
        meters_epoch.update({'epoch': epoch})

        # -------- epoch of Train ----------------------------------- #
        if(epoch > epoch0_train):
            
            meters = epoch_train(args, epoch, TrainImgLoader, trainer, optimizer)
            meters_epoch.update(meters) # record train meters in a epoch
            
            # log current process
            full_time_hour = (time.time() - start_full_time)/3600
            msg = 'Train %d | full training time = %.2f HR \n' % (epoch, full_time_hour)
            logger.info(msg)

            # save weight
            if (0 == epoch%args.freq_save):
                path_weight_epoch = os.path.join(dir_save, 'weight_%d.pkl' % epoch)
                torch.save({'state_dict': model.state_dict()}, path_weight_epoch + '.tm~')
                shutil.move(path_weight_epoch+'.tm~', path_weight_epoch)

            if (0 == args.epochs-epoch):
                path_weight_final = os.path.join(dir_save, 'weight_final.pkl')
                torch.save({'state_dict': model.state_dict()}, path_weight_final + '.tm~')
                shutil.move(path_weight_final+'.tm~', path_weight_final)

            # save checkpoint
            state_dict = {'epoch': epoch,
                          'train_loss': meters['train_loss'], 
                          'state_dict': model.state_dict(),
                          'state_dict_optim': optimizer.state_dict(),
                          'random_state': torch.random.get_rng_state(), 
                          }

            torch.save(state_dict, path_checkpoint + '.tm~')
            shutil.move(path_checkpoint+'.tm~', path_checkpoint)

        #--------- epoch of Val-------------------------------------- #
        if (True):

            meters = epoch_val(args, epoch, ValImgLoader, model)
            meters_epoch.update(meters) # record Val meters in a epoch

            # save weight with best epe during validatin
            assert 'val_epe' in meters.keys(), str(meters.keys())
            if (epe_best > meters['val_epe']):
                epe_best = meters['val_epe']
                path_weight_best = os.path.join(dir_save, 'weight_best.pkl')
                torch.save({'state_dict': model.state_dict()}, path_weight_best + '.tm~')
                shutil.move(path_weight_best+'.tm~', path_weight_best)
            meters_epoch.update({'epe_best': epe_best})

            # log current process
            full_time_hour = (time.time() - start_full_time)/3600
            msg = 'Val %d | full training time = %.2f HR \n' % (epoch, full_time_hour)
            logger.info(msg)

        # record train meters in a epoch
        meters_all.append(meters_epoch)
        meters_epoch = {}

        # save training information
        torch.save(meters_all.data, path_training_info+'.tm~')
        shutil.move(path_training_info+'.tm~', path_training_info)


def val(args):

    # ------------- model ------------------------------------------------------------------- #
    model = model_by_name(args)
    log_info_model(model, args.dir_save)

    if args.cuda: # carry model to cuda
        model = model.cuda()

    # --------------- DataLoader ------------------------------------------------------------ #
    ValImgLoader = dataloader_by_name(
                        args.datas_val, args.dir_datas_val, args.bn, training=False, 
                        crop_size=[0, 0], n=3)
    

    # --------------- load pretrained weight ------------------------------------------------ #
    if os.path.exists(str(args.loadmodel)):
        
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        msg = 'Load pretrained weight successfully ! \nLoaded weight file: %s\n' % str(args.loadmodel)
        logger.info(msg)

    elif('None' != str(args.loadmodel).strip()):
        
        msg = 'No available weight ! \nPlease check weight file[ %s ]\n\n' % str(args.loadmodel)
        logger.warning(msg)

    # --------- validation ---------------------------------------------------------------- #
    start_full_time = time.time()
    meters = epoch_val(args, 0, ValImgLoader, model)

    # ---- log overview of validatin
    full_time_hour = (time.time() - start_full_time)/3600
    msg = 'Full Val time = %.2f HR \n' % (full_time_hour)
    logger.info(msg)
    msg = '\n' + '\n'.join(['%s: %.3f'%(k, v) for k, v in meters.items()])
    logger.info(msg)


def submission(args):

    # ------------- model ------------------------------------------------------------------- #
    model = model_by_name(args)
    log_info_model(model, args.dir_save)

    if args.cuda: # carry model to cuda
        model = model.cuda()

    # --------------- DataLoader ------------------------------------------------------------ #
    ValImgLoader = dataloader_by_name(
                        args.datas_val, args.dir_datas_val, args.bn, training=False, 
                        crop_size=None, n=3)

    # -------------- load pretrained weight ------------------------------------------------- #
    if os.path.exists(str(args.loadmodel)):
        
        state_dict = torch.load(args.loadmodel)
        model.load_state_dict(state_dict['state_dict'])
        msg = 'Load pretrained weight successfully ! \nLoaded weight file: %s\n' % str(args.loadmodel)
        logger.info(msg)

    elif('None' != str(args.loadmodel).strip()):
        
        msg = 'No available weight ! \nPlease check weight file[ %s ]\n\n' % str(args.loadmodel)
        logger.warning(msg)

    # -------------- submission ------------------------------------------------------------ #
    dir_save = args.dir_save
    if(not os.path.isdir(dir_save)):
        os.makedirs(dir_save)
    start_full_time = time.time()
    meters = epoch_submission(args, ValImgLoader, model)

    # ---- log overview of submission
    full_time_hour = (time.time() - start_full_time)/3600
    msg = 'Full Submission time = %.2f HR \n' % (full_time_hour)
    logger.info(msg)
    msg = '\n' + '\n'.join(['%s: %.3f'%(k, v) for k, v in meters.items()])
    logger.info(msg)


def epoch_train(args, epoch, dataloader, trainer, optimizer):

    torch.cuda.empty_cache()
    # Meters
    #loss = SmoothGather(delay=0.99)
    loss = AverageGather()
    batch_time = AverageGather()
    data_time = AverageGather()
    
    # start
    start_time = time.time()
    iter_all = len(dataloader)*args.nloop
    for i in range(args.nloop):
        for batch_idx, batch in enumerate(dataloader):
            
            batch = rand_margin(batch[1:])
            bn = batch[0].shape[0]
            if args.cuda: # carry data to GPU
                batch = [t.cuda() for t in batch]
            data_time.update(time.time()-start_time, bn) # record time of loading data

            # train on a batch of sample
            flag_step = (batch_idx % args.freq_optim == 0) or (iter_all-1 == batch_idx)
            meters = step_train(batch, trainer, optimizer, flag_step)
            loss.update(meters['loss'], bn) # record train loss
            batch_time.update(time.time()-start_time, bn) # record time of a batch
            
            # log msg
            iter = batch_idx + i*len(dataloader)
            if(iter % args.freq_print == 0):
                msg = ('Train [{0}|{1:3d}/{2}] | '
                       'Time {batch_time.avg:.3f}({data_time.avg:.3f}) | '
                       'Loss {loss.val:6.3f} ({loss.avg:6.3f}) '
                       ''.format(epoch, iter, iter_all, batch_time=batch_time, 
                                data_time=data_time, loss=loss))
                logger.info(msg)
            
            # reset start_time
            start_time = time.time()
            
    # log msg
    msg = 'Train %d | mean loss = %.3f | mean batch_time = %.3f ' % (
            epoch, loss.avg, batch_time.avg)
    logger.info(msg)
    
    # return meters
    meters = {'train_loss': loss.avg }
    return meters


def epoch_val(args, epoch, dataloader, model):

    torch.cuda.empty_cache()
    # Meters
    err_rpe = AverageGather()
    err_epe = AverageGather()
    err_d1 = AverageGather()
    err_1px = AverageGather()
    err_2px = AverageGather()
    err_3px = AverageGather()
    err_4px = AverageGather()
    err_5px = AverageGather()
    batch_time = AverageGather()
    data_time = AverageGather()
    
    # start
    start_time = time.time()
    iter_all = len(dataloader)
    for batch_idx, (filename, imgL, imgR, dispL) in enumerate(dataloader):
        
        bn = imgL.shape[0]
        if args.cuda: # carry data to GPU
            imgL, imgR, dispL = imgL.cuda(), imgR.cuda(), dispL.cuda()
        data_time.update(time.time()-start_time, bn) # record data_time

        # validatin on a batch of sample
        meters, dispL_pred = step_val(model, imgL, imgR, dispL)
        err_rpe.update(meters['err_rpe'], bn) # record err_rpe
        err_epe.update(meters['err_epe'], bn) # record err_epe
        err_d1.update(meters['err_d1'], bn) # record err_d1
        err_1px.update(meters['err_1px'], bn) # record err_1px
        err_2px.update(meters['err_2px'], bn) # record err_2px
        err_3px.update(meters['err_3px'], bn) # record err_3px
        err_4px.update(meters['err_4px'], bn) # record err_4px
        err_5px.update(meters['err_5px'], bn) # record err_5px
        batch_time.update(time.time()-start_time, bn) # record batch_time
        
        # save visual result
        if('val'==args.mode.lower() and batch_idx<15):
            dir_save = args.dir_save
            path_disp = os.path.join(dir_save, 'disp_%02d.png' % batch_idx)
            path_disp_true = os.path.join(dir_save, 'disp_true_%02d.png' % batch_idx)
            path_err = os.path.join(dir_save, 'err_%02d(%.2f,%.2f).png' % (batch_idx, meters['err_epe'], meters['err_d1']))
            delt = (dispL_pred[0, 0] - dispL[0, 0]).abs().clamp(0, 3)
            mask = (dispL[0, 0]>0)&(dispL[0, 0]<args.maxdisp)
            delt[~mask] = 0
            plt.imsave(path_disp, dispL_pred[0, 0].cpu().data.numpy())
            plt.imsave(path_disp_true, dispL[0, 0].cpu().data.numpy())
            plt.imsave(path_err, delt.cpu().data.numpy())
#        # select a batch of sample for visualization
#        if (0 == batch_idx) and (args.dir_save is not None):
#            dispL_select = {'dispL_pred': dispL_pred[:1].cpu().data, 
#                            'dispL_true': dispL[:1].cpu().data}
#            path = os.path.join(args.dir_save, 'dispL_val_%02d_%d.pkl' % (epoch, batch_idx))
#            path_tmp = path + '.tm~'
#            torch.save(dispL_select, path_tmp)
#            shutil.move(path_tmp, path)
        
        # log msg
        if(batch_idx % args.freq_print == 0):
            iter = batch_idx
            msg = ('Val [{0}|{1:3d}/{2}] | '
                   'Time {batch_time.avg:.3f}({data_time.avg:.3f}) | '
                   'epe, d1 = {err_epe.avg:6.3f}, {err_d1.avg:6.3f}'
                   ''.format(epoch, iter, iter_all, batch_time=batch_time, 
                            data_time=data_time, err_epe=err_epe, err_d1=err_d1))
            logger.info(msg)
    
        # reset start_time
        start_time = time.time()

    # log msg
    msg = 'Val %d | batch_time = %.3f | epe, d1, rpe = %.3f, %.3f, %.3f ' % (
            epoch, batch_time.avg, err_epe.avg, err_d1.avg, err_rpe.avg, ) + \
            '\n npx(1-5) = [%.3f, %.3f, %.3f, %.3f, %.3f]' % (
            err_1px.avg, err_2px.avg, err_3px.avg, err_4px.avg, err_5px.avg, )
    logger.info(msg)
    
    # return meters
    meters = {'val_rpe': err_rpe.avg, 
              'val_epe': err_epe.avg, 
              'val_d1': err_d1.avg, 
              'val_1px': err_1px.avg, 
              'val_2px': err_2px.avg, 
              'val_3px': err_3px.avg, 
              'val_4px': err_4px.avg, 
              'val_5px': err_5px.avg, 
              'val_time': batch_time.avg}
    return meters


def epoch_submission(args, dataloader, model):

    torch.cuda.empty_cache()
    # Meters
    runtime = AverageGather()
    batch_time = AverageGather()
    data_time = AverageGather()
    
    # start 
    start_time = time.time()
    iter_all = len(dataloader)
    for batch_idx, (filename, imgL, imgR) in enumerate(dataloader):
        
        bn = imgL.shape[0]
        if args.cuda: # carry data to GPU
            imgL, imgR = imgL.cuda(), imgR.cuda()
        data_time.update(time.time()-start_time, bn) # record time of loading data

        # Submission on a batch of sample
        path_save = os.path.join(args.dir_save, filename[0])
        meters = step_submission(model, imgL, imgR, path_save)
        runtime.update(meters['runtime'], bn) # record runtime
        batch_time.update(time.time()-start_time, bn) # record batch_time

        # log msg
        if(batch_idx % args.freq_print == 0):
            iter = batch_idx
            msg = ('Submission [{0:3d}/{1}] | '
                   'Time {batch_time.avg:6.3f}({data_time.avg:6.3f}) | '
                   'Runtime {runtime.val:6.3f} ({runtime.avg:6.3f}) '
                   ''.format(iter, iter_all, batch_time=batch_time, 
                            data_time=data_time, runtime=runtime))
            logger.info(msg)
        
        # reset start_time
        start_time = time.time()

    # log msg
    msg = 'Submission | mean Runtime = %.3f | mean batch_time = %.3f' % (
            runtime.avg, batch_time.avg)
    logger.info(msg)
    
    # return meters
    meters = {'runtime': runtime.avg, 
              'batch_time': batch_time.avg,}
    return meters


def step_train(batch, trainer, optimizer, flag_optim):

    trainer.train()

    # backward
    loss = trainer(batch).mean()
    if(loss > 0):
        loss.backward()

    # update weight
    if(flag_optim):
        optimizer.step()
        optimizer.zero_grad()
    # return
    meters = {'loss': loss.item(), }
    return meters


def step_val(model, imgL, imgR, dispL):

    model.eval()
    
    # predict disp
    with torch.no_grad():
        dispL_pred = model(*preprocess_color(imgL, imgR, False))
    maxdisp = model.maxdisp

    # evaluate
    meters_dict = evaluate(imgL, imgR, dispL_pred, dispL, maxdisp)
    return meters_dict, dispL_pred


def step_submission(model, imgL, imgR, path_save):

    model.eval()

    # predict disp
    imgL, imgR = preprocess_color(imgL, imgR, False)
    start_time = time.time()
    with torch.no_grad():
        dispL_pred = model(imgL, imgR).cpu().data.numpy()
    runtime = time.time() - start_time
    
    # save result
    img = dispL_pred[0, 0]
    skimage.io.imsave(path_save, (img*256).astype('uint16'))

    # return
    meters_dict = {'runtime': runtime}
    return meters_dict


def log_info_model(model, dir_save):

    # save the text of model's Modules
    num_model_parameters = sum([p.data.nelement() for p in model.parameters()])
    msg = 'Modules of model: \n{0} \n\n Number of model parameters: {1}'.format(
            str(model), num_model_parameters)
    path_model = 'model_{}.txt'.format(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())) 
    f = open(os.path.join(dir_save, path_model), 'w')
    f.write(msg)
    f.close()


def lr_adjust(optimizer, epoch, args):
    
    lr = args.lr
    # lr_warmup
    if(epoch < args.epochs_warmup):
        lr *= 0.1
        
    # lr_decayed
    depoch = epoch - args.lr_epoch0
    if(depoch >= 0):
        count = 1 + ((epoch-args.lr_epoch0)//args.lr_stride)
        lr *= args.lr_decay**count
    
    # lr_adjust
    lr_mult = lr/optimizer.defaults['lr']
    if(0.001 < abs(1 - lr_mult)):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
        optimizer.defaults['lr'] = lr

    return lr


def rand_margin(batch):
    
    mh, mw = torch.randint(0,64,(2,))
    for i in range(len(batch)):
        batch[i] = batch[i][..., mh:, mw:]
    return batch


def preprocess_color(imgL, imgR, augment=False):
    process = [preprocess.augment_color(augment) for i in range(imgL.size(0))]
    imgL = torch.cat(list(map(lambda f, x: f(x)[None], process, imgL)), dim=0)
    imgR = torch.cat(list(map(lambda f, x: f(x)[None], process, imgR)), dim=0)
    return imgL, imgR


def evaluate(imgL, imgR, disp_pred, disp_true, maxdisp):
    
    # select valid pixels
    mask = (disp_true > 0) & (disp_true < maxdisp)
    count_valid = len(mask[mask])
    if count_valid == 0:
       return {'err_epe': 0, 'err_d1': 0, 'err_1px': 0, 'err_2px': 0, 'err_3px': 0, 'err_4px': 0, 'err_5px': 0, }

    # computing end-point-error
    disp_err = torch.abs(disp_pred[mask] - disp_true[mask])
    err_epe = torch.mean(disp_err)
    if(type(err_epe) is torch.Tensor):
        err_epe = err_epe.item() 

    # computing err_d1
    mask_d1 = ((disp_err < 3) | (disp_err < disp_true[mask]*0.05))
    err_d1 = 100*(1.0 - float(len(mask_d1[mask_d1]))/count_valid)

    # computing err_npx
    err_npx = []
    for n in range(1, 6):
        mask_npx = (disp_err < n)
        err_npx.append(100*(1.0 - float(len(mask_npx[mask_npx]))/count_valid))
    err_1px, err_2px, err_3px, err_4px, err_5px = err_npx

    # computing rpe(reconstruction pixel error)
    imL_wrap = imwrap(imgR, disp_pred*(2.0/disp_pred.size(-1)))
    mask_imL = ((imL_wrap.sum(dim=1, keepdim=True)!=0) & mask)
    if(len(mask_imL[mask_imL]) > 0):
        imL_diff = (imL_wrap - imgL).abs().mean(dim=1, keepdim=True)[mask_imL].mean()
        err_rpe = imL_diff.item()*255.0
    else:
        err_rpe = 0

    # visualize_disp
    if(visualize_disp):
        title = 'epe=%.2f, D1=%.2f, %s'%(err_epe, err_d1, str(list(mask.shape)))
        visualize_disp_mask(imgL, imgR, disp_pred, disp_true, mask, title)
    
    # return
    meters_dict = {'err_rpe': err_rpe, 'err_epe': err_epe, 'err_d1': err_d1, 'err_1px': err_1px, 
                    'err_2px': err_2px, 'err_3px': err_3px, 'err_4px': err_4px, 'err_5px': err_5px, }
    return meters_dict


def imwrap(imR, dispL_norm, rect={'xs': -1, 'xe':1, 'ys':-1, 'ye':1}):
    '''
    Wrap right image to left view according to normal left disparity 
    
    imwrap(imR, dispL_norm, rect={'xs': -1, 'xe':1, 'ys':-1, 'ye':1}) --> imL_wrap
    
    Args:

        imR: the right image, with shape of [bn, c , h0, w0]
        dispL_norm: normal left disparity, with shape of [bn, 1 , h, w]
        rect: the area of left image for the dispL_norm, 
              consist the keys of ['xs', 'xe', 'ys', 'ye'].
              'xs': start position of width direction,
              'xe': end position of width direction,
              'ys': start of height direction,
              'ye': end of height direction,
              such as rect={'xs': -1, 'xe':1, 'ys':-1, 'ye':1} for all area.
    
    Examples:

        >>> imR = torch.rand(1, 3, 32, 32)
        >>> dispL = torch.ones(1, 1, 16, 16)*0.1
        >>> rect = {'xs': -0.5, 'xe':0.5, 'ys':-0.5, 'ye':0.5}
        >>> w_imL = imwrap(imR, dispL, rect)
        >>> w_imL.shape[-2:] == dispL.shape[-2:]
        True
    '''

    # get shape of dispL_norm
    bn, c, h, w = dispL_norm.shape
    
    # create sample grid
    row = torch.linspace(rect['xs'], rect['xe'], w)
    col = torch.linspace(rect['ys'], rect['ye'], h)
    grid_x = row[:, None].expand(bn, h, w, 1)
    grid_y = col[:, None, None].expand(bn, h, w, 1)
    grid = torch.cat([grid_x, grid_y], dim=-1).type_as(dispL_norm)
    grid[..., 0] = (grid[..., 0] - dispL_norm.squeeze(1))
    
    # sample image by grid
    imL_wrap = F.grid_sample(imR, grid)
    
    # refill 0 for out-of-bound grid locations
    mask = (grid[..., 0]<-1)|(grid[..., 0]>1)
    imL_wrap[mask[:, None].expand_as(imL_wrap)] = 0
    
    return imL_wrap


def visualize_disp_mask(imL, imR, disp_pred, disp_true, mask, title):
    
    to_numpy = lambda tensor: tensor.data.cpu().numpy()

    plt.subplot(3, 2, 1); plt.imshow(to_numpy(imL[0]).transpose(1, 2, 0))
    plt.title(title)
    
    plt.subplot(3, 2, 2); plt.imshow(to_numpy(imR[0]).transpose(1, 2, 0))
    plt.title(title)

    plt.subplot(3, 2, 3); plt.imshow(to_numpy(disp_pred[0, 0]))
    plt.title(title)

    plt.subplot(3, 2, 4); plt.imshow(to_numpy(disp_true[0, 0]))
    plt.title(title)

    delt = (disp_pred[0, 0] - disp_true[0, 0]).abs().clamp(0, 3)
    delt[~mask[0, 0]] = 0
    plt.subplot(3, 2, 5); plt.imshow(to_numpy(delt))
    plt.title(title)

    plt.show()


