"""
efficienttrack.py
=================
EfficientTrack convenience class, can be used to train and troublshoot the
EfficientTrack module.
"""

import os
import numpy as np
from tqdm.autonotebook import tqdm
import traceback
import cv2
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
import onnx

from .model import EfficientTrackBackbone
from .loss import MultiLossFactory
import lib.vortex.modules.efficienttrack.utils as utils
from lib.logger.logger import NetLogger, AverageMeter
import lib.vortex.modules.efficienttrack.darkpose as darkpose


class EfficientTrack:
    """
    EfficientTrack convenience class, enables easy training and inference with
    using the EfficientTrack Module.

    :param mode: Select wether the network is loaded in training or inference
                 mode
    :type mode: string
    :param cfg: Handle for the global configuration structure
    :param weights: Path to parameter savefile to be loaded
    :type weights: string, optional
    """
    def __init__(self, mode, cfg, weights = None, run_name = None):
        self.mode = mode
        self.cfg = cfg
        self.model = EfficientTrackBackbone(self.cfg, compound_coef=self.cfg.EFFICIENTTRACK.COMPOUND_COEF)

        if mode  == 'train':
            if run_name == None:
                run_name = "Run_" + time.strftime("%Y%m%d-%H%M%S")
            self.model_savepath = os.path.join(self.cfg.savePaths['efficienttrack'], run_name)
            os.makedirs(self.model_savepath, exist_ok=True)

            self.logger = NetLogger(os.path.join(self.cfg.logPaths['efficienttrack'], run_name))
            self.lossMeter = AverageMeter()
            self.accuracyMeter = AverageMeter()

            self.load_weights(weights)

            self.criterion = MultiLossFactory(self.cfg)

            if self.cfg.NUM_GPUS > 0:
                self.model = self.model.cuda()
                if self.cfg.NUM_GPUS > 1:
                    self.model = utils.CustomDataParallel(self.model, self.cfg.NUM_GPUS)

            if self.cfg.EFFICIENTTRACK.OPTIMIZER == 'adamw':
                self.optimizer = torch.optim.AdamW(self.model.parameters(), self.cfg.EFFICIENTTRACK.LEARNING_RATE)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(), self.cfg.EFFICIENTTRACK.LEARNING_RATE, momentum=0.9, nesterov=True)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True, min_lr=0.00005, factor = 0.2)

        elif mode == 'inference':
            self.model.requires_grad_(False)
            self.model.load_state_dict(torch.load(weights))
            self.model.requires_grad_(False)
            self.model.eval()
            self.model = self.model.cuda()

        elif mode == 'export':
            self.model = EfficientTrackBackbone(num_classes=len(self.cfg.DATASET.OBJ_LIST), compound_coef=self.cfg.EFFICIENTTRACK.COMPOUND_COEF, onnx_export = True)
            self.load_weights(weights)
            self.model = self.model.cuda()



    def load_weights(self, weights_path = None):
        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path), strict=True)
            print(f'Successfully loaded weights: {os.path.basename(weights_path)}')
        else:
            print('Initializing weights...')
            utils.init_weights(self.model)


    def freeze_backbone(self):
        classname = self.model.__class__.__name__
        for ntl in ['EfficientNet', 'BiFPN']:
            if ntl in classname:
                for param in self.model.parameters():
                    param.requires_grad = False


    def train(self, training_set, validation_set, num_epochs, start_epoch = 0):
        """
        Function to train the network on a given dataset for a set number of epochs.
        Most of the training parameters can be set in the config file.

        :param training_set: training dataset
        :type training_generator: TODO
        :param val_generator: validation dataset
        :type validation_set: TODO
        :param num_epochs: Number of epochs the network is trained for
        :type num_epochs: int
        :param start_epoch: Initial epoch for the training, set this if training
            is continued from an earlier session
        """
        training_generator = DataLoader(training_set,
                                        batch_size = self.cfg.EFFICIENTTRACK.BATCH_SIZE,
                                        shuffle = True,
                                        num_workers =  self.cfg.DATALOADER_NUM_WORKERS,
                                        pin_memory = True)

        val_generator = DataLoader(validation_set,
                                        batch_size = self.cfg.EFFICIENTTRACK.BATCH_SIZE,
                                        shuffle = False,
                                        num_workers =  self.cfg.DATALOADER_NUM_WORKERS,
                                        pin_memory = True)

        epoch = start_epoch
        best_loss = 1e5
        best_epoch = 0
        self.model.train()
        if self.cfg.USE_MIXED_PRECISION:
            scaler = torch.cuda.amp.GradScaler()
        #try:
        for epoch in range(num_epochs):
            progress_bar = tqdm(training_generator)
            for data in progress_bar:
                imgs = data[0].permute(0, 3, 1, 2).float()
                heatmaps = data[1]

                if self.cfg.NUM_GPUS == 1:
                    imgs = imgs.cuda()
                    heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))

                self.optimizer.zero_grad()
                if self.cfg.USE_MIXED_PRECISION:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs)
                        heatmaps_losses, _, _ = self.criterion(outputs, heatmaps,[[],[]])
                        loss = 0
                        for idx in range(2):
                            if heatmaps_losses[idx] is not None:
                                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                                loss = loss + heatmaps_loss
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    outputs = self.model(imgs)
                    heatmaps_losses, _, _ = self.criterion(outputs, heatmaps,[[],[]])
                    loss = 0
                    for idx in range(2):
                        if heatmaps_losses[idx] is not None:
                            heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                            loss = loss + heatmaps_loss
                    loss.backward()
                    self.optimizer.step()

                self.lossMeter.update(loss.item())
                progress_bar.set_description(
                    'Epoch: {}/{}. Loss: {:.5f}'.format(
                        epoch, num_epochs, self.lossMeter.read()))


            self.logger.update_train_loss(self.lossMeter.read())
            self.scheduler.step(self.lossMeter.read())

            self.lossMeter.reset()

            if epoch % self.cfg.EFFICIENTTRACK.CHECKPOINT_SAVE_INTERVAL == 0 and epoch > 0:
                self.save_checkpoint(f'EfficientTrack-d{self.cfg.EFFICIENTTRACK.COMPOUND_COEF}_{epoch}_{step}.pth')
                print('checkpoint...')
            if epoch % self.cfg.EFFICIENTTRACK.VAL_INTERVAL == 0:
                self.model.eval()
                for data in val_generator:
                    with torch.no_grad():
                        imgs = data[0].permute(0, 3, 1, 2).float()
                        heatmaps = data[1]
                        keypoints = np.array(data[2]).reshape(-1,self.cfg.EFFICIENTTRACK.NUM_JOINTS,3)[:,:,:2]
                        if self.cfg.NUM_GPUS == 1:
                            imgs = imgs.cuda()
                            heatmaps = list(map(lambda x: x.cuda(non_blocking=True), heatmaps))

                        if self.cfg.USE_MIXED_PRECISION:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(imgs)
                                heatmaps_losses, _, _ = self.criterion(outputs, heatmaps,[[],[]])
                        else:
                            outputs = self.model(imgs)
                            heatmaps_losses, _, _ = self.criterion(outputs, heatmaps,[[],[]])

                        loss = 0
                        for idx in range(2):
                            if heatmaps_losses[idx] is not None:
                                heatmaps_loss = heatmaps_losses[idx].mean(dim=0)
                                loss = loss + heatmaps_loss

                    preds, maxvals = darkpose.get_final_preds(outputs[1].clamp(0,255).detach().cpu().numpy(), None)
                    masked = np.ma.masked_where(maxvals.reshape(self.cfg.EFFICIENTTRACK.BATCH_SIZE,self.cfg.EFFICIENTTRACK.NUM_JOINTS) < 10, np.linalg.norm((preds+0.5)*2-keypoints, axis = 2))

                    self.lossMeter.update(loss.item())
                    self.accuracyMeter.update(np.mean(masked))

                print(
                    'Val. Epoch: {}/{}. Loss: {:1.5f}. Acc: {:1.3f}'.format(
                        epoch, num_epochs, self.lossMeter.read(), self.accuracyMeter.read()))

                self.logger.update_val_loss(self.lossMeter.read())
                self.logger.update_val_accuracy(self.accuracyMeter.read())
                self.lossMeter.reset()
                self.accuracyMeter.reset()

                if loss + self.cfg.EFFICIENTTRACK.EARLY_STOPPING_MIN_DELTA < best_loss  and self.cfg.EFFICIENTTRACK.USE_EARLY_STOPPING:
                    best_loss = loss
                    best_epoch = epoch
                    self.save_checkpoint(f'EfficientTrack-d{self.cfg.EFFICIENTTRACK.COMPOUND_COEF}_{epoch}_{step}.pth')

                self.model.train()

                # Early stopping
                if epoch - best_epoch > self.cfg.EFFICIENTTRACK.EARLY_STOPPING_PATIENCE > 0 and self.cfg.EFFICIENTTRACK.USE_EARLY_STOPPING:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break

    def save_checkpoint(self, name):
        if isinstance(self.model, utils.CustomDataParallel):
            torch.save(self.module.model.state_dict(), os.path.join(self.model_savepath, name))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.model_savepath, name))

    def export_to_onnx(self, savefile_name, input_size = 256, export_params = True, opset_version = 9):
        input = torch.zeros(1,3,input_size,input_size).cuda()
        torch.onnx.export(self.model, input, savefile_name, input_names=['input'],
        	              output_names=['output'], export_params=True, opset_version=9, do_constant_folding = True)
        onnx_model = onnx.load(savefile_name)
        onnx.checker.check_model(onnx_model)
