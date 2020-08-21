import os
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import network
import test_dataset
import utils


def WGAN_tester(opt):
    
    # Save the model if pre_train == True
    def load_model_generator(net, epoch, opt):
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, 4)
        model_name = os.path.join('pretrained_model', model_name)
        pretrained_dict = torch.load(model_name)
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Build networks
    generator = utils.create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    load_model_generator(generator, opt.epoch, opt)
    print('-------------------------Pretrained Model Loaded-------------------------')

    # To device
    generator = generator.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = test_dataset.InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers, pin_memory = True)
    
    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    for batch_idx, (img, mask) in enumerate(dataloader):
        img = img.cuda()
        mask = mask.cuda()

        # Generator output
        with torch.no_grad():
            first_out, second_out = generator(img, mask)

        # forward propagation
        first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
        second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

        masked_img = img * (1 - mask) + mask
        mask = torch.cat((mask, mask, mask), 1)
        img_list = [second_out_wholeimg]
        name_list = ['second_out']
        utils.save_sample_png(sample_folder = results_path, sample_name = '%d' % (batch_idx + 1), img_list = img_list, name_list = name_list, pixel_max_cnt = 255)
        print('----------------------batch_idx%d' % (batch_idx + 1) + ' has been finished----------------------')
