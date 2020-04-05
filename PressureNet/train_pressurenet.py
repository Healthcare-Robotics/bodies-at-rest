#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

import chumpy as ch

import convnet_br as convnet
# import tf.transformations as tft

# import hrl_lib.util as ut
import cPickle as pickle


# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


import sys
sys.path.insert(0, '../lib_py')

# Pose Estimation Libraries
from visualization_lib_br import VisualizationLib
from preprocessing_lib_br import PreprocessingLib
from tensorprep_lib_br import TensorPrepLib
from unpack_batch_lib_br import UnpackBatchLib

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom

np.set_printoptions(threshold=sys.maxsize)

MAT_WIDTH = 0.762  # metres
MAT_HEIGHT = 1.854  # metres
MAT_HALF_WIDTH = MAT_WIDTH / 2
NUMOFTAXELS_X = 64  # 73 #taxels
NUMOFTAXELS_Y = 27  # 30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TRAIN = 24
NUMOFOUTPUTNODES_TEST = 10
INTER_SENSOR_DISTANCE = 0.0286  # metres

torch.set_num_threads(1)
if torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    print '######################### CUDA is available! #############################'
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    print '############################## USING CPU #################################'


class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''

    def __init__(self, training_database_file_f, training_database_file_m, testing_database_file_f,
                 testing_database_file_m, opt):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''


        self.CTRL_PNL = {}
        self.CTRL_PNL['loss_vector_type'] = opt.losstype
        self.CTRL_PNL['verbose'] = opt.verbose
        self.opt = opt
        self.CTRL_PNL['batch_size'] = 128
        self.CTRL_PNL['num_epochs'] = 100
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = True
        self.CTRL_PNL['incl_ht_wt_channels'] = opt.htwt
        self.CTRL_PNL['loss_root'] = opt.loss_root
        self.CTRL_PNL['omit_cntct_sobel'] = opt.omit_cntct_sobel
        self.CTRL_PNL['use_hover'] = opt.use_hover
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 2
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['regr_angles'] = opt.reg_angles
        self.CTRL_PNL['depth_map_labels'] = opt.pmr #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['depth_map_labels_test'] = opt.pmr #False #can only be true is we have 100% synth for testing
        self.CTRL_PNL['depth_map_output'] = self.CTRL_PNL['depth_map_labels']
        self.CTRL_PNL['depth_map_input_est'] = opt.pmr #do this if we're working in a two-part regression
        if opt.mod == 1:
            self.CTRL_PNL['adjust_ang_from_est'] = False #starts angles from scratch
        elif opt.mod == 2:
            self.CTRL_PNL['adjust_ang_from_est'] = True #gets betas and angles from prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True
        self.CTRL_PNL['normalize_per_image'] = True
        if self.CTRL_PNL['normalize_per_image'] == False:
            self.CTRL_PNL['normalize_std'] = True
        else:
            self.CTRL_PNL['normalize_std'] = False
        self.CTRL_PNL['all_tanh_activ'] = True
        self.CTRL_PNL['pmat_mult'] = int(1)
        self.CTRL_PNL['cal_noise'] = opt.calnoise
        self.CTRL_PNL['cal_noise_amt'] = 0.2
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['first_pass'] = True

        if GPU == True:
            torch.cuda.set_device(self.opt.device)

        self.weight_joints = 1.0#self.opt.j_d_ratio*2
        self.weight_depth_planes = (1-self.opt.j_d_ratio)#*2


        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['incl_pmat_cntct_input'] = False #if there's calibration noise we need to recompute this every batch
            self.CTRL_PNL['clip_sobel'] = False

        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['depth_map_input_est'] == True: #for a two part regression
            self.CTRL_PNL['num_input_channels'] += 3
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.CTRL_PNL['num_input_channels'] += 2
        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['num_input_channels'] += 1

        pmat_std_from_mult = ['N/A', 11.70153502792190, 19.90905848383454, 23.07018866032369, 0.0, 25.50538629767412]
        if self.CTRL_PNL['cal_noise'] == False:
            sobel_std_from_mult = ['N/A', 29.80360490415032, 33.33532963163579, 34.14427844692501, 0.0, 34.86393494050921]
        else:
            sobel_std_from_mult = ['N/A', 45.61635847182483, 77.74920396659292, 88.89398421073700, 0.0, 97.90075708182506]

        self.CTRL_PNL['norm_std_coeffs'] =  [1./41.80684362163343,  #contact
                                             1./16.69545796387731,  #pos est depth
                                             1./45.08513083167194,  #neg est depth
                                             1./43.55800622930469,  #cm est
                                             1./pmat_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat x5
                                             1./sobel_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat sobel
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1. / 30.216647403350,  #weight
                                             1. / 14.629298141231]  #height

        if self.CTRL_PNL['normalize_std'] == False:
            for i in range(10):
                self.CTRL_PNL['norm_std_coeffs'][i] *= 0.
                self.CTRL_PNL['norm_std_coeffs'][i] += 1.


        self.CTRL_PNL['convnet_fp_prefix'] = '../data_BR/convnets/'

        if self.CTRL_PNL['depth_map_output'] == True: #we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]



        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



        #################################### PREP TRAINING DATA ##########################################
        #load training ysnth data
        if opt.small == True and opt.mod == 1:
            reduce_data = True
        else:
            reduce_data = False

        dat_f_synth = TensorPrepLib().load_files_to_database(training_database_file_f, creation_type = 'synth', reduce_data = reduce_data)
        dat_m_synth = TensorPrepLib().load_files_to_database(training_database_file_m, creation_type = 'synth', reduce_data = reduce_data)


        self.train_x_flat = []  # Initialize the testing pressure mat list
        self.train_x_flat = TensorPrepLib().prep_images(self.train_x_flat, dat_f_synth, dat_m_synth, num_repeats = 1)
        self.train_x_flat = list(np.clip(np.array(self.train_x_flat) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100))

        if self.CTRL_PNL['cal_noise'] == False:
            self.train_x_flat = PreprocessingLib().preprocessing_blur_images(self.train_x_flat, self.mat_size, sigma=0.5)

        if len(self.train_x_flat) == 0: print("NO TRAINING DATA INCLUDED")

        if self.CTRL_PNL['depth_map_labels'] == True:
            self.depth_contact_maps = [] #Initialize the precomputed depth and contact maps. only synth has this label.
            self.depth_contact_maps = TensorPrepLib().prep_depth_contact(self.depth_contact_maps, dat_f_synth, dat_m_synth, num_repeats = 1)
        else:
            self.depth_contact_maps = None

        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.depth_contact_maps_input_est = [] #Initialize the precomputed depth and contact map input estimates
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(self.depth_contact_maps_input_est,
                                                                                             dat_f_synth, dat_m_synth, num_repeats = 1)
        else:
            self.depth_contact_maps_input_est = None

        #stack the bed height array on the pressure image as well as a sobel filtered image
        train_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.train_x_flat,
                                                                                self.mat_size,
                                                                                self.CTRL_PNL)

        #stack the depth and contact mesh images (and possibly a pmat contact image) together
        train_xa = TensorPrepLib().append_input_depth_contact(np.array(train_xa),
                                                              CTRL_PNL = self.CTRL_PNL,
                                                              mesh_depth_contact_maps_input_est = self.depth_contact_maps_input_est,
                                                              mesh_depth_contact_maps = self.depth_contact_maps)

        #normalize the input
        if self.CTRL_PNL['normalize_std'] == True:
            train_xa = TensorPrepLib().normalize_network_input(train_xa, self.CTRL_PNL)

        self.train_x_tensor = torch.Tensor(train_xa)

        train_y_flat = []  # Initialize the training ground truth list
        train_y_flat = TensorPrepLib().prep_labels(train_y_flat, dat_f_synth, num_repeats = 1,
                                                        z_adj = -0.075, gender = "f", is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                        full_body_rot = self.CTRL_PNL['full_body_rot'])
        train_y_flat = TensorPrepLib().prep_labels(train_y_flat, dat_m_synth, num_repeats = 1,
                                                        z_adj = -0.075, gender = "m", is_synth = True,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                        full_body_rot = self.CTRL_PNL['full_body_rot'])


        # normalize the height and weight
        if self.CTRL_PNL['normalize_std'] == True:
            train_y_flat = TensorPrepLib().normalize_wt_ht(train_y_flat, self.CTRL_PNL)

        self.train_y_tensor = torch.Tensor(train_y_flat)

        print self.train_x_tensor.shape, 'Input training tensor shape'
        print self.train_y_tensor.shape, 'Output training tensor shape'




        #################################### PREP TESTING DATA ##########################################
        # load in the test file
        test_dat_f_synth = TensorPrepLib().load_files_to_database(testing_database_file_f, creation_type = 'synth', reduce_data = reduce_data)
        test_dat_m_synth = TensorPrepLib().load_files_to_database(testing_database_file_m, creation_type = 'synth', reduce_data = reduce_data)
        test_dat_f_real = TensorPrepLib().load_files_to_database(testing_database_file_f, creation_type = 'real', reduce_data = reduce_data)
        test_dat_m_real = TensorPrepLib().load_files_to_database(testing_database_file_m, creation_type = 'real', reduce_data = reduce_data)

        self.test_x_flat = []  # Initialize the testing pressure mat list
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f_synth, test_dat_m_synth, num_repeats = 1)
        self.test_x_flat = list(np.clip(np.array(self.test_x_flat) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100))
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f_real, test_dat_m_real, num_repeats = 1)

        if self.CTRL_PNL['cal_noise'] == False:
            self.test_x_flat = PreprocessingLib().preprocessing_blur_images(self.test_x_flat, self.mat_size, sigma=0.5)

        if len(self.test_x_flat) == 0: print("NO TESTING DATA INCLUDED")

        if self.CTRL_PNL['depth_map_labels_test'] == True:
            self.depth_contact_maps = [] #Initialize the precomputed depth and contact maps. only synth has this label.
            self.depth_contact_maps = TensorPrepLib().prep_depth_contact(self.depth_contact_maps, test_dat_f_synth, test_dat_m_synth, num_repeats = 1)
        else:
            self.depth_contact_maps = None

        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.depth_contact_maps_input_est = [] #Initialize the precomputed depth and contact map input estimates
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(self.depth_contact_maps_input_est,
                                                                                             test_dat_f_synth, test_dat_m_synth, num_repeats = 1)
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(self.depth_contact_maps_input_est,
                                                                                             test_dat_f_real, test_dat_m_real, num_repeats = 1)
        else:
            self.depth_contact_maps_input_est = None

        print np.shape(self.test_x_flat)

        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat,
                                                                                self.mat_size,
                                                                                self.CTRL_PNL)


        test_xa = TensorPrepLib().append_input_depth_contact(np.array(test_xa),
                                                              CTRL_PNL = self.CTRL_PNL,
                                                              mesh_depth_contact_maps_input_est = self.depth_contact_maps_input_est,
                                                              mesh_depth_contact_maps = self.depth_contact_maps)

        #normalize the input
        if self.CTRL_PNL['normalize_std'] == True:
            test_xa = TensorPrepLib().normalize_network_input(test_xa, self.CTRL_PNL)

        self.test_x_tensor = torch.Tensor(test_xa)

        test_y_flat = []  # Initialize the ground truth listhave

        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_f_synth, num_repeats = 1,
                                                    z_adj = -0.075, gender = "f", is_synth = True,
                                                    loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                    initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                    full_body_rot = self.CTRL_PNL['full_body_rot'])
        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_m_synth, num_repeats = 1,
                                                    z_adj = -0.075, gender = "m", is_synth = True,
                                                    loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                    initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                    full_body_rot = self.CTRL_PNL['full_body_rot'])

        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_f_real, num_repeats = 1,
                                                    z_adj = 0.0, gender = "f", is_synth = False,
                                                    loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                    initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])
        test_y_flat = TensorPrepLib().prep_labels(test_y_flat, test_dat_m_real, num_repeats = 1,
                                                    z_adj = 0.0, gender = "m", is_synth = False,
                                                    loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                    initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'])

        if self.CTRL_PNL['normalize_std'] == True:
            test_y_flat = TensorPrepLib().normalize_wt_ht(test_y_flat, self.CTRL_PNL)

        self.test_y_tensor = torch.Tensor(test_y_flat)


        print self.test_x_tensor.shape, 'Input testing tensor shape'
        print self.test_y_tensor.shape, 'Output testing tensor shape'




        self.save_name = '_' + str(opt.mod) + '_' + opt.losstype + \
                         '_' + str(self.train_x_tensor.size()[0]) + 'ct' + \
                         '_' + str(self.CTRL_PNL['batch_size']) + 'b' + \
                         '_x' + str(self.CTRL_PNL['pmat_mult']) + 'pm'


        if self.CTRL_PNL['depth_map_labels'] == True:
            self.save_name += '_' + str(self.opt.j_d_ratio) + 'rtojtdpth'
        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.save_name += '_depthestin'
        if self.CTRL_PNL['adjust_ang_from_est'] == True:
            self.save_name += '_angleadj'
        if self.CTRL_PNL['all_tanh_activ'] == True:
            self.save_name += '_tnh'
        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.save_name += '_htwt'
        if self.CTRL_PNL['cal_noise'] == True:
            self.save_name += '_clns'+str(int(self.CTRL_PNL['cal_noise_amt']*100)) + 'p'
        if self.CTRL_PNL['double_network_size'] == True:
            self.save_name += '_dns'

        if  self.CTRL_PNL['loss_root'] == True:
            self.save_name += '_rt'
        if  self.CTRL_PNL['omit_cntct_sobel'] == True:
            self.save_name += '_ocs'
        if  self.CTRL_PNL['use_hover'] == True:
            self.save_name += '_uh'
        if  self.opt.half_shape_wt == True:
            self.save_name += '_hsw'

        print 'appending to', 'train' + self.save_name
        self.train_val_losses = {}
        self.train_val_losses['train_loss'] = []
        self.train_val_losses['val_loss'] = []
        self.train_val_losses['epoch_ct'] = []





    def init_convnet_train(self):

        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])

        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])



        print "Loading convnet model................................"

        fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations

        if self.CTRL_PNL['full_body_rot'] == True:
            fc_output_size += 3

        if self.opt.go200 == True:
            self.model = torch.load(self.CTRL_PNL['convnet_fp_prefix'] + 'convnet_1_anglesDC_184000ct_128b_x1pm_tnh_clns20p_100e_2e-05lr.pt',map_location={'cuda:' + str(self.opt.prev_device): 'cuda:' + str(self.opt.device)})

        elif self.opt.omit_cntct_sobel == True:
            self.model = convnet.CNN(fc_output_size, self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'],
                                     verts_list = self.verts_list, in_channels=self.CTRL_PNL['num_input_channels']-2)
        else:
            self.model = convnet.CNN(fc_output_size, self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'],
                                     verts_list = self.verts_list, in_channels=self.CTRL_PNL['num_input_channels'])

        #self.model = torch.load('../data_BR/convnets_camready/convnet_2_anglesDC_184000ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_tnh_clns20p_rt_50e_2e-05lr.pt',
        #                        map_location={'cuda:0': 'cuda:' + str(self.opt.device)})

        #load in a model instead if one is partially trained

        #self.model = torch.load(self.CTRL_PNL['convnet_fp_prefix']+'convnet_anglesDC_synth_184K_128b_x5pmult_1.0rtojtdpth_tnh_htwt_calnoise_100e_00002lr.pt', map_location={'cuda:2':'cuda:'+str(DEVICE)})
        #self.model = torch.load(self.CTRL_PNL['convnet_fp_prefix']+'convnet_2_anglesDC_184000ct_128b_x1pm_0.5rtojtdpth_depthestin_angleadj_tnh_clns10p_100e_2e-05lr.pt', map_location='cpu')

        pp = 0
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print 'LOADED. num params: ', pp


        # Run model on GPU if available
        if GPU == True:
            self.model = self.model.cuda()

        learning_rate = 0.00002

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.0005) #start with .00005

        # train the model one epoch at a time
        for epoch in range(1, self.CTRL_PNL['num_epochs'] + 1):
            #torch.save(self.model, self.CTRL_PNL['convnet_fp_prefix']+'convnet'+self.save_name+'_'+str(epoch)+'e'+'_'+str(learning_rate)+'lr.pt')

            self.t1 = time.time()
            self.train_convnet(epoch)

            try:
                self.t2 = time.time() - self.t1
            except:
                self.t2 = 0
            print 'Time taken by epoch',epoch,':',self.t2,' seconds'

            if epoch == self.CTRL_PNL['num_epochs'] or epoch == 10 or epoch == 20 or epoch == 30 or epoch == 40 or epoch == 50 or epoch == 60 or epoch == 70 or epoch == 80 or epoch == 90:

                if self.opt.go200 == True:
                    epoch_log = epoch + 100
                else:
                    epoch_log = epoch + 0

                print "saving convnet."
                torch.save(self.model, self.CTRL_PNL['convnet_fp_prefix']+'convnet'+self.save_name+'_'+str(epoch_log)+'e'+'_'+str(learning_rate)+'lr.pt')
                print "saved convnet."
                pkl.dump(self.train_val_losses,open(self.CTRL_PNL['convnet_fp_prefix']+'convnet_losses'+self.save_name+'_'+str(epoch_log)+'e'+'_'+str(learning_rate)+'lr.p', 'wb'))
                print "saved losses."

        print self.train_val_losses, 'trainval'
        # Save the model (architecture and weights)




    def train_convnet(self, epoch):
        '''
        Train the model for one epoch.
        '''
        # Some models use different forward passes between train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.
        self.model.train()
        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()
        with torch.autograd.set_detect_anomaly(True):

            # This will loop a total = training_images/batch_size times
            for batch_idx, batch in enumerate(self.train_loader):


                self.optimizer.zero_grad()
                scores, INPUT_DICT, OUTPUT_DICT = \
                    UnpackBatchLib().unpack_batch(batch, is_training=True, model = self.model, CTRL_PNL=self.CTRL_PNL)
                #print torch.cuda.max_memory_allocated(), '1post train'
                self.CTRL_PNL['first_pass'] = False

                scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype),
                                        requires_grad=True)

                if self.CTRL_PNL['full_body_rot'] == True:
                    OSA = 6
                    if self.opt.loss_root == True:
                        loss_bodyrot = self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * self.weight_joints
                    else:
                        loss_bodyrot = self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * 0.0
                    #if self.CTRL_PNL['adjust_ang_from_est'] == True:
                    #    loss_bodyrot *= 0
                else: OSA = 0

                loss_eucl = self.criterion(scores[:, 10+OSA:34+OSA], scores_zeros[:, 10+OSA:34+OSA])*self.weight_joints
                if self.opt.half_shape_wt == True:
                    loss_betas = self.criterion(scores[:, 0:10], scores_zeros[:, 0:10]) * self.weight_joints * 0.5
                else:
                    loss_betas = self.criterion(scores[:, 0:10], scores_zeros[:, 0:10]) * self.weight_joints


                if self.CTRL_PNL['regr_angles'] == True:
                    loss_angs = self.criterion2(scores[:, 34+OSA:106+OSA], scores_zeros[:, 34+OSA:106+OSA])*self.weight_joints
                    loss = (loss_betas + loss_eucl + loss_bodyrot + loss_angs)
                else:
                    loss = (loss_betas + loss_eucl + loss_bodyrot)


                #print INPUT_DICT['batch_mdm'].size(), OUTPUT_DICT['batch_mdm_est'].size()
                if self.CTRL_PNL['depth_map_labels'] == True:
                    hover_map = OUTPUT_DICT['batch_mdm_est'].clone()
                    hover_map[hover_map < 0] = 0

                    INPUT_DICT['batch_mdm'][INPUT_DICT['batch_mdm'] > 0] = 0
                    if self.CTRL_PNL['mesh_bottom_dist'] == True:
                        OUTPUT_DICT['batch_mdm_est'][OUTPUT_DICT['batch_mdm_est'] > 0] = 0

                    loss_mesh_depth = self.criterion2(INPUT_DICT['batch_mdm'], OUTPUT_DICT['batch_mdm_est'])*self.weight_depth_planes * (1. / 44.46155340000357) * (1. / 44.46155340000357)
                    loss_mesh_contact = self.criterion(INPUT_DICT['batch_cm'], OUTPUT_DICT['batch_cm_est'])*self.weight_depth_planes * (1. / 0.4428100696329912)

                    loss += loss_mesh_depth
                    loss += loss_mesh_contact



                loss.backward()
                self.optimizer.step()
                loss *= 1000


                if batch_idx % opt.log_interval == 0:# and batch_idx > 0:

                    if GPU == True:
                        print "GPU memory:", torch.cuda.max_memory_allocated()

                    val_n_batches = 4
                    print "evaluating on ", val_n_batches

                    im_display_idx = 0 #random.randint(0,INPUT_DICT['batch_images'].size()[0])


                    if GPU == True:
                        VisualizationLib().print_error_train(INPUT_DICT['batch_targets'].cpu(), OUTPUT_DICT['batch_targets_est'].cpu(),
                                                             self.output_size_train, self.CTRL_PNL['loss_vector_type'],
                                                             data='train')
                    else:
                        VisualizationLib().print_error_train(INPUT_DICT['batch_targets'], OUTPUT_DICT['batch_targets_est'],
                                                             self.output_size_train, self.CTRL_PNL['loss_vector_type'],
                                                             data='train')

                   # print INPUT_DICT['batch_images'][im_display_idx, 4:, :].type()

                    if self.CTRL_PNL['depth_map_labels'] == True: #pmr regression
                        self.cntct_in = INPUT_DICT['batch_images'][im_display_idx, 0, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][0]  #contact
                        self.pimage_in = INPUT_DICT['batch_images'][im_display_idx, 1, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][4] #pmat
                        self.sobel_in = INPUT_DICT['batch_images'][im_display_idx, 2, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][5]  #sobel
                        self.pmap_recon = (OUTPUT_DICT['batch_mdm_est'][im_display_idx, :, :].squeeze()*-1).cpu().data #est depth output
                        self.cntct_recon = (OUTPUT_DICT['batch_cm_est'][im_display_idx, :, :].squeeze()).cpu().data #est depth output
                        self.hover_recon = (hover_map[im_display_idx, :, :].squeeze()).cpu().data #est depth output
                        self.pmap_recon_gt = (INPUT_DICT['batch_mdm'][im_display_idx, :, :].squeeze()*-1).cpu().data #ground truth depth
                        self.cntct_recon_gt = (INPUT_DICT['batch_cm'][im_display_idx, :, :].squeeze()).cpu().data #ground truth depth
                    else:
                        self.cntct_in = INPUT_DICT['batch_images'][im_display_idx, 0, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][0]  #contact
                        self.pimage_in = INPUT_DICT['batch_images'][im_display_idx, 1, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][4]  #pmat
                        self.sobel_in = INPUT_DICT['batch_images'][im_display_idx, 2, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][5]  #sobel
                        self.pmap_recon = None
                        self.cntct_recon = None
                        self.hover_recon = None
                        self.pmap_recon_gt = None
                        self.cntct_recon_gt = None

                    if self.CTRL_PNL['depth_map_input_est'] == True: #this is a network 2 option ONLY
                        self.pmap_recon_in = INPUT_DICT['batch_images'][im_display_idx, 2, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][2] #pmat
                        self.cntct_recon_in = INPUT_DICT['batch_images'][im_display_idx, 3, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][3] #pmat
                        self.hover_recon_in = INPUT_DICT['batch_images'][im_display_idx, 1, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][1] #pmat
                        self.pimage_in = INPUT_DICT['batch_images'][im_display_idx, 4, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][4] #pmat
                        self.sobel_in = INPUT_DICT['batch_images'][im_display_idx, 5, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][5]  #sobel
                    else:
                        self.pmap_recon_in = None
                        self.cntct_recon_in = None
                        self.hover_recon_in = None




                    self.tar_sample = INPUT_DICT['batch_targets']
                    self.tar_sample = self.tar_sample[im_display_idx, :].squeeze() / 1000
                    self.sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
                    self.sc_sample = self.sc_sample[im_display_idx, :].squeeze() / 1000
                    self.sc_sample = self.sc_sample.view(self.output_size_train)

                    train_loss = loss.data.item()
                    examples_this_epoch = batch_idx * len(INPUT_DICT['batch_images'])
                    epoch_progress = 100. * batch_idx / len(self.train_loader)

                    val_loss = self.validate_convnet(n_batches=val_n_batches)


                    print_text_list = [ 'Train Epoch: {} ',
                                        '[{}',
                                        '/{} ',
                                        '({:.0f}%)]\t']
                    print_vals_list = [epoch,
                                      examples_this_epoch,
                                      len(self.train_loader.dataset),
                                      epoch_progress]
                    if self.CTRL_PNL['loss_vector_type'] == 'anglesR' or self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':
                        print_text_list.append('Train Loss Joints: {:.2f}')
                        print_vals_list.append(1000*loss_eucl.data)
                        print_text_list.append('\n\t\t\t\t\t\t   Betas Loss: {:.2f}')
                        print_vals_list.append(1000*loss_betas.data)
                        if self.CTRL_PNL['full_body_rot'] == True:
                            print_text_list.append('\n\t\t\t\t\t\tBody Rot Loss: {:.2f}')
                            print_vals_list.append(1000*loss_bodyrot.data)
                        if self.CTRL_PNL['regr_angles'] == True:
                            print_text_list.append('\n\t\t\t\t\t\t  Angles Loss: {:.2f}')
                            print_vals_list.append(1000*loss_angs.data)
                        if self.CTRL_PNL['depth_map_labels'] == True:
                            print_text_list.append('\n\t\t\t\t\t\t   Mesh Depth: {:.2f}')
                            print_vals_list.append(1000*loss_mesh_depth.data)
                            print_text_list.append('\n\t\t\t\t\t\t Mesh Contact: {:.2f}')
                            print_vals_list.append(1000*loss_mesh_contact.data)

                    print_text_list.append('\n\t\t\t\t\t\t   Total Loss: {:.2f}')
                    print_vals_list.append(train_loss)

                    print_text_list.append('\n\t\t\t\t\t  Val Total Loss: {:.2f}')
                    print_vals_list.append(val_loss)



                    print_text = ''
                    for item in print_text_list:
                        print_text += item
                    print(print_text.format(*print_vals_list))


                    print 'appending to alldata losses'
                    self.train_val_losses['train_loss'].append(train_loss)
                    self.train_val_losses['epoch_ct'].append(epoch)
                    self.train_val_losses['val_loss'].append(val_loss)


    def validate_convnet(self, verbose=False, n_batches=None):
        self.model.eval()
        loss = 0.
        n_examples = 0
        batch_ct = 1

        if True:
            for batch_i, batch in enumerate(self.test_loader):

                scores, INPUT_DICT_VAL, OUTPUT_DICT_VAL = \
                    UnpackBatchLib().unpack_batch(batch, is_training=False, model=self.model, CTRL_PNL=self.CTRL_PNL)
                scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype),
                                        requires_grad=False)

                loss_to_add = 0

                if self.CTRL_PNL['full_body_rot'] == True:
                    OSA = 6
                    if self.opt.loss_root == True:
                        loss_bodyrot = float(self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * self.weight_joints)
                    else:
                        loss_bodyrot = float(self.criterion(scores[:, 10:16], scores_zeros[:, 10:16]) * 0.0)
                    loss_to_add += loss_bodyrot
                else: OSA = 0

                loss_eucl = float(self.criterion(scores[:, 10+OSA:34+OSA], scores_zeros[:,  10+OSA:34+OSA]) * self.weight_joints)
                if self.opt.half_shape_wt == True:
                    loss_betas = float(self.criterion(scores[:, 0:10], scores_zeros[:, 0:10]) * self.weight_joints * 0.5)
                else:
                    loss_betas = float(self.criterion(scores[:, 0:10], scores_zeros[:, 0:10]) * self.weight_joints)



                if self.CTRL_PNL['regr_angles'] == True:
                    loss_angs = float(self.criterion(scores[:, 34+OSA:106+OSA], scores_zeros[:, 34+OSA:106+OSA]) * self.weight_joints)
                    loss_to_add += (loss_betas + loss_eucl + loss_angs)
                else:
                    loss_to_add += (loss_betas + loss_eucl)

                # print INPUT_DICT_VAL['batch_mdm'].size(), OUTPUT_DICT_VAL['batch_mdm_est'].size()

                if self.CTRL_PNL['depth_map_labels'] == True:
                    INPUT_DICT_VAL['batch_mdm'][INPUT_DICT_VAL['batch_mdm'] > 0] = 0
                    if self.CTRL_PNL['mesh_bottom_dist'] == True:
                        OUTPUT_DICT_VAL['batch_mdm_est'][OUTPUT_DICT_VAL['batch_mdm_est'] > 0] = 0

                    loss_mesh_depth = float(self.criterion2(INPUT_DICT_VAL['batch_mdm'],OUTPUT_DICT_VAL['batch_mdm_est']) * self.weight_depth_planes * (1. / 44.46155340000357) * (1. / 44.46155340000357))
                    loss_mesh_contact = float(self.criterion(INPUT_DICT_VAL['batch_cm'],OUTPUT_DICT_VAL['batch_cm_est']) * self.weight_depth_planes * (1. / 0.4428100696329912))

                    loss_to_add += loss_mesh_depth
                    loss_to_add += loss_mesh_contact

                loss += loss_to_add

                #print loss
                n_examples += self.CTRL_PNL['batch_size']

                if n_batches and (batch_i >= n_batches):
                    break

                batch_ct += 1
                #break


            loss /= batch_ct
            loss *= 1000

        if self.opt.visualize == True:
            VisualizationLib().visualize_pressure_map(pimage_in = self.pimage_in, cntct_in = self.cntct_in, sobel_in = self.sobel_in,
                                                      targets_raw = self.tar_sample.cpu(), scores_net1 = self.sc_sample.cpu(),
                                                      pmap_recon_in = self.pmap_recon_in, cntct_recon_in = self.cntct_recon_in,
                                                      hover_recon_in = self.hover_recon_in,
                                                      pmap_recon = self.pmap_recon, cntct_recon = self.cntct_recon, hover_recon = self.hover_recon,
                                                      pmap_recon_gt=self.pmap_recon_gt, cntct_recon_gt = self.cntct_recon_gt,
                                                      block=False)

        #print "loss is:" , loss
        return loss






if __name__ == "__main__":

    import optparse
    p = optparse.OptionParser()

    p.add_option('--hd', action='store_true', dest='hd', default=False,
                 help='Read and write to data on an external harddrive.')

    p.add_option('--losstype', action='store', type = 'string', dest='losstype', default='anglesDC',
                 help='Choose direction cosine or euler angle regression.')

    p.add_option('--j_d_ratio', action='store', type = 'float', dest='j_d_ratio', default=0.5, #PMR parameter to adjust loss function 2
                 help='Set the loss mix: joints to depth planes. Only used for PMR regression.')

    p.add_option('--mod', action='store', type = 'int', dest='mod', default=0,
                 help='Choose a network.')

    p.add_option('--prev_device', action='store', type = 'int', dest='prev_device', default=0,
                 help='Choose a GPU core that it was previously on.')

    p.add_option('--device', action='store', type = 'int', dest='device', default=0,
                 help='Choose a GPU core.')

    p.add_option('--qt', action='store_true', dest='quick_test', default=False,
                 help='Do a quick test.')

    p.add_option('--pmr', action='store_true', dest='pmr', default=False,
                 help='Run PMR on input plus precomputed spatial maps.')

    p.add_option('--go200', action='store_true', dest='go200', default=False,
                 help='Run network 1 for 100 to 200 epochs.')

    p.add_option('--small', action='store_true', dest='small', default=False,
                 help='Make the dataset 1/4th of the original size.')

    p.add_option('--htwt', action='store_true', dest='htwt', default=False,
                 help='Include height and weight info on the input.')

    p.add_option('--omit_cntct_sobel', action='store_true', dest='omit_cntct_sobel', default=False,
                 help='Cut contact and sobel from input.')

    p.add_option('--use_hover', action='store_true', dest='use_hover', default=False,
                 help='Cut hovermap from pmr input.')

    p.add_option('--calnoise', action='store_true', dest='calnoise', default=False,
                 help='Apply calibration noise to the input to facilitate sim to real transfer.')




    p.add_option('--half_shape_wt', action='store_true', dest='half_shape_wt', default=False,
                 help='Half betas.')

    p.add_option('--viz', action='store_true', dest='visualize', default=False,
                 help='Visualize training.')

    p.add_option('--loss_root', action='store_true', dest='loss_root', default=False,
                 help='Use root in loss function.')

    p.add_option('--rgangs', action='store_true', dest='reg_angles', default=False, #I found this option doesn't help much.
                 help='Regress the angles as well as betas and joint pos.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=True, help='Printout everything (under construction).')

    p.add_option('--log_interval', type=int, default=100, metavar='N',
                 help='number of batches between logging train status') #if you visualize too often it will slow down training.

    opt, args = p.parse_args()

    if opt.hd == False:
        data_fp_prefix = "../data_BR/"
    else:
        data_fp_prefix = "/media/henry/multimodal_data_2/data_BR/"

    data_fp_suffix = ''

    if opt.mod == 2 or opt.quick_test == True:
        data_fp_suffix = '_convnet_1_'+str(opt.losstype)

        if opt.small == True:
            data_fp_suffix += '_46000ct'
        else:
            data_fp_suffix += '_184000ct'

        data_fp_suffix += '_128b_x1pm_tnh'

        if opt.htwt == True:
            data_fp_suffix += '_htwt'
        if opt.calnoise == True:
            data_fp_suffix += '_clns20p'
        if opt.loss_root == True:
            data_fp_suffix += '_rt'
        if opt.omit_cntct_sobel == True:
            data_fp_suffix += '_ocs'
        if opt.half_shape_wt == True:
            data_fp_suffix += '_hsw'

        data_fp_suffix += '_100e_'+str(0.00002)+'lr'

    elif opt.mod == 1:
        data_fp_suffix = ''

    else:
        print "Please choose a valid network. You can specify '--net 1' or '--net 2'."
        sys.exit()

    training_database_file_f = []
    training_database_file_m = []
    test_database_file_f = []
    test_database_file_m = [] #141 total training loss at epoch 9



    if opt.quick_test == True:
        #run a quick test
        training_database_file_f.append(data_fp_prefix+'synth/quick_test/test_rollpi_f_lay_set23to24_3000_qt'+data_fp_suffix+'.p')
        test_database_file_f.append(data_fp_prefix+'synth/quick_test/test_rollpi_f_lay_set23to24_3000_qt'+data_fp_suffix+'.p')

    else:
        #General partition - 104,000 train + 12,000 test
        training_database_file_f.append(data_fp_prefix + 'synth/general/train_rollpi_f_lay_set18to22_10000' + data_fp_suffix + '.p')
        training_database_file_f.append(data_fp_prefix + 'synth/general/train_rollpi_plo_f_lay_set18to22_10000' + data_fp_suffix + '.p')
        training_database_file_m.append(data_fp_prefix + 'synth/general/train_rollpi_m_lay_set18to22_10000' + data_fp_suffix + '.p')
        training_database_file_m.append(data_fp_prefix + 'synth/general/train_rollpi_plo_m_lay_set18to22_10000' + data_fp_suffix + '.p')
        training_database_file_f.append(data_fp_prefix + 'synth/general/train_rollpi_f_lay_set10to17_16000' + data_fp_suffix + '.p')
        training_database_file_f.append(data_fp_prefix + 'synth/general/train_rollpi_plo_f_lay_set10to17_16000' + data_fp_suffix + '.p')
        training_database_file_m.append(data_fp_prefix + 'synth/general/train_rollpi_m_lay_set10to17_16000' + data_fp_suffix + '.p')
        training_database_file_m.append(data_fp_prefix + 'synth/general/train_rollpi_plo_m_lay_set10to17_16000' + data_fp_suffix + '.p')

        test_database_file_f.append(data_fp_prefix+'synth/general/test_rollpi_f_lay_set23to24_3000'+data_fp_suffix+'.p')
        test_database_file_f.append(data_fp_prefix+'synth/general/test_rollpi_plo_f_lay_set23to24_3000'+data_fp_suffix+'.p')
        test_database_file_m.append(data_fp_prefix+'synth/general/test_rollpi_m_lay_set23to24_3000'+data_fp_suffix+'.p')
        test_database_file_m.append(data_fp_prefix+'synth/general/test_rollpi_plo_m_lay_set23to24_3000'+data_fp_suffix+'.p')



        #General supine partition - 52,000 train + 6,000 test
        training_database_file_f.append(data_fp_prefix + 'synth/general_supine/train_roll0_f_lay_set5to7_5000' + data_fp_suffix + '.p')
        training_database_file_f.append(data_fp_prefix + 'synth/general_supine/train_roll0_plo_f_lay_set5to7_5000' + data_fp_suffix + '.p')
        training_database_file_m.append(data_fp_prefix + 'synth/general_supine/train_roll0_m_lay_set5to7_5000' + data_fp_suffix + '.p')
        training_database_file_m.append(data_fp_prefix + 'synth/general_supine/train_roll0_plo_m_lay_set5to7_5000' + data_fp_suffix + '.p')
        training_database_file_f.append(data_fp_prefix + 'synth/general_supine/train_roll0_f_lay_set10to13_8000' + data_fp_suffix + '.p')
        training_database_file_f.append(data_fp_prefix + 'synth/general_supine/train_roll0_plo_f_lay_set10to13_8000' + data_fp_suffix + '.p')
        training_database_file_m.append(data_fp_prefix + 'synth/general_supine/train_roll0_m_lay_set10to13_8000' + data_fp_suffix + '.p')
        training_database_file_m.append(data_fp_prefix + 'synth/general_supine/train_roll0_plo_m_lay_set10to13_8000' + data_fp_suffix + '.p')

        test_database_file_f.append(data_fp_prefix+'synth/general_supine/test_roll0_f_lay_set14_1500'+data_fp_suffix+'.p')
        test_database_file_f.append(data_fp_prefix+'synth/general_supine/test_roll0_plo_f_lay_set14_1500'+data_fp_suffix+'.p')
        test_database_file_m.append(data_fp_prefix+'synth/general_supine/test_roll0_m_lay_set14_1500'+data_fp_suffix+'.p')
        test_database_file_m.append(data_fp_prefix+'synth/general_supine/test_roll0_plo_m_lay_set14_1500'+data_fp_suffix+'.p')


        #Hands behind head partition - 4,000 train + 1,000 test
        training_database_file_f.append(data_fp_prefix+'synth/hands_behind_head/train_roll0_plo_hbh_f_lay_set1to2_2000'+data_fp_suffix+'.p')
        training_database_file_m.append(data_fp_prefix+'synth/hands_behind_head/train_roll0_plo_hbh_m_lay_set2pa1_2000'+data_fp_suffix+'.p')
        test_database_file_f.append(data_fp_prefix+'synth/hands_behind_head/test_roll0_plo_hbh_f_lay_set4_500'+data_fp_suffix+'.p')
        test_database_file_m.append(data_fp_prefix+'synth/hands_behind_head/test_roll0_plo_hbh_m_lay_set1_500'+data_fp_suffix+'.p')

        #Prone hands up partition - 8,000 train + 1,000 test
        training_database_file_f.append(data_fp_prefix+'synth/prone_hands_up/train_roll0_plo_phu_f_lay_set2pl4_4000'+data_fp_suffix+'.p')
        training_database_file_m.append(data_fp_prefix+'synth/prone_hands_up/train_roll0_plo_phu_m_lay_set2pl4_4000'+data_fp_suffix+'.p')
        test_database_file_f.append(data_fp_prefix+'synth/prone_hands_up/test_roll0_plo_phu_f_lay_set1pa3_500'+data_fp_suffix+'.p')
        test_database_file_m.append(data_fp_prefix+'synth/prone_hands_up/test_roll0_plo_phu_m_lay_set1pa3_500'+data_fp_suffix+'.p')

        #Straight limbs partition - 8,000 train + 1,000 test
        training_database_file_f.append(data_fp_prefix+'synth/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000'+data_fp_suffix+'.p')
        training_database_file_m.append(data_fp_prefix+'synth/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000'+data_fp_suffix+'.p')
        test_database_file_f.append(data_fp_prefix+'synth/straight_limbs/test_roll0_sl_f_lay_set1both_500'+data_fp_suffix+'.p')
        test_database_file_m.append(data_fp_prefix+'synth/straight_limbs/test_roll0_sl_m_lay_set1both_500'+data_fp_suffix+'.p')

        #Crossed legs partition - 8,000 train + 1,000 test
        training_database_file_f.append(data_fp_prefix+'synth/crossed_legs/train_roll0_xl_f_lay_set2both_4000'+data_fp_suffix+'.p')
        training_database_file_m.append(data_fp_prefix+'synth/crossed_legs/train_roll0_xl_m_lay_set2both_4000'+data_fp_suffix+'.p')
        test_database_file_f.append(data_fp_prefix+'synth/crossed_legs/test_roll0_xl_f_lay_set1both_500'+data_fp_suffix+'.p')
        test_database_file_m.append(data_fp_prefix+'synth/crossed_legs/test_roll0_xl_m_lay_set1both_500'+data_fp_suffix+'.p')


    p = PhysicalTrainer(training_database_file_f, training_database_file_m, test_database_file_f, test_database_file_m, opt)

    p.init_convnet_train()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
