#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import lib_pyrender as libPyRender

# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from smpl.smpl_webuser.serialization import load_model
import lib_kinematics as libKinematics
import chumpy as ch
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)


import convnet as convnet
# import tf.transformations as tft

# import hrl_lib.util as ut
import cPickle as pickle


# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Pose Estimation Libraries
from visualization_lib import VisualizationLib
from preprocessing_lib import PreprocessingLib
from synthetic_lib import SyntheticLib
from tensorprep_lib import TensorPrepLib
from unpack_batch_lib import UnpackBatchLib

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom
# from skimage.feature import hog
# from skimage import data, color, exposure


#from sklearn.cluster import KMeans
#from sklearn.preprocessing import scale
#from sklearn.preprocessing import normalize
#from sklearn import svm, linear_model, decomposition, kernel_ridge, neighbors
#from sklearn import metrics
#from sklearn.utils import shuffle
#from sklearn.multioutput import MultiOutputRegressor

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
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)
TEST_SUBJECT = 9
CAM_BED_DIST = 1.66
DEVICE = 0

torch.set_num_threads(1)
if False:#torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    torch.cuda.set_device(DEVICE)
    print '######################### CUDA is available! #############################'
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    print '############################## USING CPU #################################'


#TESTING_FILENAME = "test_roll0_xl_m_lay_set1both_500"
TESTING_FILENAME = "test_roll0_plo_m_lay_set14_1500"
GENDER = "m"
#NETWORK_1 = "1.0rtojtdpth_tnh_htwt_calnoise"
NETWORK_1 = "1.0rtojtdpth_tnh_htwt_clns10p"
#NETWORK_2 = "1.0rtojtdpth_angleadj_tnhFIXN_htwt_calnoise"
DATA_QUANT = "184K"
NETWORK_2 = "0.5rtojtdpth_depthestin_angleadj_tnh_htwt_clns10p"

class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''

    def __init__(self, training_database_file_f, training_database_file_m, testing_database_file_f,
                 testing_database_file_m, opt):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''

        # change this to 'direct' when you are doing baseline methods

        self.CTRL_PNL = {}
        self.CTRL_PNL['loss_vector_type'] = opt.losstype

        self.CTRL_PNL['verbose'] = opt.verbose
        self.opt = opt
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['num_epochs'] = 100
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = True
        self.CTRL_PNL['omit_root'] = opt.omit_root
        self.CTRL_PNL['omit_cntct_sobel'] = opt.omit_cntct_sobel
        self.CTRL_PNL['align_procr'] = opt.align_procr
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['dropout'] = False
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 3
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        repeat_real_data_ct = 3
        self.CTRL_PNL['regr_angles'] = opt.reg_angles
        self.CTRL_PNL['aws'] = self.opt.aws
        self.CTRL_PNL['depth_map_labels'] = True #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['depth_map_labels_test'] = True #can only be true is we have 100% synth for testing
        self.CTRL_PNL['depth_map_output'] = self.CTRL_PNL['depth_map_labels']
        self.CTRL_PNL['depth_map_input_est'] = False  #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['depth_map_input_est'] #holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True
        self.CTRL_PNL['normalize_input'] = True
        self.CTRL_PNL['all_tanh_activ'] = True
        self.CTRL_PNL['L2_contact'] = True
        self.CTRL_PNL['pmat_mult'] = int(5)
        self.CTRL_PNL['cal_noise'] = False
        self.CTRL_PNL['cal_noise_amt'] = 0.1
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['first_pass'] = True


        self.weight_joints = 1.0#self.opt.j_d_ratio*2
        self.weight_depth_planes = (1-self.opt.j_d_ratio)#*2

        if opt.losstype == 'direct':
            self.CTRL_PNL['depth_map_labels'] = False
            self.CTRL_PNL['depth_map_output'] = False

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
                                             1./1.0,                #bed height mat
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1. / 30.216647403350,  #weight
                                             1. / 14.629298141231]  #height


        self.CTRL_PNL['filepath_prefix'] = '~/'

        if self.CTRL_PNL['depth_map_output'] == True: #we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]

        print self.CTRL_PNL['num_epochs'], 'NUM EPOCHS!'
        # Entire pressure dataset with coordinates in world frame

        self.save_name = '_' + opt.losstype + \
                         '_synth_32000' + \
                         '_' + str(self.CTRL_PNL['batch_size']) + 'b' + \
                         '_' + str(self.CTRL_PNL['num_epochs']) + 'e' + \
                         '_x' + str(self.CTRL_PNL['pmat_mult']) + 'pmult'


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
        if  self.CTRL_PNL['omit_root'] == True:
            self.save_name += '_or'
        if  self.CTRL_PNL['omit_cntct_sobel'] == True:
            self.save_name += '_ocs'
        if  self.CTRL_PNL['align_procr'] == True:
            self.save_name += '_ap'


        # self.save_name = '_' + opt.losstype+'_real_s9_alltest_' + str(self.CTRL_PNL['batch_size']) + 'b_'# + str(self.CTRL_PNL['num_epochs']) + 'e'

        print 'appending to', 'train' + self.save_name
        self.train_val_losses = {}
        self.train_val_losses['train' + self.save_name] = []
        self.train_val_losses['val' + self.save_name] = []
        self.train_val_losses['epoch' + self.save_name] = []

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



        #################################### PREP TRAINING DATA ##########################################
        #load training ysnth data
        dat_f_synth = TensorPrepLib().load_files_to_database(training_database_file_f, 'synth')
        dat_m_synth = TensorPrepLib().load_files_to_database(training_database_file_m, 'synth')
        dat_f_real = TensorPrepLib().load_files_to_database(training_database_file_f, 'real')
        dat_m_real = TensorPrepLib().load_files_to_database(training_database_file_m, 'real')


        self.train_x_flat = []  # Initialize the testing pressure mat list
        self.train_x_flat = TensorPrepLib().prep_images(self.train_x_flat, dat_f_synth, dat_m_synth, num_repeats = 1)
        self.train_x_flat = list(np.clip(np.array(self.train_x_flat) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100))
        self.train_x_flat = TensorPrepLib().prep_images(self.train_x_flat, dat_f_real, dat_m_real, num_repeats = repeat_real_data_ct)

        if self.CTRL_PNL['cal_noise'] == False:
            self.train_x_flat = PreprocessingLib().preprocessing_blur_images(self.train_x_flat, self.mat_size, sigma=0.5)

        if len(self.train_x_flat) == 0: print("NO TRAINING DATA INCLUDED")

        self.train_a_flat = []  # Initialize the training pressure mat angle list
        self.train_a_flat = TensorPrepLib().prep_angles(self.train_a_flat, dat_f_synth, dat_m_synth, num_repeats = 1)
        self.train_a_flat = TensorPrepLib().prep_angles(self.train_a_flat, dat_f_real, dat_m_real, num_repeats = repeat_real_data_ct)

        if self.CTRL_PNL['depth_map_labels'] == True:
            self.depth_contact_maps = [] #Initialize the precomputed depth and contact maps. only synth has this label.
            self.depth_contact_maps = TensorPrepLib().prep_depth_contact(self.depth_contact_maps, dat_f_synth, dat_m_synth, num_repeats = 1)
        else:
            self.depth_contact_maps = None

        if self.CTRL_PNL['depth_map_input_est'] == True:
            self.depth_contact_maps_input_est = [] #Initialize the precomputed depth and contact map input estimates
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(self.depth_contact_maps_input_est,
                                                                                             dat_f_synth, dat_m_synth, num_repeats = 1)
            self.depth_contact_maps_input_est = TensorPrepLib().prep_depth_contact_input_est(self.depth_contact_maps_input_est,
                                                                                             dat_f_real, dat_m_real, num_repeats = repeat_real_data_ct)
        else:
            self.depth_contact_maps_input_est = None

        #stack the bed height array on the pressure image as well as a sobel filtered image
        train_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.train_x_flat,
                                                                                self.train_a_flat,
                                                                                self.mat_size,
                                                                                self.CTRL_PNL)

        #stack the depth and contact mesh images (and possibly a pmat contact image) together
        train_xa = TensorPrepLib().append_input_depth_contact(np.array(train_xa),
                                                              CTRL_PNL = self.CTRL_PNL,
                                                              mesh_depth_contact_maps_input_est = self.depth_contact_maps_input_est,
                                                              mesh_depth_contact_maps = self.depth_contact_maps)

        #normalize the input
        if self.CTRL_PNL['normalize_input'] == True:
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

        train_y_flat = TensorPrepLib().prep_labels(train_y_flat, dat_f_real, num_repeats = repeat_real_data_ct,
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                        full_body_rot = self.CTRL_PNL['full_body_rot'])
        train_y_flat = TensorPrepLib().prep_labels(train_y_flat, dat_m_real, num_repeats = repeat_real_data_ct,
                                                        z_adj = 0.0, gender = "m", is_synth = False,
                                                        loss_vector_type = self.CTRL_PNL['loss_vector_type'],
                                                        initial_angle_est = self.CTRL_PNL['adjust_ang_from_est'],
                                                        full_body_rot = self.CTRL_PNL['full_body_rot'])

        # normalize the height and weight
        if self.CTRL_PNL['normalize_input'] == True:
            train_y_flat = TensorPrepLib().normalize_wt_ht(train_y_flat, self.CTRL_PNL)

        self.train_y_tensor = torch.Tensor(train_y_flat)

        print self.train_x_tensor.shape, 'Input training tensor shape'
        print self.train_y_tensor.shape, 'Output training tensor shape'




        #################################### PREP TESTING DATA ##########################################
        # load in the test file
        test_dat_f_synth = TensorPrepLib().load_files_to_database(testing_database_file_f, 'synth')
        test_dat_m_synth = TensorPrepLib().load_files_to_database(testing_database_file_m, 'synth')
        test_dat_f_real = TensorPrepLib().load_files_to_database(testing_database_file_f, 'real')
        test_dat_m_real = TensorPrepLib().load_files_to_database(testing_database_file_m, 'real')

        self.test_x_flat = []  # Initialize the testing pressure mat list
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f_synth, test_dat_m_synth, num_repeats = 1)
        self.test_x_flat = list(np.clip(np.array(self.test_x_flat) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100))
        self.test_x_flat = TensorPrepLib().prep_images(self.test_x_flat, test_dat_f_real, test_dat_m_real, num_repeats = 1)

        if self.CTRL_PNL['cal_noise'] == False:
            self.test_x_flat = PreprocessingLib().preprocessing_blur_images(self.test_x_flat, self.mat_size, sigma=0.5)

        if len(self.test_x_flat) == 0: print("NO TESTING DATA INCLUDED")

        self.test_a_flat = []  # Initialize the testing pressure mat angle listhave
        self.test_a_flat = TensorPrepLib().prep_angles(self.test_a_flat, test_dat_f_synth, test_dat_m_synth, num_repeats = 1)
        self.test_a_flat = TensorPrepLib().prep_angles(self.test_a_flat, test_dat_f_real, test_dat_m_real, num_repeats = 1)


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

        print np.shape(self.test_x_flat), np.shape(self.test_a_flat)

        test_xa = PreprocessingLib().preprocessing_create_pressure_angle_stack(self.test_x_flat,
                                                                               self.test_a_flat,
                                                                                self.mat_size,
                                                                                self.CTRL_PNL)


        test_xa = TensorPrepLib().append_input_depth_contact(np.array(test_xa),
                                                              CTRL_PNL = self.CTRL_PNL,
                                                              mesh_depth_contact_maps_input_est = self.depth_contact_maps_input_est,
                                                              mesh_depth_contact_maps = self.depth_contact_maps)

        #normalize the input
        if self.CTRL_PNL['normalize_input'] == True:
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

        if self.CTRL_PNL['normalize_input'] == True:
            test_y_flat = TensorPrepLib().normalize_wt_ht(test_y_flat, self.CTRL_PNL)

        self.test_y_tensor = torch.Tensor(test_y_flat)


        print self.test_x_tensor.shape, 'Input testing tensor shape'
        print self.test_y_tensor.shape, 'Output testing tensor shape'



    def init_convnet_train(self):

        print self.train_x_tensor.size(), self.train_y_tensor.size()
        #self.train_x_tensor = self.train_x_tensor[476:, :, :, :]
        #self.train_y_tensor = self.train_y_tensor[476:, :]
        print self.train_x_tensor.size(), self.train_y_tensor.size()

        #self.train_x_tensor = self.train_x_tensor.unsqueeze(1)
        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])

        print "Loading convnet model................................"
        if self.CTRL_PNL['loss_vector_type'] == 'direct':
            fc_output_size = 30
            self.model = convnet.CNN(fc_output_size, self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'],
                                     verts_list = self.verts_list, filepath=self.CTRL_PNL['filepath_prefix'],
                                     in_channels=self.CTRL_PNL['num_input_channels'])

        elif self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':
            fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations

            if self.CTRL_PNL['full_body_rot'] == True:
                fc_output_size += 3

            #self.model = convnet.CNN(fc_output_size, self.CTRL_PNL['loss_vector_type'], self.CTRL_PNL['batch_size'],
            #                         verts_list = self.verts_list, filepath=self.CTRL_PNL['filepath_prefix'], in_channels=self.CTRL_PNL['num_input_channels'])

            self.model = torch.load("/home/NAME/data/convnets/planesreg/FINAL/convnet_anglesDC_synth_184000_128b_x5pmult_"+NETWORK_1+"_100e_00002lr.pt", map_location='cpu')
            self.model2 = torch.load("/home/NAME/data/convnets/planesreg_correction/FINAL/convnet_anglesDC_synth_184000_128b_x5pmult_"+NETWORK_2+"_100e_00002lr.pt", map_location='cpu')
            #self.model2 = None
        pp = 0
        for p in list(self.model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        print 'LOADED. num params: ', pp


        # Run model on GPU if available
        #if torch.cuda.is_available():
        if GPU == True:
            self.model = self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00002, weight_decay=0.0005) #start with .00005

        # train the model one epoch at a time
        for epoch in range(1):#, self.CTRL_PNL['num_epochs'] + 1):
            self.t1 = time.time()
            #self.val_convnet_special(epoch)
            self.val_convnet_general(epoch)


    def val_convnet_special(self, epoch):
        if GENDER == "f":
            model_path = '../smpl/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = '../smpl/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'

        self.m = load_model(model_path)

        self.pyRender = libPyRender.pyRenderMesh()

        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.
        self.model.eval()#train()
        self.model2.eval()#train()
        with torch.autograd.set_detect_anomaly(True):

            # This will loop a total = training_images/batch_size times
            for batch_idx, batch in enumerate(self.train_loader):
                if GPU == True:
                    print "GPU memory:", torch.cuda.max_memory_allocated()
                if self.CTRL_PNL['loss_vector_type'] == 'direct':

                    self.optimizer.zero_grad()
                    scores, INPUT_DICT, OUTPUT_DICT = \
                        UnpackBatchLib().unpackage_batch_dir_pass(batch, is_training=True, model=self.model, CTRL_PNL = self.CTRL_PNL)

                    scores_zeros = np.zeros((batch[0].numpy().shape[0], 10))  # 24 is joint euclidean errors
                    scores_zeros = Variable(torch.Tensor(scores_zeros).type(dtype))

                    loss = self.criterion(scores, scores_zeros)



                elif self.CTRL_PNL['loss_vector_type'] == 'anglesR' or self.CTRL_PNL['loss_vector_type'] == 'anglesDC' or self.CTRL_PNL['loss_vector_type'] == 'anglesEU':
                    #print torch.cuda.max_memory_allocated(), '1pre train'
                    betas_gt = torch.mean(batch[1][:, 72:82], dim = 0)
                    angles_gt = torch.mean(batch[1][:, 82:154], dim = 0)
                    root_shift_est_gt = torch.mean(batch[1][:, 154:157], dim = 0)

                    batch_cloned = []
                    batch_cloned.append(batch[0].clone())
                    batch_cloned.append(batch[1].clone())


                    self.CTRL_PNL['output_only_prev_est'] = True
                    scoresp, INPUT_DICTp, OUTPUT_DICTp = \
                        UnpackBatchLib().unpackage_batch_kin_pass(batch, is_training=False, model = self.model, CTRL_PNL=self.CTRL_PNL)
                    #print torch.cuda.max_memory_allocated(), '1post train'

                    self.CTRL_PNL['first_pass'] = False

                    self.CTRL_PNL['output_only_prev_est'] = False
                    scores, INPUT_DICT, OUTPUT_DICT = \
                        UnpackBatchLib().unpackage_batch_kin_pass(batch_cloned, is_training=False, model = self.model, CTRL_PNL=self.CTRL_PNL)
                    #print torch.cuda.max_memory_allocated(), '1post train'

                    betas_est_prev = OUTPUT_DICTp['batch_betas_est_post_clip'].cpu().numpy()
                    angles_est_prev = OUTPUT_DICTp['batch_angles_est_post_clip']
                    root_shift_est_prev = OUTPUT_DICTp['batch_root_xyz_est_post_clip'].cpu().numpy()

                    angles_est_prev = torch.mean(angles_est_prev, dim=0).reshape(72)
                    betas_est_prev = np.mean(betas_est_prev, axis=0)
                    root_shift_est_prev = np.mean(root_shift_est_prev, axis=0)

                    betas_est = OUTPUT_DICT['batch_betas_est_post_clip'].cpu().numpy()
                    angles_est = OUTPUT_DICT['batch_angles_est_post_clip']
                    root_shift_est = OUTPUT_DICT['batch_root_xyz_est_post_clip'].cpu().numpy()

                    angles_est = torch.mean(angles_est, dim=0).reshape(72)
                    betas_est = np.mean(betas_est, axis=0)
                    root_shift_est = np.mean(root_shift_est, axis=0)



                    #print betas_est.shape, betas_net1
                    smpl_verts_both = []
                    for smpl_model in ['ground_truth', 'network_output1', 'network_output2']:
                        for idx in range(10):
                            if smpl_model == 'ground_truth':
                                self.m.betas[idx] = float(betas_gt[idx])
                            elif smpl_model == 'network_output1':
                                self.m.betas[idx] = betas_est_prev[idx]
                            elif smpl_model == 'network_output2':
                                self.m.betas[idx] = betas_est[idx]

                        for idx in range(72):
                            if smpl_model == 'ground_truth':
                                self.m.pose[idx] = float(angles_gt[idx])
                            elif smpl_model == 'network_output1':
                                self.m.pose[idx] = angles_est_prev[idx]
                            elif smpl_model == 'network_output2':
                                self.m.pose[idx] = angles_est[idx]

                        init_root = np.array(self.m.pose[0:3]) + 0.000001
                        init_rootR = libKinematics.matrix_from_dir_cos_angles(init_root)
                        root_rot = libKinematics.eulerAnglesToRotationMatrix([np.pi, 0.0, np.pi / 2])
                        # print root_rot
                        trans_root = libKinematics.dir_cos_angles_from_matrix(np.matmul(root_rot, init_rootR))

                        self.m.pose[0] = trans_root[0]
                        self.m.pose[1] = trans_root[1]
                        self.m.pose[2] = trans_root[2]

                        if smpl_model == 'ground_truth':
                            root_shift = np.copy(root_shift_est_gt.numpy())
                        elif smpl_model == 'network_output1':
                            root_shift = root_shift_est_prev
                        elif smpl_model == 'network_output2':
                            root_shift = root_shift_est

                        print root_shift, 'root shift'
                        # get SMPL mesh
                        smpl_verts = (self.m.r - self.m.J_transformed[0, :]) + [root_shift[1] - 0.286 + 0.15,
                                                                                root_shift[0] - 0.286,
                                                                                0.12 - root_shift[2]]  # *228./214.
                        smpl_faces = np.array(self.m.f)

                        smpl_verts_both.append(smpl_verts)

                    camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]

                    bedangle=0

                    print INPUT_DICT['batch_images'].shape

                    # render in 3D pyrender with pressure mat
                    viz_type = "leg_correction"
                    #viz_type = "arm_penetration"


                    ims_to_display = []
                    if viz_type == "arm_penetration":
                        ims_to_display.append(INPUT_DICT['batch_images'][0, 4, :]*0. + 75.) #uniform
                        segment_limbs = True
                    elif viz_type == "leg_correction":
                        ims_to_display.append(INPUT_DICT['batch_images'][0, 4, :]*23.) #pmat
                        ims_to_display.append(INPUT_DICT['batch_images'][0, 2, :]*45.) #Q-, 1
                        ims_to_display.append(INPUT_DICT['batch_mdm'].data.numpy().reshape(64, 27)*-1) #Q-, GT
                        ims_to_display.append(INPUT_DICT['batch_images'][0, 4, :]*23.) #pmat
                        ims_to_display.append(INPUT_DICT['batch_images'][0, 2, :]*45.) #Q-, 1
                        ims_to_display.append(OUTPUT_DICT['batch_mdm_est'].data.numpy().reshape(64, 27)*-1) #Q-, 2
                        ims_to_display.append(INPUT_DICT['batch_mdm'].data.numpy().reshape(64, 27)*-1) #Q-, GT
                        segment_limbs = True

                    print np.shape(ims_to_display[-1])

                    self.pyRender.render_mesh_bed_special(smpl_verts_both, smpl_faces, bedangle,
                                                              pmats=ims_to_display, viz_type=viz_type,
                                                              segment_limbs=segment_limbs)


                    #self.pyRender.render_only_human_gt(self.m)

                    time.sleep(300)

    def val_convnet_general(self, epoch):

        if GENDER == "m":
            model_path = '../smpl/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = '../smpl/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.m = load_model(model_path)


        self.pyRender = libPyRender.pyRenderMesh(render = True)

        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.
        self.model.eval()#train()
        self.model2.eval()#train()

        RESULTS_DICT = {}
        RESULTS_DICT['j_err'] = []
        RESULTS_DICT['betas'] = []
        RESULTS_DICT['dir_v_err'] = []
        RESULTS_DICT['dir_v_limb_err'] = []
        RESULTS_DICT['v_to_gt_err'] = []
        RESULTS_DICT['v_limb_to_gt_err'] = []
        RESULTS_DICT['gt_to_v_err'] = []
        RESULTS_DICT['precision'] = []
        RESULTS_DICT['recall'] = []
        RESULTS_DICT['overlap_d_err'] = []
        RESULTS_DICT['all_d_err'] = []
        init_time = time.time()

        with torch.autograd.set_detect_anomaly(True):

            # This will loop a total = training_images/batch_size times
            for batch_idx, batch in enumerate(self.train_loader):

                batch1 = batch[1].clone()

                betas_gt = torch.mean(batch[1][:, 72:82], dim = 0).numpy()
                angles_gt = torch.mean(batch[1][:, 82:154], dim = 0).numpy()
                root_shift_est_gt = torch.mean(batch[1][:, 154:157], dim = 0).numpy()

                NUMOFOUTPUTDIMS = 3
                NUMOFOUTPUTNODES_TRAIN = 24
                self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)

                self.CTRL_PNL['adjust_ang_from_est'] = False
                self.CTRL_PNL['depth_map_labels'] = True
                print batch[0].size(), "batch 0 shape"


                print torch.max(batch[0][0, 0, : ,:]), torch.max(batch[0][0, 1, : ,:]), torch.max(batch[0][0, 2, : ,:]), torch.max(batch[0][0, 3, : ,:]), torch.max(batch[0][0, 4, : ,:]), torch.max(batch[0][0, 5, : ,:]),


                scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpackage_batch_kin_pass(batch, False, self.model,
                                                                                            self.CTRL_PNL)
                print OUTPUT_DICT['batch_betas_est_post_clip'].cpu().numpy()[0], 'betas init'
                mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1) / 16.69545796387731
                mdm_est_neg = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1) / 45.08513083167194
                mdm_est_pos[mdm_est_pos < 0] = 0
                mdm_est_neg[mdm_est_neg > 0] = 0
                mdm_est_neg *= -1
                cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100 / 43.55800622930469

                # 1. / 16.69545796387731,  # pos est depth
                # 1. / 45.08513083167194,  # neg est depth
                # 1. / 43.55800622930469,  # cm est

                sc_sample1 = OUTPUT_DICT['batch_targets_est'].clone()
                sc_sample1 = sc_sample1[0, :].squeeze() / 1000
                sc_sample1 = sc_sample1.view(self.output_size_train)
                # print sc_sample1

                if self.model2 is not None:
                    print "Using model 2"
                    batch_cor = []
                    batch_cor.append(torch.cat((batch[0][:, 0:1, :, :],
                                                mdm_est_pos.type(torch.FloatTensor),
                                                mdm_est_neg.type(torch.FloatTensor),
                                                cm_est.type(torch.FloatTensor),
                                                batch[0][:, 1:, :, :]), dim=1))


                    if self.CTRL_PNL['full_body_rot'] == False:
                        batch_cor.append(torch.cat((batch1,
                                                    OUTPUT_DICT['batch_betas_est'].cpu(),
                                                    OUTPUT_DICT['batch_angles_est'].cpu(),
                                                    OUTPUT_DICT['batch_root_xyz_est'].cpu()), dim=1))
                    elif self.CTRL_PNL['full_body_rot'] == True:
                        batch_cor.append(torch.cat((batch1,
                                                    OUTPUT_DICT['batch_betas_est'].cpu(),
                                                    OUTPUT_DICT['batch_angles_est'].cpu(),
                                                    OUTPUT_DICT['batch_root_xyz_est'].cpu(),
                                                    OUTPUT_DICT['batch_root_atan2_est'].cpu()), dim=1))



                    self.CTRL_PNL['adjust_ang_from_est'] = True
                    self.CTRL_PNL['depth_map_labels'] = False
                    scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpackage_batch_kin_pass(batch_cor, False,
                                                                                                self.model2,
                                                                                                self.CTRL_PNL)

                # print betas_est, root_shift_est, angles_est
                if self.CTRL_PNL['dropout'] == True:
                    #print OUTPUT_DICT['verts'].shape
                    smpl_verts = np.mean(OUTPUT_DICT['verts'], axis=0)
                    dropout_variance = np.std(OUTPUT_DICT['verts'], axis=0)
                    dropout_variance = np.linalg.norm(dropout_variance, axis=1)
                else:
                    smpl_verts = OUTPUT_DICT['verts'][0, :, :]
                    dropout_variance = None




                smpl_verts = np.concatenate((smpl_verts[:, 1:2] - 0.286 + 0.0143, smpl_verts[:, 0:1] - 0.286 + 0.0143,
                                             - smpl_verts[:, 2:3]), axis=1)

                smpl_faces = np.array(self.m.f)


                q = OUTPUT_DICT['batch_mdm_est'].data.numpy().reshape(OUTPUT_DICT['batch_mdm_est'].size()[0], 64, 27) * -1
                q = np.mean(q, axis=0)

                camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]

                bedangle = 0.0
                # print smpl_verts

                RESULTS_DICT['betas'].append(OUTPUT_DICT['batch_betas_est_post_clip'].cpu().numpy()[0])
                print RESULTS_DICT['betas'][-1], "BETAS"

                viz_type = "3D"

                if viz_type == "2D":
                    from visualization_lib import VisualizationLib
                    if self.model2 is not None:
                        self.im_sample = INPUT_DICT['batch_images'][0, 4:,:].squeeze() * 20.  # normalizing_std_constants[4]*5.  #pmat
                    else:
                        self.im_sample = INPUT_DICT['batch_images'][0, 1:,:].squeeze() * 20.  # normalizing_std_constants[4]*5.  #pmat
                    self.im_sample_ext = INPUT_DICT['batch_images'][0, 0:,:].squeeze() * 20.  # normalizing_std_constants[0]  #pmat contact
                    # self.im_sample_ext2 = INPUT_DICT['batch_images'][im_display_idx, 2:, :].squeeze()*20.#normalizing_std_constants[4]  #sobel
                    self.im_sample_ext3 = OUTPUT_DICT['batch_mdm_est'][0, :, :].squeeze().unsqueeze(0) * -1  # est depth output

                    # print scores[0, 10:16], 'scores of body rot'

                    # print self.im_sample.size(), self.im_sample_ext.size(), self.im_sample_ext2.size(), self.im_sample_ext3.size()

                    # self.publish_depth_marker_array(self.im_sample_ext3)

                    self.tar_sample = INPUT_DICT['batch_targets']
                    self.tar_sample = self.tar_sample[0, :].squeeze() / 1000
                    sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
                    sc_sample = sc_sample[0, :].squeeze() / 1000

                    sc_sample = sc_sample.view(self.output_size_train)

                    VisualizationLib().visualize_pressure_map(self.im_sample, sc_sample1, sc_sample,
                                                              # self.im_sample_ext, None, None,
                                                              self.im_sample_ext3, None, None,
                                                              # , self.tar_sample_val, self.sc_sample_val,
                                                              block=False)


                elif viz_type == "3D":
                    pmat = batch[0][0, 1, :, :].clone().numpy()*25.50538629767412
                    #print pmat.shape

                    for beta in range(betas_gt.shape[0]):
                        self.m.betas[beta] = betas_gt[beta]
                    for angle in range(angles_gt.shape[0]):
                        self.m.pose[angle] = angles_gt[angle]

                    smpl_verts_gt = np.array(self.m.r)
                    for s in range(root_shift_est_gt.shape[0]):
                        smpl_verts_gt[:, s] += (root_shift_est_gt[s] - float(self.m.J_transformed[0, s]))

                    smpl_verts_gt = np.concatenate(
                        (smpl_verts_gt[:, 1:2] - 0.286 + 0.0143, smpl_verts_gt[:, 0:1] - 0.286 + 0.0143,
                          0.0 - smpl_verts_gt[:, 2:3]), axis=1)



                    joint_cart_gt = np.array(self.m.J_transformed).reshape(24, 3)
                    for s in range(root_shift_est_gt.shape[0]):
                        joint_cart_gt[:, s] += (root_shift_est_gt[s] - float(self.m.J_transformed[0, s]))

                    #print joint_cart_gt, 'gt'

                    sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
                    sc_sample = (sc_sample[0, :].squeeze().numpy() / 1000).reshape(24, 3)

                    #print sc_sample, 'estimate'
                    joint_error = np.linalg.norm(joint_cart_gt-sc_sample, axis = 1)
                    #print joint_error
                    RESULTS_DICT['j_err'].append(joint_error)


                    camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]

                    # render everything
                    RESULTS_DICT = self.pyRender.render_mesh_pc_bed_pyrender_everything_synth(smpl_verts, smpl_faces,
                                                                            camera_point, bedangle, RESULTS_DICT,
                                                                            smpl_verts_gt=smpl_verts_gt, pmat=pmat,
                                                                            markers=None,
                                                                            dropout_variance=dropout_variance)

                #time.sleep(300)

                #print RESULTS_DICT['j_err']
                print np.mean(np.array(RESULTS_DICT['j_err']), axis = 0)
                #print RESULTS_DICT['precision']
                print np.mean(RESULTS_DICT['precision'])
                print time.time() - init_time
                #break

        #save here

        pkl.dump(RESULTS_DICT, open('~/data/final_results/results_synth_'+DATA_QUANT+'_'+TESTING_FILENAME+'_'+NETWORK_2+'.p', 'wb'))

if __name__ == "__main__":
    #Initialize trainer with a training database file

    #from visualization_msgs.msg import MarkerArray
    #from visualization_msgs.msg import Marker
    #import rospy

    #rospy.init_node('depth_cam_node')
    #pointcloudPublisher = rospy.Publisher("/point_cloud", MarkerArray)

    #import rospy

    #rospy.init_node('pose_trainer')

    import optparse
    p = optparse.OptionParser()
    p.add_option('--computer', action='store', type = 'string',
                 dest='computer', \
                 default='lab_harddrive', \
                 help='Set path to the training database on lab harddrive.')
    p.add_option('--gpu', action='store', type = 'string',
                 dest='gpu', \
                 default='0', \
                 help='Set the GPU you will use.')
    p.add_option('--losstype', action='store', type = 'string',
                 dest='losstype', \
                 default='anglesDC', \
                 help='Set if you want to do baseline ML or convnet.')
    p.add_option('--j_d_ratio', action='store', type = 'float',
                 dest='j_d_ratio', \
                 default=0.5, \
                 help='Set the loss mix: joints to depth planes.')
    p.add_option('--qt', action='store_true',
                 dest='quick_test', \
                 default=True,\
                 help='Do a quick test.')
    p.add_option('--viz', action='store_true',
                 dest='visualize', \
                 default=False, \
                 help='Visualize.')

    p.add_option('--omit_root', action='store_true',
                 dest='omit_root',
                 default=False,
                 help='Cut root from loss function.')

    p.add_option('--omit_cntct_sobel', action='store_true',
                 dest='omit_cntct_sobel',
                 default=False,
                 help='Cut contact and sobel from input.')

    p.add_option('--align_procr', action='store_true',
                 dest='align_procr',
                 default=False,
                 help='Align the procrustes. Works only on synthetic data.')

    p.add_option('--aws', action='store_true',
                 dest='aws', \
                 default=False, \
                 help='Use ubuntu user dir instead of NAME.')
    p.add_option('--rgangs', action='store_true',
                 dest='reg_angles', \
                 default=False, \
                 help='Regress the angles as well as betas and joint pos.')
    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=True, help='Printout everything (under construction).')

    p.add_option('--log_interval', type=int, default=15, metavar='N',
                 help='number of batches between logging train status')

    opt, args = p.parse_args()


    filepath_prefix = '~/data/'

    training_database_file_f = []
    training_database_file_m = []
    test_database_file_f = []
    test_database_file_m = [] #141 total training loss at epoch 9




    #training_database_file_f.append(filepath_prefix+'synth/random3_fix/test_roll0_f_lay_set14_1500.p') #were actually testing this one

    if GENDER == "f":
        training_database_file_f.append(filepath_prefix+'synth/random3/'+TESTING_FILENAME+'.p') #were actually testing this one
        test_database_file_f.append(filepath_prefix+'synth/random3/'+TESTING_FILENAME+'.p')
    else:
        training_database_file_m.append(filepath_prefix+'synth/random3/'+TESTING_FILENAME+'.p') #were actually testing this one
        test_database_file_m.append(filepath_prefix+'synth/random3/'+TESTING_FILENAME+'.p')


    p = PhysicalTrainer(training_database_file_f, training_database_file_m, test_database_file_f, test_database_file_m, opt)

    p.init_convnet_train()

        #else:
        #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
