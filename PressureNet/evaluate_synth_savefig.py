#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

sys.path.insert(0, '../lib_py')

import lib_pyrender_br_savefig as libPyRender

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

import convnet_br as convnet
# import tf.transformations as tft

# import hrl_lib.util as ut
import cPickle as pickle


# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


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
NUMOFOUTPUTNODES_TEST = 24
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


class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''

    def __init__(self, testing_database_file_f,
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
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['incl_ht_wt_channels'] = opt.htwt
        self.CTRL_PNL['loss_root'] = opt.loss_root
        self.CTRL_PNL['omit_cntct_sobel'] = opt.omit_cntct_sobel
        self.CTRL_PNL['use_hover'] = opt.use_hover
        self.CTRL_PNL['dropout'] = False
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 2
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        repeat_real_data_ct = 3
        self.CTRL_PNL['regr_angles'] = opt.reg_angles
        self.CTRL_PNL['depth_map_labels'] = False #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['depth_map_labels_test'] = False #can only be true is we have 100% synth for testing
        self.CTRL_PNL['depth_map_output'] = True #self.CTRL_PNL['depth_map_labels']
        self.CTRL_PNL['depth_map_input_est'] = False  #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['depth_map_input_est'] #holds betas and root same as prior estimate
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
        self.CTRL_PNL['L2_contact'] = True
        self.CTRL_PNL['pmat_mult'] = int(1)
        self.CTRL_PNL['cal_noise'] = opt.calnoise
        self.CTRL_PNL['cal_noise_amt'] = 0.2
        self.CTRL_PNL['output_only_prev_est'] = False
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['first_pass'] = True
        self.CTRL_PNL['align_procr'] = False


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

        print self.CTRL_PNL['num_epochs'], 'NUM EPOCHS!'
        # Entire pressure dataset with coordinates in world frame



        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_test = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



        #################################### PREP TESTING DATA ##########################################
        # load in the test file

        test_dat_f_synth = TensorPrepLib().load_files_to_database(testing_database_file_f, creation_type = 'synth', reduce_data = False, test = True)
        test_dat_m_synth = TensorPrepLib().load_files_to_database(testing_database_file_m, creation_type = 'synth', reduce_data = False, test = True)
        test_dat_f_real = TensorPrepLib().load_files_to_database(testing_database_file_f, creation_type = 'real', reduce_data = False, test = True)
        test_dat_m_real = TensorPrepLib().load_files_to_database(testing_database_file_m, creation_type = 'real', reduce_data = False, test = True)

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





    def init_convnet_test(self):

        print self.test_x_tensor.size(), self.test_y_tensor.size()
        #self.test_x_tensor = self.test_x_tensor[476:, :, :, :]
        #self.test_y_tensor = self.test_y_tensor[476:, :]
        print self.test_x_tensor.size(), self.test_y_tensor.size()

        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])


        fc_output_size = 85## 10 + 3 + 24*3 --- betas, root shift, rotations

        if self.CTRL_PNL['full_body_rot'] == True:
            fc_output_size += 3



        if opt.go200 == False:
            self.model = torch.load(FILEPATH_PREFIX + "/convnets_camready/convnet_1_anglesDC_" + NETWORK_1 + "_100e_2e-05lr.pt", map_location='cpu')
            self.model2 = torch.load(FILEPATH_PREFIX + "/convnets_camready/convnet_2_anglesDC_" + NETWORK_2 + "_100e_2e-05lr.pt", map_location='cpu')
        else:
            self.model = torch.load(FILEPATH_PREFIX + "/convnets_camready/convnet_1_anglesDC_" + NETWORK_1 + "_200e_2e-05lr.pt", map_location='cpu')
            self.model2 = None


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
            if self.model2 is not None:
                self.model2 = self.model2.cuda()

        self.model = self.model.eval()
        if self.model2 is not None:
            self.model2 = self.model2.eval()


        # test the model one epoch at a time
        for epoch in range(1):#, self.CTRL_PNL['num_epochs'] + 1):
            self.t1 = time.time()
            #self.val_convnet_special(epoch)
            self.val_convnet_general(epoch)


    def val_convnet_general(self, epoch):

        if GENDER == "m":
            model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.m = load_model(model_path)


        self.pyRender = libPyRender.pyRenderMesh(render = True)

        '''
        Train the model for one epoch.
        '''
        # Some models use slightly different forward passes and train and test
        # time (e.g., any model with Dropout). This puts the model in train mode
        # (as opposed to eval mode) so it knows which one to use.

        RESULTS_DICT = {}
        RESULTS_DICT['j_err'] = []
        RESULTS_DICT['betas'] = []
        RESULTS_DICT['dir_v_err'] = []
        RESULTS_DICT['v2v_err'] = []
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
            for batch_idx, batch in enumerate(self.test_loader):
                if batch_idx > BATCH_IDX_START and batch_idx < 500: #57:

                    batch1 = batch[1].clone()

                    betas_gt = torch.mean(batch[1][:, 72:82], dim = 0).numpy()
                    angles_gt = torch.mean(batch[1][:, 82:154], dim = 0).numpy()
                    root_shift_est_gt = torch.mean(batch[1][:, 154:157], dim = 0).numpy()

                    NUMOFOUTPUTDIMS = 3
                    NUMOFOUTPUTNODES_TEST = 24
                    self.output_size_test = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)

                    self.CTRL_PNL['adjust_ang_from_est'] = False
                    self.CTRL_PNL['depth_map_labels'] = False
                    self.CTRL_PNL['align_procr'] = False

                    print self.CTRL_PNL['num_input_channels_batch0'], batch[0].size()

                    scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpack_batch(batch, False, self.model,
                                                                                                self.CTRL_PNL)

                    mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)  # / 16.69545796387731
                    mdm_est_neg = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)  # / 45.08513083167194
                    mdm_est_pos[mdm_est_pos < 0] = 0
                    mdm_est_neg[mdm_est_neg > 0] = 0
                    mdm_est_neg *= -1
                    cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100  # / 43.55800622930469

                    # 1. / 16.69545796387731,  # pos est depth
                    # 1. / 45.08513083167194,  # neg est depth
                    # 1. / 43.55800622930469,  # cm est

                    sc_sample1 = OUTPUT_DICT['batch_targets_est'].clone()
                    sc_sample1 = sc_sample1[0, :].squeeze() / 1000
                    sc_sample1 = sc_sample1.view(self.output_size_test)
                    # print sc_sample1

                    if self.model2 is not None:
                        print "Using model 2"
                        batch_cor = []

                        if self.CTRL_PNL['cal_noise'] == False:
                            batch_cor.append(torch.cat((batch[0][:, 0:1, :, :],
                                                        mdm_est_pos.type(torch.FloatTensor),
                                                        mdm_est_neg.type(torch.FloatTensor),
                                                        cm_est.type(torch.FloatTensor),
                                                        batch[0][:, 1:, :, :]), dim=1))
                        else:
                            if self.opt.pmr == True:
                                batch_cor.append(torch.cat((mdm_est_pos.type(torch.FloatTensor),
                                                            mdm_est_neg.type(torch.FloatTensor),
                                                            cm_est.type(torch.FloatTensor),
                                                            batch[0][:, 0:, :, :]), dim=1))
                            else:
                                batch_cor.append(batch[0])

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

                        if self.opt.pmr == True:
                            self.CTRL_PNL['num_input_channels_batch0'] += 3

                        print self.CTRL_PNL['num_input_channels_batch0'], batch_cor[0].size()

                        self.CTRL_PNL['align_procr'] = self.opt.align_procr

                        scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpack_batch(batch_cor, is_training=False,
                                                                                        model=self.model2,
                                                                                        CTRL_PNL=self.CTRL_PNL)
                        if self.opt.pmr == True:
                            self.CTRL_PNL['num_input_channels_batch0'] -= 3

                    self.CTRL_PNL['first_pass'] = False



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

                    viz_dim = self.opt.viz_dim

                    if viz_dim == "2D":
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

                        sc_sample = sc_sample.view(self.output_size_test)

                        VisualizationLib().visualize_pressure_map(self.im_sample, sc_sample1, sc_sample,
                                                                  # self.im_sample_ext, None, None,
                                                                  self.im_sample_ext3, None, None,
                                                                  # , self.tar_sample_val, self.sc_sample_val,
                                                                  block=False)


                    elif viz_dim == "3D":
                        pmat = batch[0][0, 0, :, :].clone().numpy()
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


                        save_name = NETWORK_2+'_'+TESTING_FILENAME+'_'+str(batch_idx)
                        if self.opt.align_procr == True: save_name += '_ap'
                        # render everything
                        RESULTS_DICT = self.pyRender.render_mesh_pc_bed_pyrender_everything_synth(smpl_verts, smpl_faces,
                                                                                camera_point, bedangle, RESULTS_DICT,
                                                                                smpl_verts_gt=smpl_verts_gt, pmat=pmat,
                                                                                markers=None,
                                                                                dropout_variance=dropout_variance,
                                                                                save_name = save_name)

                    #time.sleep(300)

                    #print RESULTS_DICT['j_err']
                    print np.mean(np.array(RESULTS_DICT['j_err']), axis = 0)
                    #print RESULTS_DICT['precision']
                    print np.mean(RESULTS_DICT['precision'])
                    print time.time() - init_time
                    #break

        #save here

        #dir = FILEPATH_PREFIX + '/final_results/'+NETWORK_2
        #if not os.path.exists(dir):
        #    os.mkdir(dir)

        #pkl.dump(RESULTS_DICT, open(dir+'/results_synth_'+TESTING_FILENAME+'.p', 'wb'))


if __name__ == "__main__":
    print "Got here"

    import optparse

    p = optparse.OptionParser()

    p.add_option('--losstype', action='store', type = 'string', dest='losstype', default='anglesDC',
                 help='Choose direction cosine or euler angle regression.')

    p.add_option('--gpu', action='store', type = 'string',
                 dest='gpu', \
                 default='0', \
                 help='Set the GPU you will use.')

    p.add_option('--j_d_ratio', action='store', type = 'float', dest='j_d_ratio', default=0.5, #PMR parameter to adjust loss function 2
                 help='Set the loss mix: joints to depth planes. Only used for PMR regression.')

    p.add_option('--test', action='store', type = 'int', dest='test', default=1, #PMR parameter to adjust loss function 2
                 help='Set the testing block.')


    p.add_option('--hd', action='store_true', dest='hd', default=False,
                 help='Read and write to data on an external harddrive.')

    p.add_option('--pmr', action='store_true', dest='pmr', default=False,
                 help='Run PMR on input plus precomputed spatial maps.')

    p.add_option('--small', action='store_true', dest='small', default=False,
                 help='Make the dataset 1/4th of the original size.')

    p.add_option('--htwt', action='store_true', dest='htwt', default=False,
                 help='Include height and weight info on the input.')

    p.add_option('--calnoise', action='store_true', dest='calnoise', default=False,
                 help='Apply calibration noise to the input to facilitate sim to real transfer.')

    p.add_option('--viz_dim', action='store', dest='viz_dim', default='3D',
                 help='Specify `2D` or `3D`.')

    p.add_option('--viz', action='store_true', dest='viz', default=False,
                 help='Visualize training.')

    p.add_option('--go200', action='store_true', dest='go200', default=False,
                 help='Run network 1 for 100 to 200 epochs.')

    p.add_option('--loss_root', action='store_true', dest='loss_root', default=False,
                 help='Use root in loss function.')

    p.add_option('--use_hover', action='store_true', dest='use_hover', default=False,
                 help='Use hovermap for pmr input.')

    p.add_option('--omit_cntct_sobel', action='store_true', dest='omit_cntct_sobel', default=False,
                 help='Cut contact and sobel from input.')

    p.add_option('--align_procr', action='store_true', dest='align_procr', default=False,
                 help='Align procrustes. Only available on synthetic data.')

    p.add_option('--half_shape_wt', action='store_true', dest='half_shape_wt', default=False,
                 help='Half betas.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=True, help='Printout everything (under construction).')

    p.add_option('--log_interval', type=int, default=15, metavar='N',
                 help='number of batches between logging train status')


    p.add_option('--rgangs', action='store_true', dest='reg_angles', default=False, #I found this option doesn't help much.
                 help='Regress the angles as well as betas and joint pos.')




    opt, args = p.parse_args()



    if opt.hd == False:
        FILEPATH_PREFIX = "../data_BR"
    else:
        FILEPATH_PREFIX = "/media/henry/multimodal_data_2/data_BR"


    if opt.small == True:
        NETWORK_1 = "46000ct_"
        NETWORK_2 = "46000ct_"
    else:
        NETWORK_1 = "184000ct_"
        NETWORK_2 = "184000ct_"


    NETWORK_1 += "128b_x1pm_tnh"

    if opt.go200 == True:
        NETWORK_2 += "128b_x1pm_tnh"
    elif opt.pmr == True:
        NETWORK_2 += "128b_x1pm_0.5rtojtdpth_depthestin_angleadj_tnh"
    else:
        NETWORK_2 += "128b_x1pm_angleadj_tnh"


    if opt.htwt == True:
        NETWORK_1 += "_htwt"
        NETWORK_2 += "_htwt"

    if opt.calnoise == True:
        NETWORK_1 += "_clns20p"
        NETWORK_2 += "_clns20p"

    if opt.loss_root == True:
        NETWORK_1 += "_rt"
        NETWORK_2 += "_rt"

    if opt.omit_cntct_sobel == True:
        NETWORK_1 += "_ocs"
        NETWORK_2 += "_ocs"

    if opt.use_hover == True:
        NETWORK_2 += "_uh"

    if opt.half_shape_wt == True:
        NETWORK_1 += "_hsw"
        NETWORK_2 += "_hsw"


    BATCH_IDX_START = 6

    if opt.test == 1:
        testing_filename_list = [#["f", "hands_behind_head/","test_roll0_plo_hbh_f_lay_set4_500"],
                                 ["m","general/","test_rollpi_m_lay_set23to24_3000"],
                                 ["m","general/","test_rollpi_plo_m_lay_set23to24_3000"]]
    elif opt.test == 2:
        testing_filename_list = [["f","general/","test_rollpi_f_lay_set23to24_3000"],
                                 ["f","general/","test_rollpi_plo_f_lay_set23to24_3000"]]

    elif opt.test == 3:
        testing_filename_list = [["m","general_supine/", "test_roll0_m_lay_set14_1500"],
                                 ["m","general_supine/", "test_roll0_plo_m_lay_set14_1500"],
                                  ["m","crossed_legs/", "test_roll0_xl_m_lay_set1both_500"],
                                  ["f","general_supine/", "test_roll0_f_lay_set14_1500"],
                                  ["f","general_supine/", "test_roll0_plo_f_lay_set14_1500"],
                                  ["f","crossed_legs/", "test_roll0_xl_f_lay_set1both_500"]]

    elif opt.test == 4:
        testing_filename_list = [["m", "hands_behind_head/", "test_roll0_plo_hbh_m_lay_set1_500"],
                                 ["m", "prone_hands_up/", "test_roll0_plo_phu_m_lay_set1pa3_500"],
                                 ["m", "straight_limbs/",  "test_roll0_sl_m_lay_set1both_500"],
                                 ["f", "hands_behind_head/", "test_roll0_plo_hbh_f_lay_set4_500"],
                                 ["f", "prone_hands_up/", "test_roll0_plo_phu_f_lay_set1pa3_500"],
                                 ["f", "straight_limbs/", "test_roll0_sl_f_lay_set1both_500"]]

    for item_test in testing_filename_list:
        GENDER = item_test[0]
        PARTITION = item_test[1]
        TESTING_FILENAME = item_test[2]

        test_database_file_f = []
        test_database_file_m = [] #141 total training loss at epoch 9



        if GENDER == "f":
            test_database_file_f.append(FILEPATH_PREFIX+'/synth/'+PARTITION+TESTING_FILENAME+'.p')
        else:
            test_database_file_m.append(FILEPATH_PREFIX+'/synth/'+PARTITION+TESTING_FILENAME+'.p')


        p = PhysicalTrainer(test_database_file_f, test_database_file_m, opt)

        p.init_convnet_test()

            #else:
            #    print 'Please specify correct training type:1. HoG_KNN 2. convnet_2'
