#!/usr/bin/env python

#Bodies at Rest: Code to visualize synthetic dataset.
#(c) Henry M. Clever
#Major updates made for CVPR release: December 10, 2019


import sys
import os
import time
import numpy as np

import lib_pyrender_br as libPyRender

# PyTorch libraries. we use pytorch to load in the data. there are better ways to do it but this is what I've done--
# the data here is loaded the same as it is for PressureNet training.
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
from smpl.smpl_webuser.serialization import load_model

import cPickle as pickle


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Pose Estimation Libraries


import sys
sys.path.insert(0, '../lib_py')

from preprocessing_lib_br import PreprocessingLib
from tensorprep_lib_br import TensorPrepLib

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom


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


class PhysicalTrainer():

    def __init__(self, training_database_file_f, training_database_file_m):
        #if this is your first time looking at the code don't pay much attention to this __init__ function.
        #it's all just boilerplate for loading in the synthetic data

        self.CTRL_PNL = {}
        self.CTRL_PNL['loss_vector_type'] = 'anglesDC'
        self.CTRL_PNL['verbose'] = False
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['num_epochs'] = 100
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = False
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['dropout'] = False
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 2
        self.CTRL_PNL['GPU'] = False
        self.CTRL_PNL['regr_angles'] = False
        self.CTRL_PNL['aws'] = False
        self.CTRL_PNL['depth_map_labels'] = True #can only be true if we have 100% synthetic data for training
        self.CTRL_PNL['depth_map_labels_test'] = True #can only be true is we have 100% synth for testing
        self.CTRL_PNL['depth_map_output'] = self.CTRL_PNL['depth_map_labels']
        self.CTRL_PNL['depth_map_input_est'] = False  #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = self.CTRL_PNL['depth_map_input_est'] #holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True
        self.CTRL_PNL['normalize_input'] = False
        self.CTRL_PNL['all_tanh_activ'] = True
        self.CTRL_PNL['L2_contact'] = True
        self.CTRL_PNL['pmat_mult'] = int(1)
        self.CTRL_PNL['cal_noise'] = False
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['first_pass'] = True


        self.weight_joints = 1.0#self.opt.j_d_ratio*2
        self.weight_depth_planes = (1-0.5)#*2

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
                                             1./pmat_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat
                                             1./sobel_std_from_mult[int(self.CTRL_PNL['pmat_mult'])], #pmat sobel
                                             1./1.0,                #bed height mat
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1./1.0,                #OUTPUT DO NOTHING
                                             1. / 30.216647403350,  #weight
                                             1. / 14.629298141231]  #height


        self.CTRL_PNL['filepath_prefix'] = '/home/henry/'
            #self.CTRL_PNL['filepath_prefix'] = '/media/henry/multimodal_data_2/'

        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)





        #################################### PREP TRAINING DATA ##########################################

        dat_f_synth = TensorPrepLib().load_files_to_database(training_database_file_f, creation_type = 'synth', reduce_data = False)
        dat_m_synth = TensorPrepLib().load_files_to_database(training_database_file_m, creation_type = 'synth', reduce_data = False)


        self.train_x_flat = []  # Initialize the testing pressure mat list
        self.train_x_flat = TensorPrepLib().prep_images(self.train_x_flat, dat_f_synth, dat_m_synth, num_repeats = 1)
        self.train_x_flat = list(np.clip(np.array(self.train_x_flat) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100))

        #if self.CTRL_PNL['cal_noise'] == False:
        #    self.train_x_flat = PreprocessingLib().preprocessing_blur_images(self.train_x_flat, self.mat_size, sigma=0.5)

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


        # normalize the height and weight
        if self.CTRL_PNL['normalize_input'] == True:
            train_y_flat = TensorPrepLib().normalize_wt_ht(train_y_flat, self.CTRL_PNL)

        self.train_y_tensor = torch.Tensor(train_y_flat)

        print self.train_x_tensor.shape, 'Input training tensor shape'
        print self.train_y_tensor.shape, 'Output training tensor shape'


        self.train_dataset = torch.utils.data.TensorDataset(self.train_x_tensor, self.train_y_tensor)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])


    def val_convnet_general(self):

        self.m = load_model(model_path)

        self.pyRender = libPyRender.pyRenderMesh(render = True)


        # This will loop a total = training_images/batch_size times
        for batch_idx, batch in enumerate(self.train_loader):

            if batch_idx in [0, 1]: continue

            betas_gt = torch.mean(batch[1][:, 72:82], dim = 0).numpy()
            angles_gt = torch.mean(batch[1][:, 82:154], dim = 0).numpy()
            root_shift_est_gt = torch.mean(batch[1][:, 154:157], dim = 0).numpy()



            pmat = batch[0][0, 1, :, :].clone().numpy()#* 11.70153502792190


            for beta in range(betas_gt.shape[0]):
                self.m.betas[beta] = betas_gt[beta]

            for angle in range(angles_gt.shape[0]):
                self.m.pose[angle] = angles_gt[angle]


            #ground truth human mesh vertices
            smpl_verts_gt = np.array(self.m.r)
            for s in range(root_shift_est_gt.shape[0]):
                smpl_verts_gt[:, s] += (root_shift_est_gt[s] - float(self.m.J_transformed[0, s]))
            smpl_verts_gt = np.concatenate(
                (smpl_verts_gt[:, 1:2] - 0.286 + 0.0143,
                 smpl_verts_gt[:, 0:1] - 0.286 + 0.0143,
                  0.0 - smpl_verts_gt[:, 2:3]), axis=1)

            #mesh faces
            smpl_faces = np.array(self.m.f)

            #this gets the joint cartesian positions. no code here to visualize.
            joint_cart_gt = np.array(self.m.J_transformed).reshape(24, 3)
            for s in range(root_shift_est_gt.shape[0]):
                joint_cart_gt[:, s] += (root_shift_est_gt[s] - float(self.m.J_transformed[0, s]))


            if opt.red_verts == False:
            #this code renders the whole ground truth mesh.
                self.pyRender.render_3D_data(camera_point = None, pmat = pmat, smpl_verts_gt = smpl_verts_gt,
                                             smpl_faces = smpl_faces, segment_limbs = opt.seg_limbs)
            else:
                #this code renders only camera-facing vertices in the mesh
                camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]
                self.pyRender.render_3D_data(camera_point=camera_point, pmat=pmat, smpl_verts_gt=smpl_verts_gt,
                                             smpl_faces=smpl_faces, segment_limbs = opt.seg_limbs)

            time.sleep(1000)

if __name__ == "__main__":

    import optparse

    p = optparse.OptionParser()
    p.add_option('--red', action='store_true', dest='red_verts', default=False,
                 help='Do a quick test.')
    p.add_option('--seg', action='store_true', dest='seg_limbs', default=False,
                 help='Do a quick test.')

    opt, args = p.parse_args()



    filepath_prefix = '/media/henry/multimodal_data_2/data_BR/synth/'
    GENDER = "m"

    #Replace this with some subset of data of your choice
    TESTING_FILENAME = "general_supine/test_roll0_plo_"+GENDER+"_lay_set14_1500"


    test_database_file_f = []
    test_database_file_m = []

    if GENDER == "f":
        test_database_file_f.append(filepath_prefix+TESTING_FILENAME+'.p')
        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    else:
        test_database_file_m.append(filepath_prefix+TESTING_FILENAME+'.p')
        model_path = '/home/henry/git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'


    p = PhysicalTrainer(test_database_file_f, test_database_file_m)

    p.val_convnet_general()
