#!/usr/bin/env python

#Bodies at Rest: Code to visualize real dataset.
#(c) Henry M. Clever
#Major updates made for CVPR release: December 10, 2019


import numpy as np
import random
import copy

import cPickle as pkl

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
import sys
sys.path.insert(0, '../lib_py')

import optparse

import lib_pyrender_br as libPyRender
from visualization_lib_br import VisualizationLib
from preprocessing_lib_br import PreprocessingLib
from tensorprep_lib_br import TensorPrepLib
from unpack_batch_lib_br import UnpackBatchLib


from smpl.smpl_webuser.serialization import load_model

import os



#volumetric pose gen libraries
from time import sleep
from scipy.stats import mode
import os.path as osp
import imutils
from scipy.ndimage.filters import gaussian_filter


import matplotlib.cm as cm #use cm.jet(list)

DATASET_CREATE_TYPE = 1

import cv2

import math
from random import shuffle
import torch
import torch.nn as nn

import cPickle as pickle
VERT_CUT, HORIZ_CUT = 0, 50
pre_VERT_CUT = 40
DROPOUT = False

#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

MAT_SIZE = (64, 27)


CAM_BED_DIST = 1.66

torch.set_num_threads(1)
if torch.cuda.is_available():
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


class Viz3DPose():
    def __init__(self, opt):

        if opt.viz == '3D':
            self.pyRender = libPyRender.pyRenderMesh(render = True)
        else:
            self.pyRender = libPyRender.pyRenderMesh(render = False)


        ##load participant info
        participant_info = load_pickle(FILEPATH_PREFIX+"/real/"+PARTICIPANT+"/participant_info_red.p")
        for entry in participant_info:
            print entry, participant_info[entry]

        self.gender = participant_info['gender']
        self.height_in = participant_info['height_in']
        self.weight_lbs = participant_info['weight_lbs']

        self.opt = opt

        self.index_queue = []
        if self.gender == "m":
            model_path = '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        else:
            model_path = '../../../git/SMPL_python_v.1.0.0/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'

        self.reset_pose = False
        self.m = load_model(model_path)

        self.pressure = None

        self.CTRL_PNL = {}
        self.CTRL_PNL['batch_size'] = 1
        self.CTRL_PNL['loss_vector_type'] = 'anglesDC'
        self.CTRL_PNL['verbose'] = False
        self.CTRL_PNL['num_epochs'] = 101
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = opt.htwt
        self.CTRL_PNL['loss_root'] = opt.loss_root
        self.CTRL_PNL['omit_cntct_sobel'] = opt.omit_cntct_sobel
        self.CTRL_PNL['use_hover'] = opt.use_hover
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['num_input_channels'] = 2
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['repeat_real_data_ct'] = 1
        self.CTRL_PNL['regr_angles'] = 1
        self.CTRL_PNL['dropout'] = DROPOUT
        self.CTRL_PNL['depth_map_labels'] = False
        self.CTRL_PNL['depth_map_output'] = True
        self.CTRL_PNL['depth_map_input_est'] = False#opt.pmr  # rue #do this if we're working in a two-part regression
        self.CTRL_PNL['adjust_ang_from_est'] = False#self.CTRL_PNL['depth_map_input_est']  # holds betas and root same as prior estimate
        self.CTRL_PNL['clip_sobel'] = True
        self.CTRL_PNL['clip_betas'] = True
        self.CTRL_PNL['mesh_bottom_dist'] = True
        self.CTRL_PNL['full_body_rot'] = True  # False
        self.CTRL_PNL['normalize_per_image'] = True
        if self.CTRL_PNL['normalize_per_image'] == False:
            self.CTRL_PNL['normalize_std'] = True
        else:
            self.CTRL_PNL['normalize_std'] = False
        self.CTRL_PNL['all_tanh_activ'] = True  # False
        self.CTRL_PNL['L2_contact'] = True  # False
        self.CTRL_PNL['pmat_mult'] = int(1)
        self.CTRL_PNL['cal_noise'] = opt.calnoise
        self.CTRL_PNL['cal_noise_amt'] = 0.1
        self.CTRL_PNL['output_only_prev_est'] = False
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['first_pass'] = True
        self.CTRL_PNL['align_procr'] = False

        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL[
                'incl_pmat_cntct_input'] = False  # if there's calibration noise we need to recompute this every batch
            self.CTRL_PNL['clip_sobel'] = False

        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            self.CTRL_PNL['num_input_channels'] += 1
        if self.CTRL_PNL['depth_map_input_est'] == True:  # for a two part regression
            self.CTRL_PNL['num_input_channels'] += 3
        self.CTRL_PNL['num_input_channels_batch0'] = np.copy(self.CTRL_PNL['num_input_channels'])
        if self.CTRL_PNL['incl_ht_wt_channels'] == True:
            self.CTRL_PNL['num_input_channels'] += 2
        if self.CTRL_PNL['cal_noise'] == True:
            self.CTRL_PNL['num_input_channels'] += 1

        pmat_std_from_mult = ['N/A', 11.70153502792190, 19.90905848383454, 23.07018866032369, 0.0, 25.50538629767412]
        if self.CTRL_PNL['cal_noise'] == False:
            sobel_std_from_mult = ['N/A', 29.80360490415032, 33.33532963163579, 34.14427844692501, 0.0,
                                   34.86393494050921]
        else:
            sobel_std_from_mult = ['N/A', 45.61635847182483, 77.74920396659292, 88.89398421073700, 0.0,
                                   97.90075708182506]

        self.CTRL_PNL['norm_std_coeffs'] = [1. / 41.80684362163343,  # contact
                                            1. / 16.69545796387731,  # pos est depth
                                            1. / 45.08513083167194,  # neg est depth
                                            1. / 43.55800622930469,  # cm est
                                            1. / pmat_std_from_mult[int(self.CTRL_PNL['pmat_mult'])],  # pmat x5
                                            1. / sobel_std_from_mult[int(self.CTRL_PNL['pmat_mult'])],  # pmat sobel
                                            1. / 1.0,  # bed height mat
                                            1. / 1.0,  # OUTPUT DO NOTHING
                                            1. / 1.0,  # OUTPUT DO NOTHING
                                            1. / 30.216647403350,  # weight
                                            1. / 14.629298141231]  # height

        if self.CTRL_PNL['normalize_std'] == False:
            for i in range(10):
                self.CTRL_PNL['norm_std_coeffs'][i] *= 0.
                self.CTRL_PNL['norm_std_coeffs'][i] += 1.


        if self.CTRL_PNL['depth_map_output'] == True:  # we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"

        self.TPL = TensorPrepLib()

        self.count = 0

        self.CTRL_PNL['filepath_prefix'] = '/home/henry/'
        self.CTRL_PNL['aws'] = False
        self.CTRL_PNL['lock_root'] = False

        self.color, self.depth_r, self.pressure = 0, 0, 0

        self.kinect_im_size = (960, 540)

        self.final_dataset = {}





    def load_new_participant_info(self, participant_directory):
        ##load participant info
        participant_info = load_pickle(participant_directory+"/participant_info_red.p")
        print "participant directory: ", participant_directory
        for entry in participant_info:
            print entry, participant_info[entry]

        self.gender = participant_info['gender']
        self.height_in = participant_info['height_in']
        self.weight_lbs = participant_info['weight_lbs']



    def depth_image(self, depth_r_orig):
        # DEPTH'
        depth_r_reshaped = depth_r_orig / 3 - 300


        depth_r_reshaped = np.clip(depth_r_reshaped, 0, 255)
        depth_r_reshaped = depth_r_reshaped.astype(np.uint8)
        depth_r_reshaped = np.stack((depth_r_reshaped,) * 3, -1)

        depth_r_reshaped = np.rot90(depth_r_reshaped)
        depth_r_orig = np.rot90(depth_r_orig)
        return depth_r_reshaped, depth_r_reshaped.shape, depth_r_orig


    def pressure_image(self, pressure_orig, color_size, tf_corners):

        start_low_pt = [int((tf_corners[0, 0] + tf_corners[2, 0] + 1) / 2),
                        int((880-tf_corners[2, 1] + 880-tf_corners[3, 1] + 1) / 2)]
        start_high_pt = [int((tf_corners[1, 0] + tf_corners[3, 0] + 1) / 2),
                        int((880-tf_corners[0, 1] + 880-tf_corners[1, 1] + 1) / 2)]

        pressure_im_size_required = [start_high_pt[1] - start_low_pt[1], start_high_pt[0] - start_low_pt[0]]


        # PRESSURE
        pressure_reshaped_temp = np.reshape(pressure_orig, MAT_SIZE)
        #pressure_reshaped_temp = np.flipud(np.fliplr(pressure_reshaped_temp))
        pressure_reshaped = cm.jet(1-pressure_reshaped_temp/100)[:, :, 0:3]
        pressure_reshaped = (pressure_reshaped * 255).astype(np.uint8)
        pressure_reshaped = cv2.resize(pressure_reshaped, (pressure_im_size_required[1], pressure_im_size_required[0])).astype(np.uint8)
        pressure_reshaped = np.rot90(pressure_reshaped, 3)
        pressure_reshaped_temp2 = np.zeros((color_size[1], color_size[0], color_size[2])).astype(np.uint8)
        pressure_reshaped_temp2[:, :, 0] = 50

        pmat_reshaped_size = pressure_reshaped.shape

        pressure_reshaped_temp2[start_low_pt[0]:start_low_pt[0]+pmat_reshaped_size[0], \
                                start_low_pt[1]:start_low_pt[1]+pmat_reshaped_size[1], :] = pressure_reshaped

        #pressure_reshaped_temp2[start_low_pt[0]-2:start_low_pt[0]+2,start_low_pt[1]-2:start_low_pt[1]+2,: ] = 255

        pressure_reshaped = pressure_reshaped_temp2

        pressure_reshaped = np.rot90(pressure_reshaped)
        return pressure_reshaped, pressure_reshaped.shape



    def evaluate_data(self, dat, model, model2):


        self.RESULTS_DICT = {}
        self.RESULTS_DICT['body_roll_rad'] = []
        self.RESULTS_DICT['v_to_gt_err'] = []
        self.RESULTS_DICT['v_limb_to_gt_err'] = []
        self.RESULTS_DICT['gt_to_v_err'] = []
        self.RESULTS_DICT['precision'] = []
        self.RESULTS_DICT['recall'] = []
        self.RESULTS_DICT['overlap_d_err'] = []
        self.RESULTS_DICT['all_d_err'] = []
        self.RESULTS_DICT['betas'] = []


        #for im_num in range(29, 100):
        for im_num in range(0, len(dat['images'])):#self.color_all.shape[0]):


            pmat_corners = dat['pmat_corners'][im_num]
            rgb = dat['RGB'][im_num]
            print "Pose type: ", dat['pose_type'][im_num]

            rgb[int(pmat_corners[0][1]+0.5)-2:int(pmat_corners[0][1]+0.5)+2, \
                int(pmat_corners[0][0]+0.5)-2:int(pmat_corners[0][0]+0.5)+2, :] = 0
            rgb[int(pmat_corners[1][1]+0.5)-2:int(pmat_corners[1][1]+0.5)+2, \
                int(pmat_corners[1][0]+0.5)-2:int(pmat_corners[1][0]+0.5)+2, :] = 0
            rgb[int(pmat_corners[2][1]+0.5)-2:int(pmat_corners[2][1]+0.5)+2, \
                int(pmat_corners[2][0]+0.5)-2:int(pmat_corners[2][0]+0.5)+2, :] = 0
            rgb[int(pmat_corners[3][1]+0.5)-2:int(pmat_corners[3][1]+0.5)+2, \
                int(pmat_corners[3][0]+0.5)-2:int(pmat_corners[3][0]+0.5)+2, :] = 0



            #PRESSURE
            self.pressure = dat['images'][im_num]


            #because we used a sheet on the bed the overall pressure is lower than calibration, which was done without a sheet. bump it up here.
            bedsheet_norm_factor = float(1)


            self.pressure = np.clip(self.pressure*bedsheet_norm_factor, 0, 100)



            #now do 3D rendering
            pmat = np.clip(self.pressure.reshape(MAT_SIZE), a_min=0, a_max=100)

            pc_autofil_red = dat['pc'][im_num]

            camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST] #[dist from foot of bed, dist from left side of mat, dist normal]

            #
            self.estimate_pose(pmat, pc_autofil_red, model, model2)

            #self.pyRender.render_3D_data(camera_point, pmat = pmat, pc = pc_autofil_red)

            self.point_cloud_array = None
            #sleep(100)



        dir = FILEPATH_PREFIX + '/final_results/'+NETWORK_2
        if not os.path.exists(dir):
            os.mkdir(dir)

        pkl.dump(self.RESULTS_DICT, open(dir+'/results_real_'+PARTICIPANT+'_'+POSE_TYPE+'_'+NETWORK_2+'.p', 'wb'))




    def estimate_pose(self, pmat, pc_autofil_red, model, model2):

        bedangle = 0

        mat_size = (64, 27)

        #pmat = np.fliplr(np.flipud(np.clip(pmat.reshape(MAT_SIZE) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100)))
        pmat = np.clip(pmat.reshape(MAT_SIZE) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100)

        if self.CTRL_PNL['cal_noise'] == False:
            pmat = gaussian_filter(pmat, sigma=0.5)

        pmat_stack = PreprocessingLib().preprocessing_create_pressure_angle_stack([pmat], mat_size, self.CTRL_PNL)[0]

        if self.CTRL_PNL['cal_noise'] == False and self.CTRL_PNL['normalize_per_image'] == False:
            pmat_stack = np.clip(pmat_stack, a_min=0, a_max=100)

        pmat_stack = np.expand_dims(np.array(pmat_stack), 0)
        print pmat_stack.shape

        if self.CTRL_PNL['incl_pmat_cntct_input'] == True:
            pmat_contact = np.copy(pmat_stack[:, 0:1, :, :])
            pmat_contact[pmat_contact > 0] = 100
            pmat_stack = np.concatenate((pmat_contact, pmat_stack), axis=1)

        weight_input = self.weight_lbs / 2.20462
        height_input = (self.height_in * 0.0254 - 1) * 100

        batch1 = np.zeros((1, 162))
        if self.gender == 'f':
            batch1[:, 157] += 1
        elif self.gender == 'm':
            batch1[:, 158] += 1
        batch1[:, 160] += weight_input
        batch1[:, 161] += height_input

        if self.CTRL_PNL['normalize_std'] == True:
            self.CTRL_PNL['depth_map_input_est'] = False
            pmat_stack = self.TPL.normalize_network_input(pmat_stack, self.CTRL_PNL)
            batch1 = self.TPL.normalize_wt_ht(batch1, self.CTRL_PNL)

        pmat_stack = torch.Tensor(pmat_stack)
        batch1 = torch.Tensor(batch1)

        if DROPOUT == True:
            pmat_stack = pmat_stack.repeat(25, 1, 1, 1)
            batch1 = batch1.repeat(25, 1)

        batch = []
        batch.append(pmat_stack)
        batch.append(batch1)

        NUMOFOUTPUTDIMS = 3
        NUMOFOUTPUTNODES_TRAIN = 24
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)

        self.CTRL_PNL['adjust_ang_from_est'] = False
        scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpack_batch(batch, is_training=False, model=model,
                                                                                    CTRL_PNL = self.CTRL_PNL)


        mdm_est_pos = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)# / 16.69545796387731
        mdm_est_neg = OUTPUT_DICT['batch_mdm_est'].clone().unsqueeze(1)# / 45.08513083167194
        mdm_est_pos[mdm_est_pos < 0] = 0
        mdm_est_neg[mdm_est_neg > 0] = 0
        mdm_est_neg *= -1
        cm_est = OUTPUT_DICT['batch_cm_est'].clone().unsqueeze(1) * 100# / 43.55800622930469

        # 1. / 16.69545796387731,  # pos est depth
        # 1. / 45.08513083167194,  # neg est depth
        # 1. / 43.55800622930469,  # cm est

        sc_sample1 = OUTPUT_DICT['batch_targets_est'].clone()
        sc_sample1 = sc_sample1[0, :].squeeze() / 1000
        sc_sample1 = sc_sample1.view(self.output_size_train)
        # print sc_sample1

        if model2 is not None:
            print "Using model 2"
            batch_cor = []

            if self.CTRL_PNL['cal_noise'] == False:
                batch_cor.append(torch.cat((pmat_stack[:, 0:1, :, :],
                                            mdm_est_pos.type(torch.FloatTensor),
                                            mdm_est_neg.type(torch.FloatTensor),
                                            cm_est.type(torch.FloatTensor),
                                            pmat_stack[:, 1:, :, :]), dim=1))
            else:
                if self.opt.pmr == True:
                    batch_cor.append(torch.cat((mdm_est_pos.type(torch.FloatTensor),
                                                mdm_est_neg.type(torch.FloatTensor),
                                                cm_est.type(torch.FloatTensor),
                                                pmat_stack[:, 0:, :, :]), dim=1))
                else:
                    batch_cor.append(pmat_stack)


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

            scores, INPUT_DICT, OUTPUT_DICT = UnpackBatchLib().unpack_batch(batch_cor, is_training=False, model=model2,
                                                                                        CTRL_PNL = self.CTRL_PNL)
            if self.opt.pmr == True:
                self.CTRL_PNL['num_input_channels_batch0'] -= 3

        self.CTRL_PNL['first_pass'] = False

        # print betas_est, root_shift_est, angles_est
        if self.CTRL_PNL['dropout'] == True:
            # print OUTPUT_DICT['verts'].shape
            smpl_verts = np.mean(OUTPUT_DICT['verts'], axis=0)
            dropout_variance = np.std(OUTPUT_DICT['verts'], axis=0)
            dropout_variance = np.linalg.norm(dropout_variance, axis=1)
        else:
            smpl_verts = OUTPUT_DICT['verts'][0, :, :]
            dropout_variance = None

        self.RESULTS_DICT['betas'].append(OUTPUT_DICT['batch_betas_est_post_clip'].cpu().numpy()[0])

        smpl_verts = np.concatenate(
            (smpl_verts[:, 1:2] - 0.286 + 0.0143, smpl_verts[:, 0:1] - 0.286 + 0.0143, 0.0 - smpl_verts[:, 2:3]),
            axis=1)

        smpl_faces = np.array(self.m.f)

        q = OUTPUT_DICT['batch_mdm_est'].data.numpy().reshape(OUTPUT_DICT['batch_mdm_est'].size()[0], 64, 27) * -1
        q = np.mean(q, axis=0)

        camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]

        SHOW_SMPL_EST = True
        if SHOW_SMPL_EST == False:
            smpl_verts *= 0.001

        # print smpl_verts

        if self.opt.viz == '2D':
            viz_type = "2D"
        else:
            viz_type = "3D"

        self.RESULTS_DICT['body_roll_rad'].append(float(OUTPUT_DICT['batch_angles_est'][0, 1]))

        if viz_type == "2D":
            im_display_idx = 0

            if model2 is not None:
                self.cntct_in = INPUT_DICT['batch_images'][im_display_idx, 0, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][0]  #contact
                self.pmap_recon_in = INPUT_DICT['batch_images'][im_display_idx, 2, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][2] #pmat
                self.cntct_recon_in = INPUT_DICT['batch_images'][im_display_idx, 3, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][3] #pmat
                self.hover_recon_in = INPUT_DICT['batch_images'][im_display_idx, 1, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][1] #pmat
                self.pimage_in = INPUT_DICT['batch_images'][im_display_idx, 4, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][4] #pmat
                self.sobel_in = INPUT_DICT['batch_images'][im_display_idx, 5, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][5]  #sobel
            else:
                self.cntct_in = INPUT_DICT['batch_images'][im_display_idx, 0, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][0]  #contact
                self.pmap_recon_in = None
                self.cntct_recon_in = None
                self.hover_recon_in = None
                self.pimage_in = INPUT_DICT['batch_images'][im_display_idx, 1, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][1]  #pmat
                self.sobel_in = INPUT_DICT['batch_images'][im_display_idx, 2, :].squeeze().cpu()/self.CTRL_PNL['norm_std_coeffs'][2]  #sobel

            self.pmap_recon = (OUTPUT_DICT['batch_mdm_est'][im_display_idx, :, :].squeeze() * -1).cpu().data  # est depth output
            self.cntct_recon = (OUTPUT_DICT['batch_cm_est'][im_display_idx, :, :].squeeze()).cpu().data  # est depth output
            self.hover_recon = (OUTPUT_DICT['batch_mdm_est'][im_display_idx, :, :].squeeze()).cpu().data  # est depth output


            self.tar_sample = INPUT_DICT['batch_targets']
            self.tar_sample = self.tar_sample[0, :].squeeze() / 1000
            sc_sample = OUTPUT_DICT['batch_targets_est'].clone()
            sc_sample = sc_sample[0, :].squeeze() / 1000

            sc_sample = sc_sample.view(self.output_size_train)


            VisualizationLib().visualize_pressure_map(pimage_in = self.pimage_in, cntct_in = self.cntct_in, sobel_in = self.sobel_in,
                                                      targets_raw = self.tar_sample.cpu(), scores_net1 = sc_sample.cpu(),
                                                      pmap_recon_in = self.pmap_recon_in, cntct_recon_in = self.cntct_recon_in,
                                                      hover_recon_in = self.hover_recon_in,
                                                      pmap_recon = self.pmap_recon, cntct_recon = self.cntct_recon, hover_recon = self.hover_recon,
                                                      block=False)



        elif viz_type == "3D":

            print np.min(smpl_verts[:, 0]), np.max(smpl_verts[:, 0])
            print np.min(smpl_verts[:, 1]), np.max(smpl_verts[:, 1])
            print np.min(smpl_verts[:, 2]), np.max(smpl_verts[:, 2])

            # render everything
            self.RESULTS_DICT = self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts, smpl_faces,
                                                                                     camera_point,
                                                                                     bedangle, self.RESULTS_DICT,
                                                                                     pc=pc_autofil_red, pmat=pmat,
                                                                                     smpl_render_points=False,
                                                                                     markers=[[0.0, 0.0, 0.0],
                                                                                              [0.0, 1.5, 0.0],
                                                                                              [0.0, 0.0, 0.0],
                                                                                              [0.0, 0.0, 0.0]],
                                                                                     dropout_variance=dropout_variance)

            time.sleep(1)
            self.point_cloud_array = None


        #time.sleep(100)

if __name__ ==  "__main__":

    import optparse

    p = optparse.OptionParser()

    p.add_option('--hd', action='store_true', dest='hd', default=False,
                 help='Read and write to data on an external harddrive.')

    #p.add_option('--net', action='store', type = 'int', dest='net', default=0,
    #             help='Choose a network.')

    p.add_option('--pose_type', action='store', type='string', dest='pose_type', default='none',
                 help='Choose a pose type, either `prescribed` or `p_select`.')

    p.add_option('--p_idx', action='store', type='int', dest='p_idx', default=0,
                 # PMR parameter to adjust loss function 2
                 help='Choose a participant. Enter a number from 1 to 20.')

    p.add_option('--pmr', action='store_true', dest='pmr', default=False,
                 help='Run PMR on input plus precomputed spatial maps.')

    p.add_option('--small', action='store_true', dest='small', default=False,
                 help='Make the dataset 1/4th of the original size.')

    p.add_option('--htwt', action='store_true', dest='htwt', default=False,
                 help='Include height and weight info on the input.')

    p.add_option('--calnoise', action='store_true', dest='calnoise', default=False,
                 help='Apply calibration noise to the input to facilitate sim to real transfer.')

    p.add_option('--viz', action='store', dest='viz', default='None',
                 help='Visualize training. specify `2D` or `3D`.')

    p.add_option('--go200', action='store_true', dest='go200', default=False,
                 help='Run network 1 for 100 to 200 epochs.')

    p.add_option('--loss_root', action='store_true', dest='loss_root', default=False,
                 help='Use root in loss function.')

    p.add_option('--use_hover', action='store_true', dest='use_hover', default=False,
                 help='Use hovermap for pmr input.')

    p.add_option('--omit_cntct_sobel', action='store_true', dest='omit_cntct_sobel', default=False,
                 help='Cut contact and sobel from input.')

    p.add_option('--half_shape_wt', action='store_true', dest='half_shape_wt', default=False,
                 help='Half betas.')



    opt, args = p.parse_args()

    participant_list = ["S103",
                        "S104",
                        "S107",
                        "S114",
                        "S118",
                        "S121",
                        "S130",
                        "S134",
                        "S140",
                        "S141",

                        "S145",
                        "S151",
                        "S163",
                        "S165",
                        "S170",
                        "S179",
                        "S184",
                        "S187",
                        "S188",
                        "S196", ]

    if opt.p_idx != 0:
        participant_list = [participant_list[opt.p_idx - 1]]

    for PARTICIPANT in participant_list:

        if opt.hd == False:
            FILEPATH_PREFIX = "../../../data_BR"
        else:
            FILEPATH_PREFIX = "/media/henry/multimodal_data_2/data_BR"

        participant_directory = FILEPATH_PREFIX + "/real/" + PARTICIPANT
        #participant_directory = "/media/henry/multimodal_data_2/data_BR/real/"+PARTICIPANT
        #participant_directory = "/home/henry/Desktop/CVPR2020_study/"+PARTICIPANT
        V3D = Viz3DPose(opt)

        V3D.load_new_participant_info(participant_directory)

        if opt.pose_type == "prescribed":
            dat = load_pickle(participant_directory+"/prescribed.p")
            POSE_TYPE = "2"
        elif opt.pose_type == "p_select":
            dat = load_pickle(participant_directory+"/p_select.p")
            POSE_TYPE = "1"
        else:
            print "Please choose a pose type - either prescribed poses, " \
                  "'--pose_type prescribed', or participant selected poses, '--pose_type p_select'."
            sys.exit()





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


        if opt.go200 == False:
            filename1 = FILEPATH_PREFIX+"/convnets_camready/convnet_1_anglesDC_" + NETWORK_1 + "_100e_2e-05lr.pt"
            filename2 = FILEPATH_PREFIX+"/convnets_camready/convnet_2_anglesDC_" + NETWORK_2 + "_100e_2e-05lr.pt"
        else:
            filename1 = FILEPATH_PREFIX+"/convnets_camready/convnet_1_anglesDC_" + NETWORK_1 + "_200e_2e-05lr.pt"
            filename2 = None

        if GPU == True:
            for i in range(0, 8):
                try:
                    model = torch.load(filename1, map_location={'cuda:' + str(i): 'cuda:0'})
                    model = model.cuda().eval()
                    print "Network 1 loaded."
                    break
                except:
                    pass
            if filename2 is not None:
                for i in range(0, 8):
                    try:
                        model2 = torch.load(filename2, map_location={'cuda:' + str(i): 'cuda:0'})
                        model2 = model2.cuda().eval()
                        print "Network 2 loaded."
                        break
                    except:
                        pass
            else:
                model2 = None
        else:
            model = torch.load(filename1, map_location='cpu')
            model = model.eval()
            print "Network 1 loaded."
            if filename2 is not None:
                model2 = torch.load(filename2, map_location='cpu')
                model2 = model2.eval()
                print "Network 2 loaded."
            else:
                model2 = None


        F_eval = V3D.evaluate_data(dat, model, model2)



