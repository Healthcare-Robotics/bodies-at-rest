#!/usr/bin/env python

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
import lib_pyrender_br_savefig as libPyRender
from opendr.renderer import ColoredRenderer
from opendr.renderer import DepthRenderer
from opendr.lighting import LambertianPointLight
from opendr.camera import ProjectPoints
from smpl.smpl_webuser.serialization import load_model
from cv_bridge import CvBridge, CvBridgeError

from smpl.smpl_webuser.serialization import load_model as load_smpl_model

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# some_file.py

#volumetric pose gen libraries
from multipose_lib_br import ArTagLib
from multipose_lib_br import VizLib

from visualization_lib_br import VisualizationLib
from preprocessing_lib_br import PreprocessingLib
from tensorprep_lib_br import TensorPrepLib
from unpack_batch_lib_br import UnpackBatchLib

from time import sleep
import rospy
import roslib
from sensor_msgs.msg import PointCloud2
from hrl_msgs.msg import FloatArrayBare
from ar_track_alvar_msgs.msg import AlvarMarkers
import sensor_msgs.point_cloud2
from scipy.stats import mode
import os.path as osp
import imutils

from scipy.ndimage.filters import gaussian_filter
DATASET_CREATE_TYPE = 1

import cv2
from camera import Camera

import math
from random import shuffle
import torch
import torch.nn as nn

import tensorflow as tensorflow
import cPickle as pickle
VERT_CUT, HORIZ_CUT = 0, 50
pre_VERT_CUT = 40


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL

SHOW_SMPL_EST = True
#PARTICIPANT = "S196"#"S151"
MAT_SIZE = (64, 27)

PC_WRT_ARTAG_ADJ = [0.11, -0.02, 0.07]
ARTAG_WRT_PMAT = [0.08, 0.05, 0.0]

DROPOUT = False
CAM_BED_DIST = 1.66


import sys

sys.path.insert(0, '/home/henry/git/volumetric_pose_gen/convnets')
import convnet_br as convnet
from torch.autograd import Variable

if False:#torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
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
        participant_info = load_pickle("/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/participant_info.p")
        for entry in participant_info:
            print entry, participant_info[entry]

        self.gender = participant_info['gender']
        self.height_in = participant_info['height_in']
        self.weight_lbs = participant_info['weight_lbs']
        self.adj_2 = participant_info['adj_2']
        self.pose_type_list = participant_info['pose_type']
        self.calibration_optim_values = participant_info['cal_func']
        self.tf_corners = participant_info['corners']


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
        self.setup_allmodes()




    def load_new_participant_info(self, participant_directory):
        ##load participant info
        participant_info = load_pickle(participant_directory+"/participant_info_red.p")
        print "participant directory: ", participant_directory
        for entry in participant_info:
            print entry, participant_info[entry]

        self.gender = participant_info['gender']
        self.height_in = participant_info['height_in']
        self.weight_lbs = participant_info['weight_lbs']






    def setup_allmodes(self):

        self.reset_pose = False

        self.marker0, self.marker1, self.marker2, self.marker3 = None, None, None, None
        self.pressure = None
        self.markers = [self.marker0, self.marker1, self.marker2, self.marker3]
        self.point_cloud_array = np.array([[0., 0., 0.]])
        self.pc_isnew = False

        self.bridge = CvBridge()
        self.color, self.depth_r, self.pressure = 0, 0, 0

        self.kinect_im_size = (960, 540)
        self.pressure_im_size = (64, 27)
        self.pressure_im_size_required = (64, 27)

        # initialization of kinect and thermal cam calibrations from YAML files
        dist_model = 'rational_polynomial'
        self.kcam = Camera('kinect', self.kinect_im_size, dist_model)
        self.kcam.init_from_yaml(osp.expanduser('~/catkin_ws/src/multimodal_pose/calibrations/kinect.yaml'))

        # we are at qhd not hd so need to cut the focal lengths and centers in half
        self.kcam.K[0:2, 0:3] = self.kcam.K[0:2, 0:3] / 2

       # print self.kcam.K

        self.new_K_kin, roi = cv2.getOptimalNewCameraMatrix(self.kcam.K, self.kcam.D, self.kinect_im_size, 1,
                                                            self.kinect_im_size)

        #print self.new_K_kin

        self.drawing = False  # true if mouse is pressed
        self.mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
        self.ix, self.iy = -1, -1
        self.label_index = 0
        self.coords_from_top_left = [0, 0]
        self.overall_image_scale_amount = 0.85
        self.depthcam_midpixel = [0, 0]
        self.depthcam_midpixel2 = [0, 0]
        self.select_new_calib_corners = {}
        self.select_new_calib_corners["lay"] = True
        self.select_new_calib_corners["sit"] = True
        self.calib_corners = {}
        self.calib_corners["lay"] = 8 * [[0, 0]]
        self.calib_corners["sit"] = 8 * [[0, 0]]

        self.final_dataset = {}

        self.filler_taxels = []
        for i in range(28):
            for j in range(65):
                self.filler_taxels.append([i - 1, j - 1, 20000])
        self.filler_taxels = np.array(self.filler_taxels).astype(int)



    def load_next_file(self, newpath):

        print "loading existing npy files in the new path...."
        time_orig = time.time()
        self.color_all = np.load(newpath+"/color.npy")
        self.depth_r_all = np.load(newpath+"/depth_r.npy")
        self.pressure_all = np.load(newpath+"/pressure.npy")
        self.bedstate_all = np.load(newpath+"/bedstate.npy")
        self.markers_all = np.load(newpath+"/markers.npy", allow_pickle=True)
        self.time_stamp_all = np.load(newpath+"/time_stamp.npy")
        self.point_cloud_autofil_all = np.load(newpath+"/point_cloud.npy")
        #self.config_code_all = np.load(newpath+"/config_code.npy")
        print "Finished. Time taken: ", time.time() - time_orig



    def transform_selected_points(self, image, camera_alpha_vert, camera_alpha_horiz, angle, right, up, h_scale_cut, v_scale_cut, coords_subset):
        h_scale = h_scale_cut[0]
        h_cut = h_scale_cut[1]
        v_scale = v_scale_cut[0]
        v_cut = v_scale_cut[1]
        tf_coords_subset = np.copy(coords_subset)
        #print camera_alpha_vert, camera_alpha_horiz, HORIZ_CUT, VERT_CUT, pre_VERT_CUT, right

        h = VizLib().get_new_K_kin_homography(camera_alpha_vert, camera_alpha_horiz, self.new_K_kin, flip_vert=-1)

        for i in range(4):

            new_coords = np.matmul(h, np.array([tf_coords_subset[i, 1]+pre_VERT_CUT, tf_coords_subset[i, 0]+HORIZ_CUT, 1]))
            new_coords = new_coords/new_coords[2]
            tf_coords_subset[i, 0] = new_coords[1] - HORIZ_CUT
            tf_coords_subset[i, 1] = new_coords[0] - pre_VERT_CUT


            tf_coords_subset[i, 1] = (tf_coords_subset[i, 1] - image.shape[0] / 2) * np.cos(np.deg2rad(angle)) - (
                        tf_coords_subset[i, 0] - image.shape[1] / 2) * np.sin(np.deg2rad(angle)) + image.shape[
                                  0] / 2 - up
            tf_coords_subset[i, 0] = (tf_coords_subset[i, 1] - image.shape[0] / 2) * np.sin(np.deg2rad(angle)) + (
                        tf_coords_subset[i, 0] - image.shape[1] / 2) * np.cos(np.deg2rad(angle)) + image.shape[
                                  1] / 2 - right

            tf_coords_subset[i, 0] = h_scale * (tf_coords_subset[i][0] + h_cut) - h_cut
            tf_coords_subset[i, 1] = v_scale * (tf_coords_subset[i][1] + v_cut) - v_cut

            image[int(tf_coords_subset[i][1] + 0.5) - 2:int(tf_coords_subset[i][1] + 0.5) + 2,
            int(tf_coords_subset[i][0] + 0.5) - 2:int(tf_coords_subset[i][0] + 0.5) + 2, :] = 255


        return tf_coords_subset, image

    def rotate_selected_head_points(self, pressure_im_size_required, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend, u_p_bend_calib, v_p_bend_calib):

        low_vert = np.rint(v_c_pmat[2]).astype(np.uint16)
        low_horiz = np.rint(u_c_pmat[1]).astype(np.uint16)
        legs_bend_loc2 = pressure_im_size_required[0]*20/64 + low_horiz

        HEAD_BEND_TAXEL = 41  # measured from the bottom of the pressure mat
        LEGS_BEND2_TAXEL = 20 #measured from the bottom of the pressure mat
        head_bend_loc = pressure_im_size_required[0]*HEAD_BEND_TAXEL/64 + low_horiz

        head_points_L = [np.rint(v_p_bend_calib[0]).astype(np.uint16) - 3 - HORIZ_CUT + 4,
                         380-np.rint(u_p_bend_calib[0] - head_bend_loc - 3).astype(np.uint16) - pre_VERT_CUT + 4]  # np.copy([head_points1[2][0] - decrease_from_orig_len, head_points1[2][1] - increase_across_pmat])
        head_points_R = [np.rint(v_p_bend_calib[1]).astype(np.uint16) + 4 - HORIZ_CUT - 4,
                         380-np.rint(u_p_bend_calib[1] - head_bend_loc - 3).astype(np.uint16) - pre_VERT_CUT + 4]  # np.copy([head_points1[3][0] - decrease_from_orig_len, head_points1[3][1] + increase_across_pmat])
        legs_points_pre = [pressure_im_size_required[0] * 64 / 64 - pressure_im_size_required[0] * (64 - LEGS_BEND2_TAXEL) / 64, low_vert]  # happens at legs bend2


        legs_points_L = [np.rint(v_p_bend[4]).astype(np.uint16) - 3 - HORIZ_CUT + 4,
                         head_bend_loc - pressure_im_size_required[0] * HEAD_BEND_TAXEL / 64 + 560]  # happens at legs bottom
        legs_points_R = [np.rint(v_p_bend[5]).astype(np.uint16) + 4 - HORIZ_CUT - 4,
                         head_bend_loc - pressure_im_size_required[0] * HEAD_BEND_TAXEL / 64 + 560]  # happens at legs bottom


        return [head_points_L, head_points_R, legs_points_L, legs_points_R]



    def get_pc_from_depthmap(self, bed_angle, zero_location):

       # print zero_location, 'zero loc'


        #transform 3D pc using homography!
        #bed_angle = 0.
        #x and y are pixel selections

        zero_location += 0.5
        zero_location = zero_location.astype(int)

        x = np.arange(0, 440).astype(float)
        x = np.tile(x, (880, 1))
        y = np.arange(0, 880).astype(float)
        y = np.tile(y, (440, 1)).T

        x_coord_from_camcenter = x - self.depthcam_midpixel[0]
        y_coord_from_camcenter = y - self.depthcam_midpixel[1]


        #here try transforming the 2D representation before we move on to 3D

        depth_value = self.depth_r_orig.astype(float) / 1000

        f_x, f_y, c_x, c_y = self.new_K_kin[0, 0], self.new_K_kin[1, 1], self.new_K_kin[0, 2], self.new_K_kin[1, 2]
        X = (x_coord_from_camcenter) * depth_value / f_y
        Y = (y_coord_from_camcenter) * depth_value / f_x

        x_coord_from_camcenter_single = zero_location[0] - self.depthcam_midpixel[0]
        y_coord_from_camcenter_single = zero_location[1] - self.depthcam_midpixel[1]
        X_single = (x_coord_from_camcenter_single) * CAM_BED_DIST / f_y
        Y_single = (y_coord_from_camcenter_single) * CAM_BED_DIST / f_x

        #print X_single, Y_single, 'Y single'
        X -= X_single
        Y -= (Y_single)

        Y = -Y
        Z = -depth_value + CAM_BED_DIST

        point_cloud = np.stack((Y, X, -Z))
        point_cloud = np.swapaxes(point_cloud, 0, 2)
        point_cloud = np.swapaxes(point_cloud, 0, 1)

        point_cloud_red = np.zeros((point_cloud.shape[0]/10, point_cloud.shape[1]/10, 3))
        for j in range(point_cloud_red.shape[0]):
            for i in range(point_cloud_red.shape[1]):
                point_cloud_red[j, i, :] = np.median(np.median(point_cloud[j*10:(j+1)*10, i*10:(i+1)*10, :], axis = 0), axis = 0)
        self.point_cloud_red = point_cloud_red.reshape(-1, 3)
        self.point_cloud = point_cloud.reshape(-1, 3)
        self.point_cloud[:, 0] += PC_WRT_ARTAG_ADJ[0] + ARTAG_WRT_PMAT[0]
        self.point_cloud[:, 1] += PC_WRT_ARTAG_ADJ[1] + ARTAG_WRT_PMAT[1]
        self.point_cloud[:, 2] += PC_WRT_ARTAG_ADJ[2] + ARTAG_WRT_PMAT[2]
        #print point_cloud.shape, 'pc shape'
        #print point_cloud_red.shape

        return X, Y, Z

    def trim_pc_sides(self, tf_corners, camera_alpha_vert, camera_alpha_horiz, h, kinect_rot_cw):

        f_x, f_y, c_x, c_y = self.new_K_kin[0, 0], self.new_K_kin[1, 1], self.new_K_kin[0, 2], self.new_K_kin[1, 2]
        #for i in range(3):
        #    print np.min(self.point_cloud_autofil[:, i]), np.max(self.point_cloud_autofil[:, i])


        self.point_cloud_autofil[:, 0] = self.point_cloud_autofil[:, 0]# - 0.17 - 0.036608



        #CALIBRATE THE POINT CLOUD HERE


        pc_autofil_red = np.copy(self.point_cloud_autofil)

        if pc_autofil_red.shape[0] == 0:
            pc_autofil_red = np.array([[0.0, 0.0, 0.0]])

        #warp it by the homography i.e. rotate a bit
        pc_autofil_red -=[0.0, 0.0, CAM_BED_DIST]

        theta_1 = np.arctan((camera_alpha_vert-1)*CAM_BED_DIST/(270*CAM_BED_DIST/f_y))/2 #short side
        short_side_rot = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta_1), -np.sin(theta_1)], [0.0, np.sin(theta_1), np.cos(theta_1)]])
        pc_autofil_red = np.matmul(pc_autofil_red, short_side_rot)#[0:3, :]

        theta_2 = np.arctan((1-camera_alpha_horiz)*CAM_BED_DIST/(270*CAM_BED_DIST/f_x))/2 #long side
        long_side_rot = np.array([[np.cos(theta_2), 0.0, np.sin(theta_2)], [0.0, 1.0, 0.0], [-np.sin(theta_2), 0.0, np.cos(theta_2)]])
        pc_autofil_red = np.matmul(pc_autofil_red, long_side_rot)#[0:3, :]

        pc_autofil_red +=[0.0, 0.0, CAM_BED_DIST]


        #add the warping translation
        X_single1 = h[0, 2] * CAM_BED_DIST / f_y
        Y_single1 = h[1, 2] * CAM_BED_DIST / f_x

        #print X_single1, Y_single1
        pc_autofil_red += [-Y_single1/2, -X_single1/2, 0.0]


        #rotate normal to the bed
        angle = kinect_rot_cw*np.pi/180.
        z_rot_mat = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0.0, 0.0, 1.0]])
        pc_autofil_red = np.matmul(pc_autofil_red, z_rot_mat)#[0:3, :]


        #translate by the picture shift amount in the x and y directions


        #print np.min(pc_autofil_red[:, 0]), np.max(pc_autofil_red[:, 0]), "Y min max"
        #print self.tf_corners[2], self.depthcam_midpixel2

        #translate from the 0,0 being the camera to 0,0 being the left corner of the bed measured by the clicked point
        zero_location = np.copy(self.tf_corners[2]) #TF corner needs to be manipulated!
        x_coord_from_camcenter_single = zero_location[0] - self.depthcam_midpixel2[0]
        y_coord_from_camcenter_single = zero_location[1] - self.depthcam_midpixel2[1]
        X_single2 = (x_coord_from_camcenter_single) * CAM_BED_DIST / f_y #shift dim
        Y_single2 = (y_coord_from_camcenter_single) * CAM_BED_DIST / f_x #long dim
        pc_autofil_red += [Y_single2, -X_single2, -CAM_BED_DIST]


        #adjust to fit to the lower left corner step 2
        pc_autofil_red += [self.adj_2[0], self.adj_2[1], 0.0]


        #pc_autofil_red = np.swapaxes(np.array(self.pc_all).reshape(3, 440*880), 0, 1)

        #print np.min(pc_autofil_red[:, 0]), np.max(pc_autofil_red[:, 0]), "Y min max"

        #cut off everything that's not overlying the bed.
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 1] > 0.0, :]
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 1] < 0.0286 * 27, :]

        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 0] > 0.0, :] #up and down bed
        pc_autofil_red = pc_autofil_red[pc_autofil_red[:, 0] < 0.0286 * 64 * 1.04, :] #up and down bed


        #adjust it by a half taxel width
        #pc_autofil_red += [0.0143, 0.0143, 0.0]


        return pc_autofil_red



    def estimate_pose(self, pmat, bedangle, markers_c, model, model2, tf_corners, camera_alpha_vert, camera_alpha_horiz, h, kinect_rot_cw, color_im):

        bedangle = 0

        mat_size = (64, 27)

        pmat = np.fliplr(np.flipud(np.clip(pmat.reshape(MAT_SIZE) * float(self.CTRL_PNL['pmat_mult']), a_min=0, a_max=100)))
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

        pmat_array_input = np.copy(pmat_stack)[0, 0, :, :]


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
            #print OUTPUT_DICT['verts'].shape
            smpl_verts = np.mean(OUTPUT_DICT['verts'], axis = 0)
            dropout_variance = np.std(OUTPUT_DICT['verts'], axis=0)
            dropout_variance = np.linalg.norm(dropout_variance, axis = 1)
        else:
            smpl_verts = OUTPUT_DICT['verts'][0, :, :]
            dropout_variance = None

        self.RESULTS_DICT['betas'].append(OUTPUT_DICT['batch_betas_est_post_clip'].cpu().numpy()[0])



        smpl_verts = np.concatenate((smpl_verts[:, 1:2] - 0.286 + 0.0143, smpl_verts[:, 0:1] - 0.286 + 0.0143, 0.0 -smpl_verts[:, 2:3]), axis = 1)
        smpl_faces = np.array(self.m.f)

        pc_autofil_red = self.trim_pc_sides(tf_corners, camera_alpha_vert, camera_alpha_horiz, h, kinect_rot_cw) #this is the point cloud


        q = OUTPUT_DICT['batch_mdm_est'].data.numpy().reshape(OUTPUT_DICT['batch_mdm_est'].size()[0], 64, 27) * -1
        q = np.mean(q, axis = 0)

        camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST]

        if SHOW_SMPL_EST == False:
            smpl_verts *= 0.001

        #print smpl_verts

        viz_type = "3D"

        self.RESULTS_DICT['body_roll_rad'].append(float(OUTPUT_DICT['batch_angles_est'][0, 1]))

        if viz_type == "2D":
            from visualization_lib import VisualizationLib
            if model2 is not None:
                self.im_sample = INPUT_DICT['batch_images'][0, 4:,:].squeeze() * 20.  # normalizing_std_constants[4]*5.  #pmat
            else:
                self.im_sample = INPUT_DICT['batch_images'][0, 1:,:].squeeze() * 20.  # normalizing_std_constants[4]*5.  #pmat
            self.im_sample_ext = INPUT_DICT['batch_images'][0, 0:, :].squeeze() * 20.  # normalizing_std_constants[0]  #pmat contact
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
                                                          self.im_sample_ext3, None, None, #, self.tar_sample_val, self.sc_sample_val,
                                                          block=False)

            time.sleep(4)

        elif viz_type == "3D":
            #render everything

            self.RESULTS_DICT = self.pyRender.render_mesh_pc_bed_pyrender_everything(smpl_verts, smpl_faces, camera_point,
                                                                  bedangle, self.RESULTS_DICT,
                                                                  pc = pc_autofil_red, pmat = pmat_array_input, smpl_render_points = False,
                                                                  markers = [[0.0, 0.0, 0.0],[0.0, 1.5, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0]],
                                                                                     dropout_variance = dropout_variance, color_im = color_im,
                                                                                     tf_corners=tf_corners, current_pose_type_ct=self.current_pose_type_ct,
                                                                                     participant = PARTICIPANT)






            time.sleep(1)
            self.point_cloud_array = None



    def evaluate_data(self, dat, filename1, filename2=None):

        self.pyRender = libPyRender.pyRenderMesh(render = True)

        #model = torch.load(filename1, map_location={'cuda:5': 'cuda:0'})
        if GPU == True:
            for i in range(0, 8):
                try:
                    model = torch.load(filename1, map_location={'cuda:'+str(i):'cuda:0'})
                    if self.CTRL_PNL['dropout'] == True:
                        model = model.cuda().train()
                    else:
                        model = model.cuda().eval()
                    break
                except:
                    pass
            if filename2 is not None:
                for i in range(0, 8):
                    try:
                        model2 = torch.load(filename2, map_location={'cuda:'+str(i):'cuda:0'})
                        if self.CTRL_PNL['dropout'] == True:
                            model2 = model2.cuda().train()
                        else:
                            model2 = model2.cuda().eval()
                        break
                    except:
                        pass
            else:
                model2 = None
        else:
            model = torch.load(filename1, map_location='cpu')
            if self.CTRL_PNL['dropout'] == True:
                model = model.train()
            else:
                model = model.eval()
            if filename2 is not None:
                model2 = torch.load(filename2, map_location='cpu')
                if self.CTRL_PNL['dropout'] == True:
                    model2 = model2.train()
                else:
                    model2 = model2.eval()
            else:
                model2 = None

        #function_input = np.array(function_input)*np.array([10, 10, 10, 10, 10, 10, 0.1, 0.1, 0.1, 0.1, 1])
        #function_input += np.array([2.2, 32, -1, 1.2, 32, -5, 1.0, 1.0, 0.96, 0.95, 0.8])
        function_input = np.array(self.calibration_optim_values)*np.array([10, 10, 10, 0.1, 0.1, 0.1, 0.1])
        function_input += np.array([1.2, 32, -5, 1.0, 1.0, 0.96, 0.95])


        kinect_rotate_angle = function_input[3-3]
        kinect_shift_up = int(function_input[4-3])# - 40
        kinect_shift_right = int(function_input[5-3])# - 20
        camera_alpha_vert = function_input[6-3]
        camera_alpha_horiz = function_input[7-3]
        pressure_horiz_scale = function_input[8-3]
        pressure_vert_scale = function_input[9-3]
        #head_angle_multiplier = function_input[10-3]

        #print kinect_shift_up, kinect_shift_right, "SHIFT UP RIGHT"
        #print pressure_horiz_scale, pressure_vert_scale, "PRESSURE SCALES" #1.04 for one too far to left

        #file_dir = "/media/henry/multimodal_data_1/all_hevans_data/0905_2_Evening/0255"
        #file_dir_list = ["/media/henry/multimodal_data_2/test_data/data_072019_0001/"]
        blah = True

        #file_dir = "/media/henry/multimodal_data_2/test_data/data_072019_0007"
        #file_dir = "/media/henry/multimodal_data_2/test_data/data_072019_0006"
        #file_dir = "/home/henry/ivy_test_data/data_102019_kneeup_0000"
        #file_dir = "/media/henry/multimodal_data_1/CVPR2020_study/P000/data_102019_kneeup_0000"

        if PARTICIPANT == "P106":
            #file_dir = "/media/henry/multimodal_data_1/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"_000"
            file_dir = "/home/henry/Desktop/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"_000"
            file_dirs = [#file_dir+str(0),
                         file_dir+str(1),
                         file_dir+str(2),
                         file_dir+str(3),
                         file_dir+str(4),
                         file_dir+str(5)]
        else:
            #file_dir = "/media/henry/multimodal_data_1/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"-2_000"
            file_dir = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_checked_"+PARTICIPANT+"-"+POSE_TYPE
            file_dirs = [file_dir]
            #file_dir = "/media/henry/multimodal_data_1/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"-2_000"
            #file_dir = "/media/henry/multimodal_data_2/CVPR2020_study/"+PARTICIPANT+"/data_"+PARTICIPANT+"-C_0000"
            #file_dirs = [file_dir]




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

        init_time = time.time()

        for file_dir in file_dirs:
            V3D.load_next_file(file_dir)

            start_num = 0
            #print self.color_all.shape

            #for im_num in range(29, 100):
            for im_num in range(start_num, self.color_all.shape[0]):

                #For P188: skip 5. 13 good cross legs

                print "NEXT IM!", im_num, " ", time.time() - init_time

                if PARTICIPANT == "S114" and POSE_TYPE == "2"  and im_num in [26, 29]: continue #these don't have point clouds
                if PARTICIPANT == "S165" and POSE_TYPE == "2" and im_num in [1, 3, 15]: continue #these don't have point clouds
                if PARTICIPANT == "S188" and POSE_TYPE == "2"  and im_num in [5, 17, 21]: continue

                if POSE_TYPE == "2":
                    im_num_ct = im_num + 1
                else:
                    im_num_ct = float(im_num)

                if POSE_TYPE == "1":
                    self.current_pose_type_ct = '{:02}'.format(im_num_ct)+'_natural_'+NETWORK_2

                elif POSE_TYPE == "2":
                    self.current_pose_type_ct = '{:02}'.format(im_num_ct)+'_'+self.pose_type_list[im_num]+'_'+NETWORK_2


                #good picks: 103 - 6 good for what info is there
                            #151 11 is  good
                            #179 - 7 is great
                            #187 natural poses very good
                            #196 - 11 has great smile :)

                self.overall_image_scale_amount = 0.85

                half_w_half_l = [0.4, 0.4, 1.1, 1.1]

                all_image_list = []
                self.label_single_image = []

                self.label_index = 0

                self.color = self.color_all[im_num]
                self.depth_r = self.depth_r_all[im_num]
                self.pressure = self.pressure_all[im_num]
                self.bed_state = self.bedstate_all[im_num]

                if self.point_cloud_autofil_all[im_num].shape[0] == 0:
                    self.point_cloud_autofil_all[im_num] = np.array([[0.0, 0.0, 0.0]])
                self.point_cloud_autofil = self.point_cloud_autofil_all[im_num] + self.markers_all[im_num][2]#[0.0, 0.0, 0.0]#0.1]
                #print self.markers_all[im_num]
                #print self.point_cloud_autofil.shape, 'PC AUTOFIL ORIG'



                self.bed_state[0] = self.bed_state[0]*0.0#*head_angle_multiplier
                self.bed_state *= 0
                #self.bed_state += 60.
                #print self.bed_state, np.shape(self.pressure)

                if im_num == start_num and blah == True:
                    markers_c = []
                    markers_c.append(self.markers_all[im_num][0])
                    markers_c.append(self.markers_all[im_num][1])
                    markers_c.append(self.markers_all[im_num][2])
                    markers_c.append(self.markers_all[im_num][3])
                    #for idx in range(4):
                        #if markers_c[idx] is not None:
                            #markers_c[idx] = np.array(markers_c[idx])*213./228.
                blah = False

                #print markers_c, 'Markers C'

                # Get the marker points in 2D on the color image
                u_c, v_c = ArTagLib().color_2D_markers(markers_c, self.new_K_kin)

                # Get the marker points dropped to the height of the pressure mat
                u_c_drop, v_c_drop, markers_c_drop = ArTagLib().color_2D_markers_drop(markers_c, self.new_K_kin)


                #print markers_c_drop, self.new_K_kin, self.pressure_im_size_required, self.bed_state, half_w_half_l
                # Get the geometry for sizing the pressure mat
                pmat_ArTagLib = ArTagLib()
                self.pressure_im_size_required, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend, half_w_half_l = \
                    pmat_ArTagLib.p_mat_geom(markers_c_drop, self.new_K_kin, self.pressure_im_size_required, self.bed_state, half_w_half_l)

                tf_corners = np.zeros((8, 2))
                tf_corners[0:8,:] = np.copy(self.tf_corners)

                #COLOR
                #if self.color is not 0:
                color_reshaped, color_size = VizLib().color_image(self.color, self.kcam, self.new_K_kin,
                                                                  u_c, v_c, u_c_drop, v_c_drop, u_c_pmat, v_c_pmat, camera_alpha_vert, camera_alpha_horiz)
                color_reshaped = imutils.rotate(color_reshaped, kinect_rotate_angle)
                color_reshaped = color_reshaped[pre_VERT_CUT+kinect_shift_up:-pre_VERT_CUT+kinect_shift_up,  HORIZ_CUT+kinect_shift_right : 540 - HORIZ_CUT+kinect_shift_right, :]
                tf_corners[0:4, :], color_reshaped = self.transform_selected_points(color_reshaped,
                                                                                             camera_alpha_vert,
                                                                                             camera_alpha_horiz,
                                                                                             kinect_rotate_angle,
                                                                                             kinect_shift_right,
                                                                                             kinect_shift_up, [1.0, 0],
                                                                                             [1.0, 0],
                                                                                             np.copy(self.tf_corners[0:4][:]))

                all_image_list.append(color_reshaped)

                #DEPTH
                h = VizLib().get_new_K_kin_homography(camera_alpha_vert, camera_alpha_horiz, self.new_K_kin)


                depth_r_orig = cv2.warpPerspective(self.depth_r, h, (self.depth_r.shape[1], self.depth_r.shape[0]))
                depth_r_orig = imutils.rotate(depth_r_orig, kinect_rotate_angle)
                depth_r_orig = depth_r_orig[HORIZ_CUT + kinect_shift_right: 540 - HORIZ_CUT + kinect_shift_right, pre_VERT_CUT - kinect_shift_up:-pre_VERT_CUT - kinect_shift_up]
                depth_r_reshaped, depth_r_size, depth_r_orig = VizLib().depth_image(depth_r_orig, u_c, v_c)
                self.depth_r_orig = depth_r_orig
                self.depthcam_midpixel = [self.new_K_kin[1, 2] - HORIZ_CUT - kinect_shift_right, (960-self.new_K_kin[0, 2]) - pre_VERT_CUT - kinect_shift_up]
                self.depthcam_midpixel2 = [self.new_K_kin[1, 2] - HORIZ_CUT, (960-self.new_K_kin[0, 2]) - pre_VERT_CUT]

                #print h, "H" #warping perspective
                #print kinect_rotate_angle #the amount to rotate counterclockwise about normal vector to the bed
                #print kinect_shift_right, kinect_shift_up #pixel shift of depth im. convert this to meters based on depth of

                depth_r_orig_nowarp = imutils.rotate(self.depth_r, 0)
                depth_r_orig_nowarp = depth_r_orig_nowarp[HORIZ_CUT + 0: 540 - HORIZ_CUT + 0, pre_VERT_CUT - 0:-pre_VERT_CUT - 0]
                depth_r_reshaped_nowarp, depth_r_size, depth_r_orig_nowarp = VizLib().depth_image(depth_r_orig_nowarp, u_c, v_c) #this just does two rotations

                all_image_list.append(depth_r_reshaped)
                all_image_list.append(depth_r_reshaped_nowarp)

                X,Y,Z = self.get_pc_from_depthmap(self.bed_state[0], tf_corners[2, :])


                #print self.pressure_im_size_required, color_size, u_c_drop, v_c_drop, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend


                #PRESSURE
                #pressure_vert_scale = 1.0
                #pressure_horiz_scale = 1.0
                #self.pressure = np.clip(self.pressure*4, 0, 100)
                pressure_reshaped, pressure_size, coords_from_top_left = VizLib().pressure_image(self.pressure, self.pressure_im_size,
                                                                           self.pressure_im_size_required, color_size,
                                                                           u_c_drop, v_c_drop, u_c_pmat, v_c_pmat,
                                                                           u_p_bend, v_p_bend)
                pressure_shape = pressure_reshaped.shape
                pressure_reshaped = cv2.resize(pressure_reshaped, None, fx=pressure_horiz_scale,
                                              fy=pressure_vert_scale)[0:pressure_shape[0],
                                              0:pressure_shape[1], :]

                if pressure_horiz_scale < 1.0 or pressure_vert_scale < 1.0:
                    pressure_reshaped_padded = np.zeros(pressure_shape).astype(np.uint8)
                    pressure_reshaped_padded[0:pressure_reshaped.shape[0], 0:pressure_reshaped.shape[1], :] += pressure_reshaped
                    pressure_reshaped = np.copy(pressure_reshaped_padded)

                coords_from_top_left[0] -= coords_from_top_left[0]*(1-pressure_horiz_scale)
                coords_from_top_left[1] += (960 - coords_from_top_left[1])*(1-pressure_vert_scale)

                pressure_reshaped = pressure_reshaped[pre_VERT_CUT:-pre_VERT_CUT,  HORIZ_CUT : 540 - HORIZ_CUT, :]


                all_image_list.append(pressure_reshaped)



                self.all_images = np.zeros((960-np.abs(pre_VERT_CUT)*2, 1, 3)).astype(np.uint8)
                for image in all_image_list:
                    #print image.shape
                    self.all_images = np.concatenate((self.all_images, image), axis = 1)

                self.all_images = self.all_images[VERT_CUT : 960 - VERT_CUT, :, :]



                is_not_mult_4 = True
                while is_not_mult_4 == True:
                    is_not_mult_4 = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount).shape[1]%4
                    self.overall_image_scale_amount+= 0.001

                coords_from_top_left[0] -= (HORIZ_CUT)
                coords_from_top_left[1] = 960 - pre_VERT_CUT - coords_from_top_left[1]
                self.coords_from_top_left = (np.array(coords_from_top_left) * self.overall_image_scale_amount)
                #print self.coords_from_top_left

                self.all_images = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount)
                self.cursor_shift = self.all_images.shape[1]/4


                self.all_images_clone = self.all_images.copy()


                cv2.imshow('all_images', self.all_images)
                k = cv2.waitKey(1)
                #cv2.waitKey(0)

                self.pc_all= [Y,X,-Z]
                #print np.shape(self.pc_all), "PC ALL SHAPE"

                #print self.tf_corners
                #print kinect_shift_up
                self.estimate_pose(self.pressure, self.bed_state[0], markers_c, model, model2, tf_corners, camera_alpha_vert,
                                   camera_alpha_horiz, h, kinect_rotate_angle, color_reshaped)


        #pkl.dump(self.RESULTS_DICT, open('/media/henry/multimodal_data_2/data/final_results/results_real_46K_'+PARTICIPANT+'_'+POSE_TYPE+'_'+NETWORK_2+'.p', 'wb'))



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
                        "S196",
                        ]

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



        F_eval = V3D.evaluate_data(dat, filename1, filename2)



