#!/usr/bin/env python

#Bodies at Rest: Code to visualize real dataset.
#(c) Henry M. Clever
#Major updates made for CVPR release: December 10, 2019


import numpy as np
import random
import copy

import sys
import os
sys.path.append(os.path.abspath('..'))
sys.path.insert(0, '../lib_py')
import lib_pyrender_basic as libPyRender
from smpl.smpl_webuser.serialization import load_model

import cPickle as pkl

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



#volumetric pose gen libraries
from time import sleep
from scipy.stats import mode
import os.path as osp
import imutils


import matplotlib.cm as cm #use cm.jet(list)

DATASET_CREATE_TYPE = 1

import cv2
from camera import Camera

import math
from random import shuffle
import torch
import torch.nn as nn

import cPickle as pickle
VERT_CUT, HORIZ_CUT = 0, 50
pre_VERT_CUT = 40

#MISC
import time as time
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

MAT_SIZE = (64, 27)


CAM_BED_DIST = 1.66



class Viz3DPose():
    def __init__(self):
        self.pyRender = libPyRender.pyRenderMesh(render = True)
        self.pressure_im_size = (64, 27)
        self.pressure_im_size_required = (64, 27)
        self.overall_image_scale_amount = 0.85



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



    def evaluate_data(self, dat):



        #for im_num in range(29, 100):
        for im_num in range(0, len(dat['images'])):#self.color_all.shape[0]):

            all_image_list = []

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

            all_image_list.append(rgb)


            depth_r_orig = dat['depth'][im_num]
            depth_r_reshaped, depth_r_size, depth_r_orig = self.depth_image(depth_r_orig)
            all_image_list.append(depth_r_reshaped)



            self.pressure = dat['images'][im_num]
            #PRESSURE


            #because we used a sheet on the bed the overall pressure is lower than calibration, which was done without a sheet. bump it up here.
            bedsheet_norm_factor = float(4)

            self.pressure = np.clip(self.pressure*bedsheet_norm_factor, 0, 100)
            pressure_reshaped, pressure_size = self.pressure_image(self.pressure, rgb.shape, pmat_corners)

            #pressure_reshaped = pressure_reshaped[pre_VERT_CUT:-pre_VERT_CUT,  HORIZ_CUT : 540 - HORIZ_CUT, :]
            all_image_list.append(pressure_reshaped)



            self.all_images = np.zeros((960-np.abs(pre_VERT_CUT)*2, 1, 3)).astype(np.uint8)
            for image in all_image_list:
                self.all_images = np.concatenate((self.all_images, image), axis = 1)

            self.all_images = self.all_images[VERT_CUT : 960 - VERT_CUT, :, :]



            is_not_mult_4 = True
            while is_not_mult_4 == True:
                is_not_mult_4 = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount).shape[1]%4
                self.overall_image_scale_amount+= 0.001


            self.all_images = cv2.resize(self.all_images, (0, 0), fx=self.overall_image_scale_amount, fy=self.overall_image_scale_amount)
            self.cursor_shift = self.all_images.shape[1]/4


            self.all_images_clone = self.all_images.copy()


            cv2.imshow('all_images', self.all_images)
            k = cv2.waitKey(1)
            #cv2.waitKey(0)



            #now do 3D rendering
            pmat = np.clip(self.pressure.reshape(MAT_SIZE), a_min=0, a_max=100)

            pc_autofil_red = dat['pc'][im_num]

            camera_point = [1.09898028, 0.46441343, -CAM_BED_DIST] #[dist from foot of bed, dist from left side of mat, dist normal]

            #
            self.pyRender.render_3D_data(camera_point, pmat = pmat, pc = pc_autofil_red)

            self.point_cloud_array = None
            sleep(1)




if __name__ ==  "__main__":

    import optparse



    GENDER = "f"

    #Replace this with some subset of data of your choice
    TESTING_FILENAME = "quick_test/test_rollpi_"+GENDER+"_lay_set23to24_3000"




    p = optparse.OptionParser()
    p.add_option('--pose_type', action='store', type='string', dest='pose_type', default='none',
                 help='Choose a pose type, either `prescribed` or `p_select`.')

    p.add_option('--p_idx', action='store', type='int', dest='p_idx', default=0,
                 # PMR parameter to adjust loss function 2
                 help='Choose a participant. Enter a number from 1 to 20.')

    p.add_option('--hd', action='store_true', dest='hd', default=False,
                 help='Read and write to data on an external harddrive.')
    

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

    if opt.p_idx == 0:
        print "Please choose a participant with flag `--p_idx #`. Enter a number from 1 to 20."
        sys.exit()
    else:
        PARTICIPANT = participant_list[opt.p_idx - 1]

    V3D = Viz3DPose()


    if opt.hd == True:
        participant_directory = "/media/henry/multimodal_data_2/data_BR/real/"+PARTICIPANT
    else:
        participant_directory = "../data_BR/real/"+PARTICIPANT
        

    V3D.load_new_participant_info(participant_directory)

    if opt.pose_type == "prescribed":
        dat = load_pickle(participant_directory+"/prescribed.p")
    elif opt.pose_type == "p_select":
        dat = load_pickle(participant_directory+"/p_select.p")
    else:
        print "Please choose a pose type - either prescribed poses, " \
              "'--pose_type prescribed', or participant selected poses, '--pose_type p_select'."
        sys.exit()

    F_eval = V3D.evaluate_data(dat)



