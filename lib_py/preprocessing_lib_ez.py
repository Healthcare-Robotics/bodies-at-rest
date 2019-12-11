#!/usr/bin/env python

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *

import cPickle as pkl
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.misc import imresize
from scipy.ndimage.interpolation import zoom

from scipy.ndimage.filters import gaussian_filter


# PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import transforms
from torch.autograd import Variable


MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)



class PreprocessingLib():
    def __init__(self):
        pass

    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d



    def preprocessing_add_image_noise(self, images, pmat_chan_idx, norm_std_coeffs):

        queue = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+2, :, :])
        queue[queue != 0] = 1.


        x = np.arange(-10, 10)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=1) - ss.norm.cdf(xL,scale=1)  # scale is the standard deviation using a cumulative density function
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        image_noise = np.random.choice(x, size=(images.shape[0], 2, images.shape[2], images.shape[3]), p=prob)

        
        image_noise = image_noise*queue
        image_noise = image_noise.astype(float)
        #image_noise[:, 0, :, :] /= 11.70153502792190
        #image_noise[:, 1, :, :] /= 45.61635847182483
        image_noise[:, 0, :, :] *= norm_std_coeffs[4]
        image_noise[:, 1, :, :] *= norm_std_coeffs[5]

        images[:, pmat_chan_idx:pmat_chan_idx+2, :, :] += image_noise

        #print images[0, 0, 50, 10:25], 'added noise'

        #clip noise so we dont go outside sensor limits
        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
        images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[4])
        #images[:, pmat_chan_idx+1, :, :] = np.clip(images[:, pmat_chan_idx+1, :, :], 0, 10000)
        return images


    def preprocessing_add_calibration_noise(self, images, pmat_chan_idx, norm_std_coeffs, is_training):
        if is_training == True:
            variation_amount = 0.1
            print "ADDING CALIB NOISE", variation_amount

            #pmat_contact_orig = np.copy(images[:, pmat_chan_idx, :, :])
            #pmat_contact_orig[pmat_contact_orig != 0] = 1.
            #sobel_contact_orig = np.copy(images[:, pmat_chan_idx+1, :, :])
            #sobel_contact_orig[sobel_contact_orig != 0] = 1.

            for map_index in range(images.shape[0]):

                pmat_contact_orig = np.copy(images[map_index, pmat_chan_idx, :, :])
                pmat_contact_orig[pmat_contact_orig != 0] = 1.
                sobel_contact_orig = np.copy(images[map_index, pmat_chan_idx + 1, :, :])
                sobel_contact_orig[sobel_contact_orig != 0] = 1.


                # first multiply
                amount_to_mult_im = random.normalvariate(mu = 1.0, sigma = variation_amount) #mult a variation of 10%
                amount_to_mult_sobel = random.normalvariate(mu = 1.0, sigma = variation_amount) #mult a variation of 10%
                images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, :] * amount_to_mult_im
                images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, :] * amount_to_mult_sobel

                # then add
                #amount_to_add_im = random.normalvariate(mu = 0.0, sigma = (1./11.70153502792190)*(98.666 - 0.0)*0.1) #add a variation of 10% of the range
                #amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = (1./45.61635847182483)*(386.509 - 0.0)*0.1) #add a variation of 10% of the range
                amount_to_add_im = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[4]*(98.666 - 0.0)*variation_amount) #add a variation of 10% of the range
                amount_to_add_sobel = random.normalvariate(mu = 0.0, sigma = norm_std_coeffs[5]*(386.509 - 0.0)*variation_amount) #add a variation of 10% of the range

                images[map_index, pmat_chan_idx, :, :] = images[map_index, pmat_chan_idx, :, :] + amount_to_add_im
                images[map_index, pmat_chan_idx+1, :, :] = images[map_index, pmat_chan_idx+1, :, :] + amount_to_add_sobel
                images[map_index, pmat_chan_idx, :, :] = np.clip(images[map_index, pmat_chan_idx, :, :], a_min = 0., a_max = 10000)
                images[map_index, pmat_chan_idx+1, :, :] = np.clip(images[map_index, pmat_chan_idx+1, :, :], a_min = 0., a_max = 10000)

                #cut out the background. need to do this after adding.
                images[map_index, pmat_chan_idx, :, :] *= pmat_contact_orig#[map_index, :, :]
                images[map_index, pmat_chan_idx+1, :, :] *= sobel_contact_orig#[map_index, :, :]


                amount_to_gauss_filter_im = random.normalvariate(mu = 0.5, sigma = variation_amount)
                amount_to_gauss_filter_sobel = random.normalvariate(mu = 0.5, sigma = variation_amount)
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= amount_to_gauss_filter_im) #pmap
                images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= amount_to_gauss_filter_sobel) #sobel #NOW


            #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
            images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100.*norm_std_coeffs[4])

        else:  #if its NOT training we should still blur things by 0.5
            for map_index in range(images.shape[0]):
                images[map_index, pmat_chan_idx, :, :] = gaussian_filter(images[map_index, pmat_chan_idx, :, :], sigma= 0.5) #pmap
                images[map_index, pmat_chan_idx+1, :, :] = gaussian_filter(images[map_index, pmat_chan_idx+1, :, :], sigma= 0.5) #sobel



        #images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100/11.70153502792190)
        images[:, pmat_chan_idx, :, :] = np.clip(images[:, pmat_chan_idx, :, :], 0, 100*norm_std_coeffs[4])

        #now calculate the contact map AFTER we've blurred it
        pmat_contact = np.copy(images[:, pmat_chan_idx:pmat_chan_idx+1, :, :])
        #pmat_contact[pmat_contact != 0] = 100./41.80684362163343
        pmat_contact[pmat_contact != 0] = 100.*norm_std_coeffs[0]
        images = np.concatenate((pmat_contact, images), axis = 1)

        #for i in range(0, 20):
        #    VisualizationLib().visualize_pressure_map(images[i, 0, :, :] * 20., None, None,
        #                                              images[i, 1, :, :] * 20., None, None,
        #                                              block=False)
        #    time.sleep(0.5)

        return images


    def preprocessing_pressure_array_resize(self, data, mat_size, verbose):
        '''Will resize all elements of the dataset into the dimensions of the
        pressure map'''
        p_map_dataset = []
        for map_index in range(len(data)):
            #print map_index, self.mat_size, 'mapidx'
            #Resize mat to make into a matrix
            p_map = np.reshape(data[map_index], mat_size)
            #print p_map
            p_map_dataset.append(p_map)
            #print p_map.shape
        if verbose: print len(data[0]),'x',1, 'size of an incoming pressure map'
        if verbose: print len(p_map_dataset[0]),'x',len(p_map_dataset[0][0]), 'size of a resized pressure map'
        return p_map_dataset

    def preprocessing_create_pressure_angle_stack_realtime(self, p_map, bedangle, mat_size, verbose = False):
        '''This is for creating a 2-channel input using the height of the bed. '''
        p_map = np.reshape(p_map, mat_size)

        if verbose:
            print np.shape(p_map)
            print p_map.shape
            print np.shape(bedangle), 'angle dat'

            print 'calculating height matrix and sobel filter'
        p_map_dataset = []


        height_strip = np.zeros(np.shape(p_map)[0])
        height_strip[0:25] = np.flip(np.linspace(0, 1, num=25) * 25 * 2.86 * np.sin(np.deg2rad(bedangle)),
                                      axis=0)
        height_strip = np.repeat(np.expand_dims(height_strip, axis=1), 27, 1)
        a_map = height_strip


        # this makes a sobel edge on the image
        sx = ndimage.sobel(p_map, axis=0, mode='constant')
        sy = ndimage.sobel(p_map, axis=1, mode='constant')
        p_map_inter = np.hypot(sx, sy)

        p_map_dataset.append([p_map, p_map_inter, a_map])

        return p_map_dataset


    def preprocessing_blur_images(self, x_data, mat_size, sigma):

        x_data_return = []
        for map_index in range(len(x_data)):
            p_map = np.reshape(x_data[map_index], mat_size)

            p_map = gaussian_filter(p_map, sigma= sigma)

            x_data_return.append(p_map.flatten())

        return x_data_return




    def preprocessing_create_pressure_angle_stack(self,x_data, a_data, mat_size, CTRL_PNL):
        '''This is for creating a 2-channel input using the height of the bed. '''

        if CTRL_PNL['verbose']: print np.max(x_data)
        x_data = np.clip(x_data, 0, 100)

        if CTRL_PNL['verbose']:
            print np.shape(x_data)
            print np.shape(a_data), 'angle dat'
            #print a_data

            print 'calculating height matrix and sobel filter'
        p_map_dataset = []
        for map_index in range(len(x_data)):
            # print map_index, self.mat_size, 'mapidx'
            # Resize mat to make into a matrix


            p_map = np.reshape(x_data[map_index], mat_size)

            if mat_size == (84, 47):
                p_map = p_map[10:74, 10:37]

            height_strip = np.zeros(np.shape(p_map)[0])
            height_strip[0:25] = np.flip(np.linspace(0, 1, num=25) * 25 * 2.86 * np.sin(np.deg2rad(a_data[map_index][0])), axis = 0)
            height_strip = np.repeat(np.expand_dims(height_strip, axis = 1), 27, 1)
            a_map = height_strip

            #ZACKORY: ALTER THE VARIABLE "p_map" HERE TO STANDARDIZE. IT IS AN 84x47 MATRIX WITHIN EACH LOOP.
            #THIS ALTERATION WILL ALSO CHANGE HOW THE EDGE IS CALCULATED. IF YOU WANT TO DO THEM SEPARATELY,
            #THEN DO IT AFTER THE 'INCLUDE INTER' IF STATEMENT.
            #p_map = standardize(p_map)


            if CTRL_PNL['incl_inter'] == True:
                # this makes a sobel edge on the image
                sx = ndimage.sobel(p_map, axis=0, mode='constant')
                sy = ndimage.sobel(p_map, axis=1, mode='constant')
                p_map_inter = np.hypot(sx, sy)
                if CTRL_PNL['clip_sobel'] == True:
                    p_map_inter = np.clip(p_map_inter, a_min=0, a_max = 100)
                p_map_dataset.append([p_map, p_map_inter, a_map])
            else:
                p_map_dataset.append([p_map, a_map])
        if CTRL_PNL['verbose']: print len(x_data[0]), 'x', 1, 'size of an incoming pressure map'
        if CTRL_PNL['verbose']: print len(p_map_dataset[0][0]), 'x', len(p_map_dataset[0][0][0]), 'size of a resized pressure map'
        if CTRL_PNL['verbose']: print len(p_map_dataset[0][1]), 'x', len(p_map_dataset[0][1][0]), 'size of sobel filtered map'
        if CTRL_PNL['verbose']: print len(p_map_dataset[0][2]), 'x', len(p_map_dataset[0][2][0]), 'size of angle array'

        return p_map_dataset


    def preprocessing_pressure_map_upsample(self, data, multiple, order=1):
        '''Will upsample an incoming pressure map dataset'''
        p_map_highres_dataset = []


        if len(np.shape(data)) == 3:
            for map_index in range(len(data)):
                #Upsample the current map using bilinear interpolation
                p_map_highres_dataset.append(
                        ndimage.zoom(data[map_index], multiple, order=order))
        elif len(np.shape(data)) == 4:
            for map_index in range(len(data)):
                p_map_highres_dataset_subindex = []
                for map_subindex in range(len(data[map_index])):
                    #Upsample the current map using bilinear interpolation
                    p_map_highres_dataset_subindex.append(ndimage.zoom(data[map_index][map_subindex], multiple, order=order))
                p_map_highres_dataset.append(p_map_highres_dataset_subindex)

        return p_map_highres_dataset



    def pad_pressure_mats(self,NxHxWimages):
        padded = np.zeros((NxHxWimages.shape[0],NxHxWimages.shape[1]+20,NxHxWimages.shape[2]+20))
        padded[:,10:74,10:37] = NxHxWimages
        NxHxWimages = padded
        return NxHxWimages



    def person_based_loocv(self):
        '''Computes Person Based Leave One Out Cross Validation. This means
        that if we have 10 participants, we train using 9 participants and test
        on 1 participant, and so on.
        To run this function, make sure that each subject_* directory in the
        dataset/ directory has a pickle file called individual_database.p
        If you don't have it in some directory that means you haven't run,
        create_raw_database.py on that subject's dataset. So create it and
        ensure that the pkl file is created successfully'''
        #Entire pressure dataset with coordinates in world frame
        dataset_dirname = os.path.dirname(os.path.realpath(training_database_file))
        print dataset_dirname
        subject_dirs = [x[0] for x in os.walk(dataset_dirname)]
        subject_dirs.pop(0)
        print subject_dirs
        dat = []
        for i in range(len(subject_dirs)):
            try:
                dat.append(pkl.load(open(os.path.join(subject_dirs[i],
                    'individual_database.p'), "rb")))
            except:
                print "Following dataset directory not formatted correctly. Is there an individual_dataset pkl file for every subject?"
                print os.path.join(subject_dirs[i], 'individual_database.p')
                sys.exit()
        print "Inserted all individual datasets into a list of dicts"
        print "Number of subjects:"
        print len(dat)
        mean_joint_error = np.zeros((len(dat), 10))
        std_joint_error = np.zeros((len(dat), 10))
        for i in range(len(dat)):
            train_dat = {}
            test_dat = dat[i]
            for j in range(len(dat)):
                if j == i:
                    print "#of omitted data points"
                    print len(dat[j].keys())
                    pass
                else:
                    print len(dat[j].keys())
                    print j
                    train_dat.update(dat[j])
            rand_keys = train_dat.keys()
            print "Training Dataset Size:"
            print len(rand_keys)
            print "Testing dataset size:"
            print len(test_dat.keys())
            self.train_y = [] #Initialize the training coordinate list
            self.dataset_y = [] #Initialization for the entire dataset
            self.train_x_flat = rand_keys[:]#Pressure maps
            [self.train_y.append(train_dat[key]) for key in self.train_x_flat]#Coordinates
            self.test_x_flat = test_dat.keys()#Pressure maps(test dataset)
            self.test_y = [] #Initialize the ground truth list
            [self.test_y.append(test_dat[key]) for key in self.test_x_flat]#ground truth
            self.dataset_x_flat = rand_keys[:]#Pressure maps
            [self.dataset_y.append(train_dat[key]) for key in self.dataset_x_flat]
            self.cv_fold = 3 # Value of k in k-fold cross validation
            self.mat_frame_joints = []
            p.train_hog_knn()
            (mean_joint_error[i][:], std_joint_error[i][:]) = self.test_learning_algorithm(self.regr)
            print "Mean Error:"
            print mean_joint_error
        print "MEAN ERROR AFTER PERSON LOOCV:"
        total_mean_error = np.mean(mean_joint_error, axis=0)
        total_std_error = np.mean(std_joint_error, axis=0)
        print total_mean_error
        print "STD DEV:"
        print total_std_error
        pkl.dump(mean_joint_error, open('./dataset/mean_loocv_results.p', 'w'))
        pkl.dump(mean_joint_error, open('./dataset/std_loocv_results.p', 'w'))

