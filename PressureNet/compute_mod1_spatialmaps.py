#!/usr/bin/env python
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *


#PyTorch libraries
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import chumpy as ch

import convnet as convnet
import tf.transformations as tft

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

MAT_WIDTH = 0.762 #metres
MAT_HEIGHT = 1.854 #metres
MAT_HALF_WIDTH = MAT_WIDTH/2
NUMOFTAXELS_X = 64#73 #taxels
NUMOFTAXELS_Y = 27#30
NUMOFOUTPUTDIMS = 3
NUMOFOUTPUTNODES_TRAIN = 24
NUMOFOUTPUTNODES_TEST = 10
INTER_SENSOR_DISTANCE = 0.0286#metres
LOW_TAXEL_THRESH_X = 0
LOW_TAXEL_THRESH_Y = 0
HIGH_TAXEL_THRESH_X = (NUMOFTAXELS_X - 1)
HIGH_TAXEL_THRESH_Y = (NUMOFTAXELS_Y - 1)

DROPOUT = False


torch.set_num_threads(1)
if torch.cuda.is_available():
    # Use for GPU
    GPU = True
    dtype = torch.cuda.FloatTensor
    print'######################### CUDA is available! #############################'
else:
    # Use for CPU
    GPU = False
    dtype = torch.FloatTensor
    print'############################## USING CPU #################################'


class PhysicalTrainer():
    '''Gets the dictionary of pressure maps from the training database,
    and will have API to do all sorts of training with it.'''


    def __init__(self, testing_database_file_f, testing_database_file_m, opt, filename):
        '''Opens the specified pickle files to get the combined dataset:
        This dataset is a dictionary of pressure maps with the corresponding
        3d position and orientation of the markers associated with it.'''

        # change this to 'direct' when you are doing baseline methods
        self.CTRL_PNL = {}
        self.CTRL_PNL['batch_size'] = 64
        self.CTRL_PNL['loss_vector_type'] = opt.losstype
        self.CTRL_PNL['verbose'] = opt.verbose
        self.opt = opt
        self.CTRL_PNL['num_epochs'] = 100
        self.CTRL_PNL['incl_inter'] = True
        self.CTRL_PNL['shuffle'] = False
        self.CTRL_PNL['incl_ht_wt_channels'] = opt.htwt
        self.CTRL_PNL['omit_cntct_sobel'] = opt.omit_cntct_sobel
        self.CTRL_PNL['use_hover'] = opt.use_hover
        self.CTRL_PNL['incl_pmat_cntct_input'] = True
        self.CTRL_PNL['lock_root'] = False
        self.CTRL_PNL['num_input_channels'] = 2
        self.CTRL_PNL['GPU'] = GPU
        self.CTRL_PNL['dtype'] = dtype
        self.CTRL_PNL['repeat_real_data_ct'] = 1
        self.CTRL_PNL['regr_angles'] = 1
        self.CTRL_PNL['depth_map_labels'] = False
        self.CTRL_PNL['dropout'] = False
        self.CTRL_PNL['depth_map_labels_test'] = True #can only be true is we have 100% synth for testing
        self.CTRL_PNL['depth_map_output'] = True
        self.CTRL_PNL['depth_map_input_est'] = False #do this if we're working in a two-part regression
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
        self.CTRL_PNL['pmat_mult'] = int(1)
        self.CTRL_PNL['cal_noise'] = opt.calnoise
        self.CTRL_PNL['cal_noise_amt'] = 0.2
        self.CTRL_PNL['double_network_size'] = False
        self.CTRL_PNL['first_pass'] = True

        self.filename = filename

        if GPU == True:
            torch.cuda.set_device(self.opt.device)

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


        if self.CTRL_PNL['normalize_std'] == False:
            for i in range(10):
                self.CTRL_PNL['norm_std_coeffs'][i] *= 0.
                self.CTRL_PNL['norm_std_coeffs'][i] += 1.


        if self.CTRL_PNL['depth_map_output'] == True: #we need all the vertices if we're going to regress the depth maps
            self.verts_list = "all"
        else:
            self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]


        self.mat_size = (NUMOFTAXELS_X, NUMOFTAXELS_Y)
        self.output_size_train = (NUMOFOUTPUTNODES_TRAIN, NUMOFOUTPUTDIMS)
        self.output_size_val = (NUMOFOUTPUTNODES_TEST, NUMOFOUTPUTDIMS)
        self.parents = np.array([4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]).astype(np.int32)



        #################################### PREP TESTING DATA ##########################################
        #load training ysnth data
        if opt.small == True:
            reduce_data = True
        else:
            reduce_data = False

        # load in the test file
        test_dat_f_synth = TensorPrepLib().load_files_to_database(testing_database_file_f, creation_type = 'synth', reduce_data = reduce_data)
        test_dat_m_synth = TensorPrepLib().load_files_to_database(testing_database_file_m, creation_type = 'synth', reduce_data = reduce_data)
        test_dat_f_real = TensorPrepLib().load_files_to_database(testing_database_file_f, creation_type = 'real', reduce_data = reduce_data)
        test_dat_m_real = TensorPrepLib().load_files_to_database(testing_database_file_m, creation_type = 'real', reduce_data = reduce_data)


        for possible_dat in [test_dat_f_synth, test_dat_m_synth, test_dat_f_real, test_dat_m_real]:
            if possible_dat is not None:
                self.dat = possible_dat
                self.dat['mdm_est'] = []
                self.dat['cm_est'] = []
                self.dat['angles_est'] = []
                self.dat['root_xyz_est'] = []
                self.dat['betas_est'] = []
                self.dat['root_atan2_est'] = []



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

        if self.CTRL_PNL['verbose']: print self.test_x_tensor.size(), 'length of the testing dataset'
        if self.CTRL_PNL['verbose']: print self.test_y_tensor.size(), 'size of the testing database output'


        #self.test_x_tensor = self.test_x_tensor.unsqueeze(1)
        self.test_dataset = torch.utils.data.TensorDataset(self.test_x_tensor, self.test_y_tensor)
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, self.CTRL_PNL['batch_size'], shuffle=self.CTRL_PNL['shuffle'])


        self.model_name = 'convnet_1_'+str(self.opt.losstype)
        if self.opt.small == True: self.model_name += '_46000ct'
        else: self.model_name += '_184000ct'

        self.model_name += '_128b_x'+str(self.CTRL_PNL['pmat_mult'])+'pm_tnh'

        if self.opt.htwt == True: self.model_name += '_htwt'
        if self.opt.calnoise == True: self.model_name += '_clns20p'
        if self.opt.loss_root == True: self.model_name += '_rt'
        if self.opt.omit_cntct_sobel == True: self.model_name += '_ocs'
        if self.opt.half_shape_wt == True: self.model_name += '_hsw'


        self.model_name += '_100e_'+str(0.00002)+'lr'

        if GPU == True:
            self.model = torch.load('../data_BR/convnets/'+self.model_name + '.pt', map_location={'cuda:' + str(self.opt.prev_device):'cuda:' + str(self.opt.device)}).cuda()
        else:
            self.model = torch.load('../data_BR/convnets/'+self.model_name + '.pt', map_location='cpu')




        print 'Loaded ConvNet.'

        self.validate_convnet('test')


    def validate_convnet(self, verbose=False, n_batches=None):

        if DROPOUT == True:
            self.model.train()
        else:
            self.model.eval()
        loss = 0.
        n_examples = 0

        for batch_i, batch in enumerate(self.test_loader):

            if DROPOUT == True:
                batch[0] = batch[0].repeat(25, 1, 1, 1)
                batch[1] = batch[1].repeat(25, 1)
            #self.model.train()



            scores, INPUT_DICT, OUTPUT_DICT = \
                UnpackBatchLib().unpack_batch(batch, is_training=True, model=self.model,
                                                          CTRL_PNL=self.CTRL_PNL)

            self.CTRL_PNL['first_pass'] = False

            self.criterion = nn.L1Loss()
            scores_zeros = Variable(torch.Tensor(np.zeros((batch[0].shape[0], scores.size()[1]))).type(dtype),
                                    requires_grad=False)

            loss_curr = self.criterion(scores[:, 10:34], scores_zeros[:, 10:34]).data.item() / 10.

            loss += loss_curr



            print OUTPUT_DICT['batch_angles_est'].shape, n_examples
            for item in range(OUTPUT_DICT['batch_angles_est'].shape[0]):
                self.dat['mdm_est'].append(OUTPUT_DICT['batch_mdm_est'][item].cpu().numpy().astype(float32))
                self.dat['cm_est'].append(OUTPUT_DICT['batch_cm_est'][item].cpu().numpy().astype(int16))
                self.dat['angles_est'].append(OUTPUT_DICT['batch_angles_est'][item].cpu().numpy().astype(float32))
                self.dat['root_xyz_est'].append(OUTPUT_DICT['batch_root_xyz_est'][item].cpu().numpy().astype(float32))
                self.dat['betas_est'].append(OUTPUT_DICT['batch_betas_est'][item].cpu().numpy().astype(float32))
                if self.CTRL_PNL['full_body_rot'] == True:
                    self.dat['root_atan2_est'].append(OUTPUT_DICT['batch_root_atan2_est'][item].cpu().numpy().astype(float32))

            n_examples += self.CTRL_PNL['batch_size']
            #print n_examples

            if n_batches and (batch_i >= n_batches):
                break


            try:
                targets_print = torch.cat([targets_print, torch.mean(INPUT_DICT['batch_targets'], dim = 0).unsqueeze(0)], dim=0)
                targets_est_print = torch.cat([targets_est_print, torch.mean(OUTPUT_DICT['batch_targets_est'], dim = 0).unsqueeze(0)], dim=0)
            except:

                targets_print = torch.mean(INPUT_DICT['batch_targets'], dim = 0).unsqueeze(0)
                targets_est_print = torch.mean(OUTPUT_DICT['batch_targets_est'], dim = 0).unsqueeze(0)


            print targets_print.shape, INPUT_DICT['batch_targets'].shape
            print targets_est_print.shape, OUTPUT_DICT['batch_targets_est'].shape


            if GPU == True:
                error_norm, error_avg, _ = VisualizationLib().print_error_val(targets_print[-2:-1,:].cpu(),
                                                                                   targets_est_print[-2:-1,:].cpu(),
                                                                                   self.output_size_val,
                                                                                   self.CTRL_PNL['loss_vector_type'],
                                                                                   data='validate')
            else:
                error_norm, error_avg, _ = VisualizationLib().print_error_val(targets_print[-2:-1,:],
                                                                              targets_est_print[-2:-1,:],
                                                                                   self.output_size_val,
                                                                                   self.CTRL_PNL['loss_vector_type'],
                                                                                   data='validate')

            for item in self.dat:
                print item, len(self.dat[item])

        print self.filename

        #pkl.dump(self.dat,open('/media/henry/multimodal_data_2/'+self.filename+'_output0p7.p', 'wb'))
        pkl.dump(self.dat,open('../'+self.filename+'_'+self.model_name+'.p', 'wb'))



if __name__ == "__main__":
    #Initialize trainer with a training database file
    import optparse
    p = optparse.OptionParser()
    p.add_option('--computer', action='store', type = 'string', dest='computer', default='lab_harddrive',
                 help='Set path to the training database on lab harddrive.')

    p.add_option('--losstype', action='store', type = 'string', dest='losstype', default='anglesDC',
                 help='Choose direction cosine or euler angle regression.')

    p.add_option('--j_d_ratio', action='store', type = 'float', dest='j_d_ratio', default=0.5, #PMR parameter to adjust loss function 2
                 help='Set the loss mix: joints to depth planes. Only used for PMR regression.')

    p.add_option('--prev_device', action='store', type = 'int', dest='prev_device', default=0,
                 help='Choose a GPU core that it was previously on.')

    p.add_option('--device', action='store', type = 'int', dest='device', default=0,
                 help='Choose a GPU core.')

    p.add_option('--small', action='store_true', dest='small', default=False,
                 help='Make the dataset 1/4th of the original size.')

    p.add_option('--qt', action='store_true', dest='quick_test', default=False,
                 help='Do a quick test.')

    p.add_option('--htwt', action='store_true', dest='htwt', default=False,
                 help='Include height and weight info on the input.')

    p.add_option('--omit_cntct_sobel', action='store_true', dest='omit_cntct_sobel', default=False,
                 help='Cut contact and sobel from input.')

    p.add_option('--use_hover', action='store_true', dest='use_hover', default=False,
                 help='Use a hovermap for pmr input.')

    p.add_option('--calnoise', action='store_true', dest='calnoise', default=False,
                 help='Apply calibration noise to the input to facilitate sim to real transfer.')

    p.add_option('--half_shape_wt', action='store_true', dest='half_shape_wt', default=False,
                 help='Half betas.')

    p.add_option('--loss_root', action='store_true', dest='loss_root', default=False,
                 help='Use root in loss function.')

    p.add_option('--verbose', '--v',  action='store_true', dest='verbose',
                 default=True, help='Printout everything (under construction).')


    opt, args = p.parse_args()

    filepath_prefix_qt = '../'

    network_design = True

    if opt.quick_test == True:
        filename_list_f = ['data_BR/synth/quick_test/test_rollpi_f_lay_set23to24_3000_qt']
        filename_list_m = []
    else:
        filename_list_f = ['data_BR/synth/general_supine/test_roll0_f_lay_set14_1500',
                           'data_BR/synth/general_supine/test_roll0_plo_f_lay_set14_1500',
                           'data_BR/synth/general/test_rollpi_plo_f_lay_set23to24_3000',
                           'data_BR/synth/general/test_rollpi_f_lay_set23to24_3000',

                           'data_BR/synth/general_supine/train_roll0_f_lay_set5to7_5000',
                           'data_BR/synth/general_supine/train_roll0_f_lay_set10to13_8000',
                           'data_BR/synth/general_supine/train_roll0_plo_f_lay_set5to7_5000',
                           'data_BR/synth/general_supine/train_roll0_plo_f_lay_set10to13_8000',
                           'data_BR/synth/general/train_rollpi_f_lay_set10to17_16000',
                           'data_BR/synth/general/train_rollpi_f_lay_set18to22_10000',
                           'data_BR/synth/general/train_rollpi_plo_f_lay_set10to17_16000',
                           'data_BR/synth/general/train_rollpi_plo_f_lay_set18to22_10000',

                           'data_BR/synth/hands_behind_head/test_roll0_plo_hbh_f_lay_set4_500',
                           'data_BR/synth/prone_hands_up/test_roll0_plo_phu_f_lay_set1pa3_500',
                           'data_BR/synth/straight_limbs/test_roll0_sl_f_lay_set1both_500',
                           'data_BR/synth/crossed_legs/test_roll0_xl_f_lay_set1both_500',

                           'data_BR/synth/hands_behind_head/train_roll0_plo_hbh_f_lay_set1to2_2000',
                           'data_BR/synth/prone_hands_up/train_roll0_plo_phu_f_lay_set2pl4_4000',
                           'data_BR/synth/straight_limbs/train_roll0_sl_f_lay_set2pl3pa1_4000',
                           'data_BR/synth/crossed_legs/train_roll0_xl_f_lay_set2both_4000',]

        filename_list_m = ['data_BR/synth/general_supine/test_roll0_m_lay_set14_1500',
                           'data_BR/synth/general_supine/test_roll0_plo_m_lay_set14_1500',
                           'data_BR/synth/general/test_rollpi_m_lay_set23to24_3000',
                           'data_BR/synth/general/test_rollpi_plo_m_lay_set23to24_3000',

                           'data_BR/synth/general_supine/train_roll0_m_lay_set5to7_5000',
                           'data_BR/synth/general_supine/train_roll0_m_lay_set10to13_8000',
                           'data_BR/synth/general_supine/train_roll0_plo_m_lay_set5to7_5000',
                           'data_BR/synth/general_supine/train_roll0_plo_m_lay_set10to13_8000',
                           'data_BR/synth/general/train_rollpi_m_lay_set10to17_16000',
                           'data_BR/synth/general/train_rollpi_m_lay_set18to22_10000',
                           'data_BR/synth/general/train_rollpi_plo_m_lay_set10to17_16000',
                           'data_BR/synth/general/train_rollpi_plo_m_lay_set18to22_10000',
                           
                           'data_BR/synth/hands_behind_head/test_roll0_plo_hbh_m_lay_set1_500',
                           'data_BR/synth/prone_hands_up/test_roll0_plo_phu_m_lay_set1pa3_500',
                           'data_BR/synth/straight_limbs/test_roll0_sl_m_lay_set1both_500',
                           'data_BR/synth/crossed_legs/test_roll0_xl_m_lay_set1both_500',

                           'data_BR/synth/hands_behind_head/train_roll0_plo_hbh_m_lay_set2pa1_2000',
                           'data_BR/synth/prone_hands_up/train_roll0_plo_phu_m_lay_set2pl4_4000',
                           'data_BR/synth/straight_limbs/train_roll0_sl_m_lay_set2pa1_4000',
                           'data_BR/synth/crossed_legs/train_roll0_xl_m_lay_set2both_4000',]



    for filename in filename_list_m:

        test_database_file_f = []
        test_database_file_m = []
        test_database_file_m.append(filepath_prefix_qt + filename + '.p')

        p = PhysicalTrainer(test_database_file_f, test_database_file_m, opt, filename)

        print "GOT HERE!"
        p.init_convnet_test()
        #p.visualize_3d_data()

    for filename in filename_list_f:

        test_database_file_f = []
        test_database_file_m = []
        test_database_file_f.append(filepath_prefix_qt + filename + '.p')

        p = PhysicalTrainer(test_database_file_f, test_database_file_m, opt, filename)

        print "GOT HERE!"
        p.init_convnet_test()
        #p.visualize_3d_data()