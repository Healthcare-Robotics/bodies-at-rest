import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.stats as ss
import torchvision
import time

import sys
sys.path.insert(0, '../lib_py')


from visualization_lib_br import VisualizationLib
from kinematics_lib_br import KinematicsLib
from mesh_depth_lib import MeshDepthLib


class CNN(nn.Module):
    def __init__(self, out_size, loss_vector_type, batch_size, verts_list, in_channels = 3):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            mat_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            out_size (int): Number of classes to score
        '''

        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #print mat_size
        self.loss_vector_type = loss_vector_type
        print '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$'

        self.count = 0


        self.CNN_pack1 = nn.Sequential(

            nn.Conv2d(in_channels, 192, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1, inplace=False),

        )

        self.CNN_packtanh = nn.Sequential(

            nn.Conv2d(in_channels, 192, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
        )

        self.CNN_packtanh_double = nn.Sequential(

            nn.Conv2d(in_channels, 384, kernel_size=7, stride=2, padding=3),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv2d(768, 768, kernel_size=3, stride=1, padding=0),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
        )

        self.CNN_pack2 = nn.Sequential(

            nn.Conv2d(in_channels, 32, kernel_size = 7, stride = 2, padding = 3),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding= 0),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1, inplace=False),

        )


        self.CNN_fc1 = nn.Sequential(
            nn.Linear(67200, out_size), #89600, out_size),
        )
        self.CNN_fc1_double = nn.Sequential(
            nn.Linear(67200*2, out_size), #89600, out_size),
        )

        print 'Out size:', out_size

        if torch.cuda.is_available():
            self.GPU = True
            # Use for self.GPU
            dtype = torch.cuda.FloatTensor
            print('######################### CUDA is available! #############################')
        else:
            self.GPU = False
            # Use for CPU
            dtype = torch.FloatTensor
            print('############################## USING CPU #################################')
        self.dtype = dtype

        self.verts_list = verts_list
        self.meshDepthLib = MeshDepthLib(loss_vector_type, batch_size, verts_list = self.verts_list)



    def forward_kinematic_angles(self, images, gender_switch, synth_real_switch, CTRL_PNL, OUTPUT_EST_DICT,
                                 targets=None, is_training = True, betas=None, angles_gt = None, root_shift = None):


        #cut out the sobel and contact channels
        if CTRL_PNL['omit_cntct_sobel'] == True:
            images = torch.cat((images[:, 1:CTRL_PNL['num_input_channels_batch0'], :, :], images[:, CTRL_PNL['num_input_channels_batch0']+1:, :, :]), dim = 1)

        reg_angles = CTRL_PNL['regr_angles']

        OUTPUT_DICT = {}

        self.GPU = CTRL_PNL['GPU']
        self.dtype = CTRL_PNL['dtype']

        #print(torch.cuda.max_memory_allocated(), 'conv0', images.size())
        if CTRL_PNL['first_pass'] == False:
            x = self.meshDepthLib.bounds
            #print blah
            #self.GPU = False
            #self.dtype = torch.FloatTensor

        else:
            if CTRL_PNL['GPU'] == True:
                self.GPU = True
                self.dtype = torch.cuda.FloatTensor
            else:
                self.GPU = False
                self.dtype = torch.FloatTensor
            if CTRL_PNL['depth_map_output'] == True:
                self.verts_list = "all"
            else:
                self.verts_list = [1325, 336, 1032, 4515, 1374, 4848, 1739, 5209, 1960, 5423]
            self.meshDepthLib = MeshDepthLib(loss_vector_type=self.loss_vector_type,
                                             batch_size=images.size(0), verts_list = self.verts_list)

        if CTRL_PNL['all_tanh_activ'] == True:
            if CTRL_PNL['double_network_size'] == False:
                scores_cnn = self.CNN_packtanh(images)
            else:
                scores_cnn = self.CNN_packtanh_double(images)

        else:
            scores_cnn = self.CNN_pack1(images)

        scores_size = scores_cnn.size()


        # This combines the height, width, and filters into a single dimension
        scores_cnn = scores_cnn.view(images.size(0),scores_size[1] *scores_size[2]*scores_size[3])

        # this output is N x 85: betas, root shift, angles
        if CTRL_PNL['double_network_size'] == False:
            scores = self.CNN_fc1(scores_cnn)
        else:
            scores = self.CNN_fc1_double(scores_cnn)


        # weight the outputs, which are already centered around 0. First make them uniformly smaller than the direct output, which is too large.
        if CTRL_PNL['adjust_ang_from_est'] == True:
            scores = torch.mul(scores.clone(), 0.01)
        else:
            scores = torch.mul(scores.clone(), 0.01)

        #normalize the output of the network based on the range of the parameters
        if self.GPU == True:
            output_norm = 10*[6.0] + [0.91, 1.98, 0.15] + list(torch.abs(self.meshDepthLib.bounds.view(72, 2)[:, 1] - self.meshDepthLib.bounds.view(72,2)[:, 0]).cpu().numpy())
        else:
            output_norm = 10*[6.0] + [0.91, 1.98, 0.15] + list(torch.abs(self.meshDepthLib.bounds.view(72, 2)[:, 1] - self.meshDepthLib.bounds.view(72, 2)[:, 0]).numpy())
        for i in range(85):
            scores[:, i] = torch.mul(scores[:, i].clone(), output_norm[i])


        #add a factor so the model starts close to the home position. Has nothing to do with weighting.

        if CTRL_PNL['lock_root'] == True:
            scores[:, 10] = torch.add(scores[:, 10].clone(), 0.6).detach()
            scores[:, 11] = torch.add(scores[:, 11].clone(), 1.2).detach()
            scores[:, 12] = torch.add(scores[:, 12].clone(), 0.1).detach()
        elif CTRL_PNL['adjust_ang_from_est'] == True:
            pass
        else:
            scores[:, 10] = torch.add(scores[:, 10].clone(), 0.6)
            scores[:, 11] = torch.add(scores[:, 11].clone(), 1.2)
            scores[:, 12] = torch.add(scores[:, 12].clone(), 0.1)

        #scores[:, 12] = torch.add(scores[:, 12].clone(), 0.06)

        if CTRL_PNL['full_body_rot'] == True:

            scores = scores.unsqueeze(0)
            scores = scores.unsqueeze(0)
            scores = F.pad(scores, (0, 3, 0, 0))
            scores = scores.squeeze(0)
            scores = scores.squeeze(0)

            if CTRL_PNL['adjust_ang_from_est'] == True:

                scores[:, 13:19] = scores[:, 13:19].clone() + OUTPUT_EST_DICT['root_atan2']


            scores[:, 22:91] = scores[:, 19:88].clone()

            scores[:, 19] = torch.atan2(scores[:, 16].clone(), scores[:, 13].clone()) #pitch x, y
            scores[:, 20] = torch.atan2(scores[:, 17].clone(), scores[:, 14].clone()) #roll x, y
            scores[:, 21] = torch.atan2(scores[:, 18].clone(), scores[:, 15].clone()) #yaw x, y

            OSA = 6 #output size adder
        else:
            OSA = 0




        #print scores[0, 0:10]
        if CTRL_PNL['adjust_ang_from_est'] == True:
            scores[:, 0:10] =  OUTPUT_EST_DICT['betas'] + scores[:, 0:10].clone()
            scores[:, 10:13] = OUTPUT_EST_DICT['root_shift'] + scores[:, 10:13].clone()
            if CTRL_PNL['full_body_rot'] == True:
                scores[:, 22:91] = scores[:, 22:91].clone() + OUTPUT_EST_DICT['angles'][:, 3:72]
            else:
                scores[:, 13:85] = scores[:, 13:85].clone() + OUTPUT_EST_DICT['angles']
            #scores[:, 13:85] = OUTPUT_EST_DICT['angles']


        OUTPUT_DICT['batch_betas_est'] = scores[:, 0:10].clone().data
        if CTRL_PNL['full_body_rot'] == True:
            OUTPUT_DICT['batch_root_atan2_est'] = scores[:, 13:19].clone().data
        OUTPUT_DICT['batch_angles_est']  = scores[:, 13+OSA:85+OSA].clone().data
        OUTPUT_DICT['batch_root_xyz_est'] = scores[:, 10:13].clone().data


        if reg_angles == True:
            add_idx = 72
        else:
            add_idx = 0


        if CTRL_PNL['clip_betas'] == True:
            scores[:, 0:10] /= 3.
            scores[:, 0:10] = scores[:, 0:10].tanh()
            scores[:, 0:10] *= 3.

        #print self.meshDepthLib.bounds

        test_ground_truth = False #can only use True when the dataset is entirely synthetic AND when we use anglesDC
        #is_training = True

        if test_ground_truth == False or is_training == False:
            # make sure the estimated betas are reasonable.

            betas_est = scores[:, 0:10].clone()#.detach() #make sure to detach so the gradient flow of joints doesn't corrupt the betas
            root_shift_est = scores[:, 10:13].clone()


            # normalize for tan activation function
            scores[:, 13+OSA:85+OSA] -= torch.mean(self.meshDepthLib.bounds[0:72, 0:2], dim=1)
            scores[:, 13+OSA:85+OSA] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
            scores[:, 13+OSA:85+OSA] = scores[:, 13+OSA:85+OSA].tanh()
            scores[:, 13+OSA:85+OSA] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
            scores[:, 13+OSA:85+OSA] += torch.mean(self.meshDepthLib.bounds[0:72, 0:2], dim=1)


            #print scores[:, 13+OSA:85+OSA]


            if self.loss_vector_type == 'anglesDC':

                Rs_est = KinematicsLib().batch_rodrigues(scores[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

            elif self.loss_vector_type == 'anglesEU':

                Rs_est = KinematicsLib().batch_euler_to_R(scores[:, 13+OSA:85+OSA].view(-1, 24, 3).clone(), self.meshDepthLib.zeros_cartesian, self.meshDepthLib.ones_cartesian).view(-1, 24, 3, 3)

        else:
            #print betas[13, :], 'betas'
            betas_est = betas
            scores[:, 0:10] = betas.clone()
            scores[:, 13+OSA:85+OSA] = angles_gt.clone()
            root_shift_est = root_shift


            if self.loss_vector_type == 'anglesDC':

                #normalize for tan activation function
                #scores[:, 13+OSA:85+OSA] -= torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)
                #scores[:, 13+OSA:85+OSA] *= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                #scores[:, 13+OSA:85+OSA] = scores[:, 13+OSA:85+OSA].tanh()
                #scores[:, 13+OSA:85+OSA] /= (2. / torch.abs(self.meshDepthLib.bounds[0:72, 0] - self.meshDepthLib.bounds[0:72, 1]))
                #scores[:, 13+OSA:85+OSA] += torch.mean(self.meshDepthLib.bounds[0:72,0:2], dim = 1)


                Rs_est = KinematicsLib().batch_rodrigues(scores[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()).view(-1, 24, 3, 3)

        #print Rs_est[0, :]


        OUTPUT_DICT['batch_betas_est_post_clip'] = scores[:, 0:10].clone().data
        if self.loss_vector_type == 'anglesEU':
            OUTPUT_DICT['batch_angles_est_post_clip']  = KinematicsLib().batch_dir_cos_angles_from_euler_angles(scores[:, 13+OSA:85+OSA].view(-1, 24, 3).clone(), self.meshDepthLib.zeros_cartesian, self.meshDepthLib.ones_cartesian)
        elif self.loss_vector_type == 'anglesDC':
            OUTPUT_DICT['batch_angles_est_post_clip']  = scores[:, 13+OSA:85+OSA].view(-1, 24, 3).clone()
        OUTPUT_DICT['batch_root_xyz_est_post_clip'] = scores[:, 10:13].clone().data


        gender_switch = gender_switch.unsqueeze(1)
        current_batch_size = gender_switch.size()[0]


        if CTRL_PNL['depth_map_output'] == True:
            # break things up into sub batches and pass through the mesh
            num_normal_sub_batches = current_batch_size / self.meshDepthLib.N
            if current_batch_size % self.meshDepthLib.N != 0:
                sub_batch_incr_list = num_normal_sub_batches * [self.meshDepthLib.N] + [
                    current_batch_size % self.meshDepthLib.N]
            else:
                sub_batch_incr_list = num_normal_sub_batches * [self.meshDepthLib.N]
            start_incr, end_incr = 0, 0

            #print len(sub_batch_incr_list), current_batch_size

            for sub_batch_incr in sub_batch_incr_list:
                end_incr += sub_batch_incr
                verts_sub, J_est_sub, targets_est_sub = self.meshDepthLib.HMR(gender_switch, betas_est,
                                                                              Rs_est, root_shift_est,
                                                                              start_incr, end_incr, self.GPU)
                if start_incr == 0:
                    verts = verts_sub.clone()
                    J_est = J_est_sub.clone()
                    targets_est = targets_est_sub.clone()
                else:
                    verts = torch.cat((verts, verts_sub), dim=0)
                    J_est = torch.cat((J_est, J_est_sub), dim=0)
                    targets_est = torch.cat((targets_est, targets_est_sub), dim=0)
                start_incr += sub_batch_incr

            bed_ang_idx = -1
            if CTRL_PNL['incl_ht_wt_channels'] == True: bed_ang_idx -= 2
            bed_angle_batch = torch.mean(images[:, bed_ang_idx, 1:3, 0], dim=1)

            OUTPUT_DICT['batch_mdm_est'], OUTPUT_DICT['batch_cm_est'] = self.meshDepthLib.PMR(verts, bed_angle_batch,
                                                                                            CTRL_PNL['mesh_bottom_dist'])

            OUTPUT_DICT['batch_mdm_est'] = OUTPUT_DICT['batch_mdm_est'].type(self.dtype)
            OUTPUT_DICT['batch_cm_est'] = OUTPUT_DICT['batch_cm_est'].type(self.dtype)

            verts_red = torch.stack([verts[:, 1325, :],
                                     verts[:, 336, :],  # head
                                     verts[:, 1032, :],  # l knee
                                     verts[:, 4515, :],  # r knee
                                     verts[:, 1374, :],  # l ankle
                                     verts[:, 4848, :],  # r ankle
                                     verts[:, 1739, :],  # l elbow
                                     verts[:, 5209, :],  # r elbow
                                     verts[:, 1960, :],  # l wrist
                                     verts[:, 5423, :]]).permute(1, 0, 2)  # r wrist

            verts_offset = verts_red.clone().detach().cpu()
            verts_offset = torch.Tensor(verts_offset.numpy()).type(self.dtype)

        else:
            shapedirs = torch.bmm(gender_switch, self.meshDepthLib.shapedirs_repeat[0:current_batch_size, :, :])\
                             .view(current_batch_size, self.meshDepthLib.B, self.meshDepthLib.R*self.meshDepthLib.D)

            betas_shapedirs_mult = torch.bmm(betas_est.unsqueeze(1), shapedirs)\
                                        .squeeze(1)\
                                        .view(current_batch_size, self.meshDepthLib.R, self.meshDepthLib.D)

            v_template = torch.bmm(gender_switch, self.meshDepthLib.v_template_repeat[0:current_batch_size, :, :])\
                              .view(current_batch_size, self.meshDepthLib.R, self.meshDepthLib.D)

            v_shaped = betas_shapedirs_mult + v_template

            J_regressor_repeat = torch.bmm(gender_switch, self.meshDepthLib.J_regressor_repeat[0:current_batch_size, :, :])\
                                      .view(current_batch_size, self.meshDepthLib.R, 24)

            Jx = torch.bmm(v_shaped[:, :, 0].unsqueeze(1), J_regressor_repeat).squeeze(1)
            Jy = torch.bmm(v_shaped[:, :, 1].unsqueeze(1), J_regressor_repeat).squeeze(1)
            Jz = torch.bmm(v_shaped[:, :, 2].unsqueeze(1), J_regressor_repeat).squeeze(1)


            J_est = torch.stack([Jx, Jy, Jz], dim=2)  # these are the joint locations with home pose (pose is 0 degree on all angles)
            #J_est = J_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)


            targets_est, A_est = KinematicsLib().batch_global_rigid_transformation(Rs_est, J_est, self.meshDepthLib.parents,
                                                                                   self.GPU, rotate_base=False)

            targets_est = targets_est - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)

            # assemble a reduced form of the transformed mesh
            v_shaped_red = torch.stack([v_shaped[:, self.verts_list[0], :],
                                        v_shaped[:, self.verts_list[1], :],  # head
                                        v_shaped[:, self.verts_list[2], :],  # l knee
                                        v_shaped[:, self.verts_list[3], :],  # r knee
                                        v_shaped[:, self.verts_list[4], :],  # l ankle
                                        v_shaped[:, self.verts_list[5], :],  # r ankle
                                        v_shaped[:, self.verts_list[6], :],  # l elbow
                                        v_shaped[:, self.verts_list[7], :],  # r elbow
                                        v_shaped[:, self.verts_list[8], :],  # l wrist
                                        v_shaped[:, self.verts_list[9], :]]).permute(1, 0, 2)  # r wrist
            pose_feature = (Rs_est[:, 1:, :, :]).sub(1.0, torch.eye(3).type(self.dtype)).view(-1, 207)
            posedirs_repeat = torch.bmm(gender_switch, self.meshDepthLib.posedirs_repeat[0:current_batch_size, :, :]) \
                .view(current_batch_size, 10 * self.meshDepthLib.D, 207) \
                .permute(0, 2, 1)
            v_posed = torch.bmm(pose_feature.unsqueeze(1), posedirs_repeat).view(-1, 10, self.meshDepthLib.D)
            v_posed = v_posed.clone() + v_shaped_red
            weights_repeat = torch.bmm(gender_switch, self.meshDepthLib.weights_repeat[0:current_batch_size, :, :]) \
                .squeeze(1) \
                .view(current_batch_size, 10, 24)
            T = torch.bmm(weights_repeat, A_est.view(current_batch_size, 24, 16)).view(current_batch_size, -1, 4, 4)
            v_posed_homo = torch.cat([v_posed, torch.ones(current_batch_size, v_posed.shape[1], 1).type(self.dtype)], dim=2)
            v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))


            verts = v_homo[:, :, :3, 0] - J_est[:, 0:1, :] + root_shift_est.unsqueeze(1)

            verts_offset = torch.Tensor(verts.clone().detach().cpu().numpy()).type(self.dtype)

            OUTPUT_DICT['batch_mdm_est'] = None
            OUTPUT_DICT['batch_cm_est'] = None


        #print verts[0:10], 'VERTS EST INIT'
        OUTPUT_DICT['verts'] = verts.clone().detach().cpu().numpy()

        targets_est_detached = torch.Tensor(targets_est.clone().detach().cpu().numpy()).type(self.dtype)
        synth_joint_addressed = [3, 15, 4, 5, 7, 8, 18, 19, 20, 21]
        for real_joint in range(10):
            verts_offset[:, real_joint, :] = verts_offset[:, real_joint, :] - targets_est_detached[:, synth_joint_addressed[real_joint], :]


        #here we need to the ground truth to make it a surface point for the mocap markers
        #if is_training == True:
        synth_real_switch_repeated = synth_real_switch.unsqueeze(1).repeat(1, 3)
        for real_joint in range(10):
            targets_est[:, synth_joint_addressed[real_joint], :] = synth_real_switch_repeated * targets_est[:, synth_joint_addressed[real_joint], :].clone() \
                                   + torch.add(-synth_real_switch_repeated, 1) * (targets_est[:, synth_joint_addressed[real_joint], :].clone() + verts_offset[:, real_joint, :])


        targets_est = targets_est.contiguous().view(-1, 72)

        OUTPUT_DICT['batch_targets_est'] = targets_est.data*1000. #after it comes out of the forward kinematics

        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, 100 + add_idx, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)


        #tweak this to change the lengths vector
        scores[:, 34+add_idx+OSA:106+add_idx+OSA] = torch.mul(targets_est[:, 0:72], 1.)

        scores[:, 0:10] = torch.mul(synth_real_switch.unsqueeze(1), torch.sub(scores[:, 0:10], betas))#*.2
        if CTRL_PNL['full_body_rot'] == True:
            scores[:, 10:16] = scores[:, 13:19].clone()
            if self.loss_vector_type == 'anglesEU':
                scores[:, 10:13] = scores[:, 10:13].clone() - torch.cos(KinematicsLib().batch_euler_angles_from_dir_cos_angles(angles_gt[:, 0:3].view(-1, 1, 3).clone()).contiguous().view(-1, 3))
                scores[:, 13:16] = scores[:, 13:16].clone() - torch.sin(KinematicsLib().batch_euler_angles_from_dir_cos_angles(angles_gt[:, 0:3].view(-1, 1, 3).clone()).contiguous().view(-1, 3))
            elif self.loss_vector_type == 'anglesDC':
                scores[:, 10:13] = scores[:, 10:13].clone() - torch.cos(angles_gt[:, 0:3].clone())
                scores[:, 13:16] = scores[:, 13:16].clone() - torch.sin(angles_gt[:, 0:3].clone())

            #print euler_root_rot_gt[0, :], 'body rot angles gt'

        #compare the output angles to the target values
        if reg_angles == True:
            if self.loss_vector_type == 'anglesDC':
                scores[:, 34+OSA:106+OSA] = angles_gt.clone().view(-1, 72) - scores[:, 13+OSA:85+OSA]
                scores[:, 34+OSA:106+OSA] = torch.mul(synth_real_switch.unsqueeze(1), torch.sub(scores[:, 34+OSA:106+OSA], angles_gt.clone().view(-1, 72)))

            elif self.loss_vector_type == 'anglesEU':
                scores[:, 34+OSA:106+OSA] = KinematicsLib().batch_euler_angles_from_dir_cos_angles(angles_gt.view(-1, 24, 3).clone()).contiguous().view(-1, 72) - scores[:, 13+OSA:85+OSA]

            scores[:, 34+OSA:106+OSA] = torch.mul(synth_real_switch.unsqueeze(1), scores[:, 34+OSA:106+OSA].clone())



        #compare the output joints to the target values

        scores[:, 34+add_idx+OSA:106+add_idx+OSA] = targets[:, 0:72]/1000 - scores[:, 34+add_idx+OSA:106+add_idx+OSA]
        scores[:, 106+add_idx+OSA:178+add_idx+OSA] = ((scores[:, 34+add_idx+OSA:106+add_idx+OSA].clone())+0.0000001).pow(2)


        for joint_num in range(24):
            if joint_num in [0, 1, 2, 6, 9, 10, 11, 12, 13, 14, 16, 17, 22, 23]: #torso is 3 but forget training it
                scores[:, 10+joint_num+OSA] = torch.mul(synth_real_switch,
                                                    (scores[:, 106+add_idx+joint_num*3+OSA] +
                                                     scores[:, 107+add_idx+joint_num*3+OSA] +
                                                     scores[:, 108+add_idx+joint_num*3+OSA]).sqrt())

            else:
                scores[:, 10+joint_num+OSA] = (scores[:, 106+add_idx+joint_num*3+OSA] +
                                           scores[:, 107+add_idx+joint_num*3+OSA] +
                                           scores[:, 108+add_idx+joint_num*3+OSA]).sqrt()


        scores = scores.unsqueeze(0)
        scores = scores.unsqueeze(0)
        scores = F.pad(scores, (0, -151, 0, 0))
        scores = scores.squeeze(0)
        scores = scores.squeeze(0)


        scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1/1.728158146914805))#1.7312621950698526)) #weight the betas by std
        if CTRL_PNL['full_body_rot'] == True:
            scores[:, 10:16] = torch.mul(scores[:, 10:16].clone(), (1/0.3684988513298487))#0.2130542427733348)*np.pi) #weight the body rotation by the std
        scores[:, 10+OSA:34+OSA] = torch.mul(scores[:, 10+OSA:34+OSA].clone(), (1/0.1752780723422608))#0.1282715100608753)) #weight the 24 joints by std
        if reg_angles == True: scores[:, 34+OSA:106+OSA] = torch.mul(scores[:, 34+OSA:106+OSA].clone(), (1/0.29641429463719227))#0.2130542427733348)) #weight the angles by how many there are

        #scores[:, 0:10] = torch.mul(scores[:, 0:10].clone(), (1./10)) #weight the betas by how many betas there are
        #scores[:, 10:34] = torch.mul(scores[:, 10:34].clone(), (1./24)) #weight the joints by how many there are
        #if reg_angles == True: scores[:, 34:106] = torch.mul(scores[:, 34:106].clone(), (1./72)) #weight the angles by how many there are


        return scores, OUTPUT_DICT

