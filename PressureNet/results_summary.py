#!/usr/bin/env python
import numpy as np
import cv2
import os.path as osp
import pickle
import time
import imutils
import math
import cPickle as pkl
import os
SHORT = False

import rospy



# from hrl_lib.util import load_pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



if __name__ == '__main__':


    import optparse

    p = optparse.OptionParser()

    p.add_option('--hd', action='store_true', dest='hd', default=False,
                 help='Read and write to data on an external harddrive.')

    p.add_option('--pose_type', action='store', type='string', dest='pose_type', default='none',
                 help='Choose a pose type, either `prescribed` or `p_select`.')

    p.add_option('--pmr', action='store_true', dest='pmr', default=False,
                 help='Run PMR on input plus precomputed spatial maps.')

    p.add_option('--small', action='store_true', dest='small', default=False,
                 help='Make the dataset 1/4th of the original size.')

    p.add_option('--htwt', action='store_true', dest='htwt', default=False,
                 help='Include height and weight info on the input.')

    p.add_option('--calnoise', action='store_true', dest='calnoise', default=False,
                 help='Apply calibration noise to the input to facilitate sim to real transfer.')

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

    p.add_option('--align_procr', action='store_true', dest='align_procr', default=False,
                 help='Align procrustes. Only works on synthetic data.')
    
    opt, args = p.parse_args()


    RESULT_TYPE = "synth"

    if RESULT_TYPE == "real":

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
                            "S165", #at least 3 pc corrupted
                            "S170",
                            "S179",
                            "S184",
                            "S187",
                            "S188", #1 bad prone posture classified as supine, 2 pc corrupted
                            "S196",]
        #participant_list=["S188"]

        general = [[]]
        general_plo = [[]]
        general_supine = [[]]
        general_plo_supine = [[]]
        hands_behind_head = [[]]
        prone_hands_up = [[]]
        crossed_legs = [[]]
        straight_limbs = [[]]


        if opt.small == True:
            NETWORK_2 = "46000ct_"
            DATA_QUANT = "46K"
        else:
            NETWORK_2 = "184000ct_"
            DATA_QUANT = "184K"


        if opt.go200 == True:
            NETWORK_2 += "128b_x1pm_tnh"
        elif opt.pmr == True:
            NETWORK_2 += "128b_x1pm_0.5rtojtdpth_depthestin_angleadj_tnh"
        else:
            NETWORK_2 += "128b_x1pm_angleadj_tnh"

        if opt.htwt == True:
            NETWORK_2 += "_htwt"
        if opt.calnoise == True:
            NETWORK_2 += "_clns20p"
        if opt.loss_root == True:
            NETWORK_2 += "_rt"
        if opt.omit_cntct_sobel == True:
            NETWORK_2 += "_ocs"
        if opt.use_hover == True:
            NETWORK_2 += "_uh"
        if opt.half_shape_wt == True:
            NETWORK_2 += "_hsw"

        if opt.hd == True:
            FILEPATH_PREFIX = '/media/henry/multimodal_data_2/data_BR'
        else:
            FILEPATH_PREFIX = '../../../data_BR'

        #NETWORK_2 = "NONE-200e"


        if opt.pose_type == 'prescribed':
            POSE_TYPE = "2"
        elif opt.pose_type == 'p_select':
            POSE_TYPE = "1"
        else:
            print "Please choose a pose type - either prescribed poses, " \
                  "'--pose_type prescribed', or participant selected poses, '--pose_type p_select'."
            sys.exit()







        recall_list = []
        precision_list = []
        overlap_d_err_list = []
        v_limb_to_gt_err_list = []
        v_to_gt_err_list = []
        gt_to_v_err_list = []



        testing_data_sz_ct = 0

        for participant in participant_list:


            #if participant in ['S145']: continue

            participant_directory = FILEPATH_PREFIX+"/real/"+participant
            participant_info = load_pickle(participant_directory + "/participant_info_red.p")

            for key in participant_info:
                print "key: ", key

            #if participant_info['gender'] == 'f': continue #use this to test gender partitions


           # pose_type_list = participant_info['pose_type']
            #if participant_info['gender'] == 'f': continue

            print "participant directory: ", participant_directory

            current_results_dict = load_pickle("/media/henry/multimodal_data_2/data_BR/final_results/"+NETWORK_2+"/results_real_"
                                                   +participant+"_"+POSE_TYPE+"_"+NETWORK_2+".p")

            print "/media/henry/multimodal_data_2/data_BR/final_results/"+NETWORK_2+"/results_real_"+participant+"_"+POSE_TYPE+"_"+NETWORK_2+".p"
            #for entry in current_results_dict:
            #    print "entry: ", entry

            #precision =
            #print participant

            #print len(current_results_dict['recall'])
            print len(current_results_dict['recall'])

            #to test posture
            idx_num = -1

            if POSE_TYPE == "1":
                num_ims = 5
            elif POSE_TYPE == "2":
                num_ims = 48

            recall_list_curr = []
            precision_list_curr = []
            overlap_d_err_list_curr = []
            v_limb_to_gt_err_list_curr = []
            v_to_gt_err_list_curr = []
            gt_to_v_err_list_curr = []

            for i in range(num_ims):


                #partition_type = pose_type_list[i]
               # print partition_type


                if participant == "S114" and POSE_TYPE == "2"  and i in [26, 29]:
                    print "skipping", i#, partition_type
                    continue #these don't have point clouds
                elif participant == "S165" and POSE_TYPE == "2" and i in [1, 3, 15]:
                    print "skipping", i#, partition_type
                    continue #these don't have point clouds
                elif participant == "S188" and POSE_TYPE == "2"  and i in [5, 17, 21]:
                    print "skipping", i#, partition_type
                    continue
                elif participant == "S145" and POSE_TYPE == "1" and i in [0]:
                    print "skipping", i
                    continue

                else: idx_num += 1

                #if POSE_TYPE == "1":
                #    participant_info['p_select_pose_type'].append(p_select_pose_type[i])
                #if POSE_TYPE == "2":
                #    participant_info['prescribed_pose_type'].append(participant_info_orig['pose_type'][i])


                #body_roll_rad = current_results_dict['body_roll_rad'][idx_num]

                #if partition_type in ['phu']:

                #if participant_info['p_select_pose_type'][idx_num] not in ['prone']: continue #use this to test partitions
                #if participant_info['prescribed_pose_type'][idx_num] not in ['supine', 'supine_plo']: continue #use this to test partitions
                #if participant_info['prescribed_pose_type'][idx_num] not in ['rollpi', 'rollpi_plo']: continue #use this to test partitions
                #if participant_info['prescribed_pose_type'][idx_num] not in ['sl']: continue #use this to test partitions


                if POSE_TYPE == "1":
                    print i, idx_num, participant_info['p_select_pose_type'][idx_num]
                elif POSE_TYPE == "2":
                    print i, idx_num, participant_info['prescribed_pose_type'][idx_num]



                recall_list_curr.append(current_results_dict['recall'][idx_num])
                precision_list_curr.append(current_results_dict['precision'][idx_num])
                overlap_d_err_list_curr.append( current_results_dict['overlap_d_err'][idx_num])
                v_limb_to_gt_err_list_curr.append(current_results_dict['v_limb_to_gt_err'][idx_num])
                v_to_gt_err_list_curr.append(current_results_dict['v_to_gt_err'][idx_num])
                gt_to_v_err_list_curr.append(current_results_dict['gt_to_v_err'][idx_num])
                testing_data_sz_ct += 1



            recall_list.append(np.mean(recall_list_curr))
            precision_list.append(np.mean(precision_list_curr))
            overlap_d_err_list.append(np.mean(overlap_d_err_list_curr))
            v_limb_to_gt_err_list.append(np.mean(v_limb_to_gt_err_list_curr))
            v_to_gt_err_list.append(np.mean(v_to_gt_err_list_curr))
            gt_to_v_err_list.append(np.mean(gt_to_v_err_list_curr))


            #print  curr_gt_to_v_err
            #break

            #height, weight = get_heightweight_from_betas(current_results_dict['betas'])

            #print "LENGTH: ", len(participant_info['p_select_pose_type'])
            #print "LENGTH: ", len(participant_info['prescribed_pose_type'])
            #pkl.dump(participant_info, open(participant_directory_orig + "/participant_info_red.p", 'wb'))


        #print "average precision: ", np.mean(precision_list)
       # print "average recall: ", np.mean(recall_list)
        #print "average overlap depth err: ", np.mean(overlap_d_err_list)

        print "testing data size ct: ", testing_data_sz_ct
        print "average v to gt err: ", np.mean(v_to_gt_err_list)*100
        print "average gt to v err: ", np.mean(gt_to_v_err_list)*100
        print "mean 3D err: ", np.mean([np.mean(v_to_gt_err_list), np.mean(gt_to_v_err_list)])




    elif RESULT_TYPE == "synth":


        if opt.small == True:
            NETWORK_2 = "46000ct_"
            DATA_QUANT = "46K"
        else:
            NETWORK_2 = "184000ct_"
            DATA_QUANT = "184K"


        if opt.go200 == True:
            NETWORK_2 += "128b_x1pm_tnh"
        elif opt.pmr == True:
            NETWORK_2 += "128b_x1pm_0.5rtojtdpth_depthestin_angleadj_tnh"
        else:
            NETWORK_2 += "128b_x1pm_angleadj_tnh"

        if opt.htwt == True:
            NETWORK_2 += "_htwt"
        if opt.calnoise == True:
            NETWORK_2 += "_clns20p"
        if opt.loss_root == True:
            NETWORK_2 += "_rt"
        if opt.omit_cntct_sobel == True:
            NETWORK_2 += "_ocs"
        if opt.use_hover == True:
            NETWORK_2 += "_uh"
        if opt.half_shape_wt == True:
            NETWORK_2 += "_hsw"
        if opt.align_procr == True:
            NETWORK_2 += "_ap"
#
        #NETWORK_2 = "BASELINE"


        filename_list = ["test_rollpi_f_lay_set23to24_1500",
                         "test_rollpi_m_lay_set23to24_1500",
                         "test_rollpi_plo_f_lay_set23to24_1500",
                         "test_rollpi_plo_m_lay_set23to24_1500",
                         "test_roll0_f_lay_set14_500",
                         "test_roll0_m_lay_set14_500",
                         "test_roll0_plo_f_lay_set14_500",
                         "test_roll0_plo_m_lay_set14_500",
                         "test_roll0_plo_hbh_f_lay_set4_500",
                         "test_roll0_plo_hbh_m_lay_set1_500",
                         "test_roll0_plo_phu_f_lay_set1pa3_500",
                         "test_roll0_plo_phu_m_lay_set1pa3_500",
                         "test_roll0_sl_f_lay_set1both_500",
                         "test_roll0_sl_m_lay_set1both_500",
                         "test_roll0_xl_f_lay_set1both_500",
                         "test_roll0_xl_m_lay_set1both_500"
                            ]
        import math
        recall_avg_list = []
        precision_avg_list = []
        overlap_d_err_avg_list = []
        v_to_gt_err_avg_list = []
        gt_to_v_err_avg_list = []
        joint_err_list = []
        v2v_err_list = []

        for filename in filename_list:
            #current_results_dict = load_pickle("/media/henry/multimodal_data_1/data/final_results/"+DATA_QUANT+"_"
            #                                   +NETWORK_2+"/results_synth_"+DATA_QUANT+"_"+filename+NETWORK_2+".p")
            #current_results_dict = load_pickle("/media/henry/multimodal_data_2/data/final_results/"+NETWORK_2+"/results_synth_"+filename+NETWORK_2+".p")
            #current_results_dict = load_pickle("/home/henry/data/final_results/"+NETWORK_2+"/results_synth_"+filename+NETWORK_2+".p")
            current_results_dict = load_pickle("/media/henry/multimodal_data_2/data_BR/final_results/"+NETWORK_2+"/results_synth_"+filename+".p")
            for entry in current_results_dict:
                print entry
            #print current_results_dict['j_err'], 'j err'
            #precision =


            for i in range(len(current_results_dict['v_to_gt_err'])):


                #recall_avg_list.append(current_results_dict['recall'][i])

                #if math.isnan(float(current_results_dict['precision'][i])): print "nan precision"
                #else: precision_avg_list.append(current_results_dict['precision'][i])


                #if math.isnan(float(current_results_dict['overlap_d_err'][i])): print "nan"
                #else: overlap_d_err_avg_list.append(current_results_dict['overlap_d_err'][i])

                v_to_gt_err_avg_list.append(current_results_dict['v_to_gt_err'][i])

                gt_to_v_err_avg_list.append(current_results_dict['gt_to_v_err'][i])
                #print  curr_gt_to_v_err

                v2v_err_list.append(current_results_dict['v2v_err'][i])

                #print current_results_dict['j_err']
                joint_err_list.append(current_results_dict['j_err'][i])
                #print np.shape(joint_err_list)

           # break

        #print np.min(overlap_d_err_avg_list), np.max(overlap_d_err_list)
        print np.shape(joint_err_list)
        print
        print len(v_to_gt_err_avg_list)

        #print "average precision: ", np.mean(precision_avg_list)
        #print "average recall: ", np.mean(recall_avg_list)
        #print "average overlap depth err: ", np.mean(overlap_d_err_avg_list)
        print "average v to gt err: ", np.mean(v_to_gt_err_avg_list)*100
        print "average gt to v err: ", np.mean(gt_to_v_err_avg_list)*100
        print "mean 3DVPE err: ", np.mean([np.mean(v_to_gt_err_avg_list), np.mean(gt_to_v_err_avg_list)])*100
        print "MPJPE: ", np.mean(joint_err_list)*100
        print "v2v: ", np.mean(v2v_err_list)*100
