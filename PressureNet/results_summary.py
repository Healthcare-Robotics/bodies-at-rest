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


def get_heightweight_from_betas(betas):

    height, weight = 0, 0


    return height, weight

if __name__ == '__main__':
    RESULT_TYPE = "real"

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


        #NETWORK_2 = "1.0rtojtdpth_angleadj_tnhFIXN_htwt_calnoise"
        NETWORK_2 = "184000_128b_x5pmult_0.5rtojtdpth_depthestin_angleadj_tnhFIXN_calnoise"
        #NETWORK_2 = "NONE-200e"
        #NETWORK_2 = "BASELINE"

        POSE_TYPE = "1"
        DATA_QUANT = "184K"

        recall_list = []
        precision_list = []
        overlap_d_err_list = []
        v_limb_to_gt_err_list = []
        v_to_gt_err_list = []
        gt_to_v_err_list = []


        for participant in participant_list:
            participant_directory = "/media/henry/multimodal_data_2/CVPR2020_study/"+participant
            participant_info = load_pickle(participant_directory + "/participant_info.p")

            pose_type_list = participant_info['pose_type']
            #if participant_info['gender'] == 'f': continue

            print "participant directory: ", participant_directory

            current_results_dict = load_pickle("/media/henry/multimodal_data_2/data_BR/final_results/"+NETWORK_2+"/results_real_"
                                                   +participant+"_"+POSE_TYPE+"_"+NETWORK_2+".p")

            print "/media/henry/multimodal_data_2/data_BR/final_results/"+NETWORK_2+"/results_real_"+participant+"_"+POSE_TYPE+"_"+NETWORK_2+".p"
            #for entry in current_results_dict:
            #    print entry

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

                partition_type = pose_type_list[i]
                print partition_type


                if participant == "S114" and POSE_TYPE == "2"  and i in [26, 29]:
                    print "skipping", i, partition_type
                    continue #these don't have point clouds
                elif participant == "S165" and POSE_TYPE == "2" and i in [1, 3, 15]:
                    print "skipping", i, partition_type
                    continue #these don't have point clouds
                elif participant == "S188" and POSE_TYPE == "2"  and i in [5, 17, 21]:
                    print "skipping", i, partition_type
                    continue
                elif participant == "S145" and POSE_TYPE == "1" and i in [0]:
                    print "skipping", i
                    continue

                else: idx_num += 1

                #body_roll_rad = current_results_dict['body_roll_rad'][idx_num]

                #if partition_type in ['phu']:
                recall_list_curr.append(current_results_dict['recall'][idx_num])
                precision_list_curr.append(current_results_dict['precision'][idx_num])
                overlap_d_err_list_curr.append( current_results_dict['overlap_d_err'][idx_num])
                v_limb_to_gt_err_list_curr.append(current_results_dict['v_limb_to_gt_err'][idx_num])
                v_to_gt_err_list_curr.append(current_results_dict['v_to_gt_err'][idx_num])
                gt_to_v_err_list_curr.append(current_results_dict['gt_to_v_err'][idx_num])
                '''
                if idx_num in [3] and participant not in ["S145", "S188", "S140"]:
                    #if partition_type in ['supine_plo', 'rollpi_plo']:
                    recall_list.append(current_results_dict['recall'][idx_num])
                    precision_list.append(current_results_dict['precision'][idx_num])
                    overlap_d_err_list.append( current_results_dict['overlap_d_err'][idx_num])
                    v_limb_to_gt_err_list.append(current_results_dict['v_limb_to_gt_err'][idx_num])
                    v_to_gt_err_list.append(current_results_dict['v_to_gt_err'][idx_num])
                    gt_to_v_err_list.append(current_results_dict['gt_to_v_err'][idx_num]) #for 140 get supine from last
                elif idx_num in [1] and participant in ["S188"]:
                    recall_list.append(current_results_dict['recall'][idx_num])
                    precision_list.append(current_results_dict['precision'][idx_num])
                    overlap_d_err_list.append( current_results_dict['overlap_d_err'][idx_num])
                    v_limb_to_gt_err_list.append(current_results_dict['v_limb_to_gt_err'][idx_num])
                    v_to_gt_err_list.append(current_results_dict['v_to_gt_err'][idx_num])
                    gt_to_v_err_list.append(current_results_dict['gt_to_v_err'][idx_num]) #for 140 get supine from last
                elif idx_num in [2] and participant in ["S145"]:
                    recall_list.append(current_results_dict['recall'][idx_num])
                    precision_list.append(current_results_dict['precision'][idx_num])
                    overlap_d_err_list.append( current_results_dict['overlap_d_err'][idx_num])
                    v_limb_to_gt_err_list.append(current_results_dict['v_limb_to_gt_err'][idx_num])
                    v_to_gt_err_list.append(current_results_dict['v_to_gt_err'][idx_num])
                    gt_to_v_err_list.append(current_results_dict['gt_to_v_err'][idx_num]) #for 140 get supine from last
                elif idx_num in [2] and participant in ["S140"]:
                    recall_list.append(current_results_dict['recall'][idx_num])
                    precision_list.append(current_results_dict['precision'][idx_num])
                    overlap_d_err_list.append( current_results_dict['overlap_d_err'][idx_num])
                    v_limb_to_gt_err_list.append(current_results_dict['v_limb_to_gt_err'][idx_num])
                    v_to_gt_err_list.append(current_results_dict['v_to_gt_err'][idx_num])
                    gt_to_v_err_list.append(current_results_dict['gt_to_v_err'][idx_num]) #for 140 get supine from last'''


            recall_list.append(np.mean(recall_list_curr))
            precision_list.append(np.mean(precision_list_curr))
            overlap_d_err_list.append(np.mean(overlap_d_err_list_curr))
            v_limb_to_gt_err_list.append(np.mean(v_limb_to_gt_err_list_curr))
            v_to_gt_err_list.append(np.mean(v_to_gt_err_list_curr))
            gt_to_v_err_list.append(np.mean(gt_to_v_err_list_curr))




            #print  curr_gt_to_v_err
            print len(v_to_gt_err_list), 'ct list'
            #break

            #height, weight = get_heightweight_from_betas(current_results_dict['betas'])


        #print "average precision: ", np.mean(precision_list)
       # print "average recall: ", np.mean(recall_list)
        #print "average overlap depth err: ", np.mean(overlap_d_err_list)
        print "average v to gt err: ", np.mean(v_to_gt_err_list)*100
        print "average gt to v err: ", np.mean(gt_to_v_err_list)*100
        print "mean 3D err: ", np.mean([np.mean(v_to_gt_err_list), np.mean(gt_to_v_err_list)])



    elif RESULT_TYPE == "synth":


        #NETWORK_1 = "1.0rtojtdpth_depthestin_angleadj_tnhFIXN_htwt_calnoise"
        NETWORK_2 = "0.5rtojtdpth_depthestin_angleadj_tnhFIXN"
        #NETWORK_2 = "1.0rtojtdpth_angleadj_tnhFIXN_calnoise"
        #NETWORK_2 = "NONE-200e"
        #NETWORK_2 = "BASELINE"
        DATA_QUANT = "184K"

        filename_list = ["test_rollpi_f_lay_set23to24_1500_",
                         "test_rollpi_m_lay_set23to24_1500_",
                         "test_rollpi_plo_f_lay_set23to24_1500_",
                         "test_rollpi_plo_m_lay_set23to24_1500_",
                         "test_roll0_f_lay_set14_500_",
                         "test_roll0_m_lay_set14_500_",
                         "test_roll0_plo_f_lay_set14_500_",
                         "test_roll0_plo_m_lay_set14_500_",
                         "test_roll0_plo_hbh_f_lay_set4_500_",
                         "test_roll0_plo_hbh_m_lay_set1_500_",
                         "test_roll0_plo_phu_f_lay_set1pa3_500_",
                         "test_roll0_plo_phu_m_lay_set1pa3_500_",
                         "test_roll0_sl_f_lay_set1both_500_",
                         "test_roll0_sl_m_lay_set1both_500_",
                         "test_roll0_xl_f_lay_set1both_500_",
                         "test_roll0_xl_m_lay_set1both_500_"
                            ]
        import math
        recall_avg_list = []
        precision_avg_list = []
        overlap_d_err_avg_list = []
        v_to_gt_err_avg_list = []
        gt_to_v_err_avg_list = []
        joint_err_list = []

        for filename in filename_list:
            #current_results_dict = load_pickle("/media/henry/multimodal_data_1/data/final_results/"+DATA_QUANT+"_"
            #                                   +NETWORK_2+"/results_synth_"+DATA_QUANT+"_"+filename+NETWORK_2+".p")
            #current_results_dict = load_pickle("/media/henry/multimodal_data_2/data/final_results/"+NETWORK_2+"/results_synth_"+filename+NETWORK_2+".p")
            #current_results_dict = load_pickle("/home/henry/data/final_results/"+NETWORK_2+"/results_synth_"+filename+NETWORK_2+".p")
            current_results_dict = load_pickle("/media/henry/multimodal_data_2/data/final_results/"+DATA_QUANT+"_"+NETWORK_2+"/results_synth_"+DATA_QUANT+"_"+filename+NETWORK_2+".p")
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
        print "mean 3D err: ", np.mean([np.mean(v_to_gt_err_avg_list), np.mean(gt_to_v_err_avg_list)])*100
        print "mean joint err: ", np.mean(joint_err_list)*100