#!/usr/bin/env python
import numpy as np
import cv2

HEAD_BEND_TAXEL = 41 #measured from the bottom of the pressure mat
LEGS_BEND1_TAXEL = 37 #measured from the bottom of the pressure mat
LEGS_BEND2_TAXEL = 20 #measured from the bottom of the pressure mat
INFLATION_FACTOR = 1.07 #we use this because even after calibrating the cameras things are a bit out of tune
HENRY_EVANS = True


class ArTagLib():
    def __init__(self, head_taxel_offset = 0):
        self.head_taxel_offset = head_taxel_offset

    def color_2D_markers(self, markers_c, new_K_kin):

        if HENRY_EVANS == True:
            if markers_c[0] is None:
                markers_c = [np.array([ 1.22181549, -0.42804009,  1.587955  ]), np.array([1.2176198 , 0.37355771, 1.59150983]), np.array([-1.15200846, -0.4033413 ,  1.52632663]), np.array([-1.16417695,  0.39847735,  1.56105968])]


        f_xc, f_yc, c_xc, c_yc = new_K_kin[0, 0], new_K_kin[1, 1], new_K_kin[0, 2], new_K_kin[1, 2]

        # Get the marker points in 2D on the color image
        u_c = [None, None, None, None]
        v_c = [None, None, None, None]
        for marker in range(0, 4):
            try:
                u_c[marker] = f_xc * markers_c[marker][0] / markers_c[marker][2] + c_xc
                v_c[marker] = f_yc * markers_c[marker][1] / markers_c[marker][2] + c_yc
            except:
                pass
        return u_c, v_c

    def color_2D_markers_drop(self, markers_c, new_K_kin):

        if HENRY_EVANS == True:
            if markers_c[0] is None:
                markers_c = [np.array([ 1.22181549, -0.42804009,  1.587955  ]), np.array([1.2176198 , 0.37355771, 1.59150983]), np.array([-1.15200846, -0.4033413 ,  1.52632663]), np.array([-1.16417695,  0.39847735,  1.56105968])]

        f_xc, f_yc, c_xc, c_yc = new_K_kin[0, 0], new_K_kin[1, 1], new_K_kin[0, 2], new_K_kin[1, 2]

        # Get the marker points in 2D on the color image
        u_c_drop = [None, None, None, None]
        v_c_drop = [None, None, None, None]
        markers_c_drop = []
        for marker in range(0, 4):
            try:
                markers_c_drop.append(markers_c[marker] + np.array([0., 0., 0.1]))
                u_c_drop[marker] = f_xc * markers_c_drop[marker][0] / markers_c_drop[marker][2] + c_xc
                v_c_drop[marker] = f_yc * markers_c_drop[marker][1] / markers_c_drop[marker][2] + c_yc
            except:
                markers_c_drop.append(None)
        return u_c_drop, v_c_drop, markers_c_drop

    def thermal_2D_markers(self, markers_c, new_K_thr, R_thermal, T_thermal):
        if HENRY_EVANS == True:
            if markers_c[0] is None:
                markers_c = [np.array([ 1.22181549, -0.42804009,  1.587955  ]), np.array([1.2176198 , 0.37355771, 1.59150983]), np.array([-1.15200846, -0.4033413 ,  1.52632663]), np.array([-1.16417695,  0.39847735,  1.56105968])]



        f_xt, f_yt, c_xt, c_yt = new_K_thr[0, 0], new_K_thr[1, 1], new_K_thr[0, 2], new_K_thr[1, 2]

        # Get the marker points in 2D on the thermal image
        u_t = [None, None, None, None]
        v_t = [None, None, None, None]
        markers_t = []
        for marker in range(0, 4):
            try:
                markers_t.append(np.matmul(R_thermal, np.array(markers_c[marker])) + T_thermal)

                u_t[marker] = f_xt * markers_t[marker][0] / markers_t[marker][2] + c_xt + 2
                v_t[marker] = f_yt * markers_t[marker][1] / markers_t[marker][2] + c_yt + 5
            except:
                markers_t.append(None)
        return u_t, v_t


    def p_mat_geom(self, markers_c_drop, new_K_kin, pressure_im_size_required, bed_state, half_w_half_l):

        if HENRY_EVANS == True:
            if markers_c_drop[0] is None:
                markers_c_drop = [[ 1.2241848 , -0.42903586,  1.69103686], [1.2176198 , 0.37355771, 1.69150983], [-1.15266852, -0.40349703,  1.62720605], [-1.16075505,  0.3974086 ,  1.65669859]]
                pressure_im_size_required = [622, 262]

        bed_head_angle = bed_state[0]
        bed_legs_angle = bed_state[2]

        f_xc, f_yc, c_xc, c_yc = new_K_kin[0, 0], new_K_kin[1, 1], new_K_kin[0, 2], new_K_kin[1, 2]

        # Get the geometry for sizing the pressure mat
        u_c_pmat = [None, None, None, None]  # [top, bottom, left(facing bed), right(facing bed)]
        v_c_pmat = [None, None, None, None]

        u_p_bend = [None, None, None, None, None, None] #top left and right corners (facing bed)
        v_p_bend = [None, None, None, None, None, None]

        self.u_p_bend_calib = [None, None]
        self.v_p_bend_calib = [None, None
                          ]
        markers_bend = []
        markers_bend_calib = []


        #basic idea of these if statements: if there aren't at least two markers showing where the two markers are
        #caddy-cornered from each other, then do not show the pressure image.
        if markers_c_drop[0] is not None and markers_c_drop[1] is not None:
            loc_top = np.mean(np.array([markers_c_drop[0], markers_c_drop[1]]), axis=0)
            half_w_half_l[0] = (markers_c_drop[1][1] - markers_c_drop[0][1])/2
        elif markers_c_drop[0] is None and markers_c_drop[1] is not None:
            loc_top = markers_c_drop[1] - [0., half_w_half_l[0], 0.]
        elif markers_c_drop[0] is not None and markers_c_drop[1] is None:
            loc_top = markers_c_drop[0] + [0., half_w_half_l[0], 0.]
        else:
            loc_top = np.array([None, None, None])

        if markers_c_drop[2] is not None and markers_c_drop[3] is not None:
            loc_bot = np.mean(np.array([markers_c_drop[2], markers_c_drop[3]]), axis=0)
            half_w_half_l[1] = (markers_c_drop[3][1] - markers_c_drop[2][1])/2
        elif markers_c_drop[2] is None and markers_c_drop[3] is not None:
            loc_bot = markers_c_drop[3] - [0., half_w_half_l[1], 0.]
        elif markers_c_drop[2] is not None and markers_c_drop[3] is None:
            loc_bot = markers_c_drop[2] + [0., half_w_half_l[1], 0.]
        else:
            loc_bot = np.array([None, None, None])

        if markers_c_drop[0] is not None and markers_c_drop[2] is not None:
            loc_left = np.mean(np.array([markers_c_drop[0], markers_c_drop[2]]), axis=0)
            half_w_half_l[2] = (markers_c_drop[0][0] - markers_c_drop[2][0])/2
        elif markers_c_drop[0] is None and markers_c_drop[2] is not None:
            loc_left = markers_c_drop[2] + [half_w_half_l[2], 0., 0.]
        elif markers_c_drop[0] is not None and markers_c_drop[2] is None:
            loc_left = markers_c_drop[0] - [half_w_half_l[2], 0., 0.]
        else:
            loc_left = np.array([None, None, None])

        if markers_c_drop[1] is not None and markers_c_drop[3] is not None:
            loc_right = np.mean(np.array([markers_c_drop[1], markers_c_drop[3]]), axis=0)
            half_w_half_l[3] = (markers_c_drop[1][0] - markers_c_drop[3][0])/2
        elif markers_c_drop[1] is None and markers_c_drop[3] is not None:
            loc_right = markers_c_drop[3] + [half_w_half_l[3], 0., 0.]
        elif markers_c_drop[1] is not None and markers_c_drop[3] is None:
            loc_right = markers_c_drop[1] - [half_w_half_l[3], 0., 0.]
        else:
            loc_right = np.array([None, None, None])

        try:
            mat_length = 0.0286*64*INFLATION_FACTOR
            mat_width = 0.0286*27*INFLATION_FACTOR
            bed_thickness = 0.15*INFLATION_FACTOR

            frame_bend_world_z = np.mean([loc_left[2],loc_right[2]])
            head_length = mat_length*(64 - HEAD_BEND_TAXEL)/64
            head_length_calib = mat_length*(64 - HEAD_BEND_TAXEL - self.head_taxel_offset)/64
            head_bend_wrt_world_x = np.mean([loc_left[0],loc_right[0]])+ mat_length*(HEAD_BEND_TAXEL - 32)/64
            legs1_length = mat_length*(LEGS_BEND1_TAXEL - LEGS_BEND2_TAXEL)/64
            legs1_bend_wrt_world_x = np.mean([loc_left[0],loc_right[0]])+ mat_length*(LEGS_BEND1_TAXEL - 32)/64
            legs2_length = mat_length*LEGS_BEND2_TAXEL/64
            legs2_bend_wrt_world_x = None #important because it's part of the kinematic chain so don't use this

            marker_to_mat_x = (loc_top[0] - loc_bot[0] - mat_length) / 2
            marker_to_mat_y = (loc_right[1] - loc_left[1] - mat_width) / 2
            u_c_pmat[0] = f_xc * (loc_top[0] - marker_to_mat_x) / loc_top[2] + c_xc
            v_c_pmat[0] = f_yc * loc_top[1] / loc_top[2] + c_yc
            u_c_pmat[1] = f_xc * (loc_bot[0] + marker_to_mat_x) / loc_bot[2] + c_xc
            v_c_pmat[1] = f_yc * loc_bot[1] / loc_bot[2] + c_yc
            u_c_pmat[2] = f_xc * loc_left[0] / loc_left[2] + c_xc
            v_c_pmat[2] = f_yc * (loc_left[1] + marker_to_mat_y) / loc_left[2] + c_yc
            u_c_pmat[3] = f_xc * loc_right[0] / loc_right[2] + c_xc
            v_c_pmat[3] = f_yc * (loc_right[1] - marker_to_mat_y) / loc_right[2] + c_yc
            pressure_im_size_required = np.rint(
                np.array([u_c_pmat[0] - u_c_pmat[1], v_c_pmat[3] - v_c_pmat[2]])).astype(np.uint16)

            #get the locations of the bent head of the bed and their respective 2D positions
            x_HeadL = head_bend_wrt_world_x + head_length*np.cos(np.deg2rad(bed_head_angle)) - bed_thickness*np.sin(np.deg2rad(bed_head_angle))
            y_HeadL = loc_left[1]
            z_HeadL = frame_bend_world_z - head_length*np.sin(np.deg2rad(bed_head_angle)) + bed_thickness*(1 - np.cos(np.deg2rad(bed_head_angle)))
            x_HeadR = head_bend_wrt_world_x + head_length*np.cos(np.deg2rad(bed_head_angle)) - bed_thickness*np.sin(np.deg2rad(bed_head_angle))
            y_HeadR = loc_right[1]
            z_HeadR = frame_bend_world_z - head_length*np.sin(np.deg2rad(bed_head_angle)) + bed_thickness*(1 - np.cos(np.deg2rad(bed_head_angle)))
            x_HeadL_calib = head_bend_wrt_world_x + head_length_calib*np.cos(np.deg2rad(bed_head_angle)) - bed_thickness*np.sin(np.deg2rad(bed_head_angle))
            y_HeadL_calib = loc_left[1]
            z_HeadL_calib = frame_bend_world_z - head_length_calib*np.sin(np.deg2rad(bed_head_angle)) + bed_thickness*(1 - np.cos(np.deg2rad(bed_head_angle)))
            x_HeadR_calib = head_bend_wrt_world_x + head_length_calib*np.cos(np.deg2rad(bed_head_angle)) - bed_thickness*np.sin(np.deg2rad(bed_head_angle))
            y_HeadR_calib = loc_right[1]
            z_HeadR_calib = frame_bend_world_z - head_length_calib*np.sin(np.deg2rad(bed_head_angle)) + bed_thickness*(1 - np.cos(np.deg2rad(bed_head_angle)))

            #get the locations of the legs position/fold #1 of the bed and their respective 2D positions
            x_Legs1L = legs1_bend_wrt_world_x - legs1_length*np.cos(np.deg2rad(bed_legs_angle))
            y_Legs1L = loc_left[1]
            z_Legs1L = frame_bend_world_z - legs1_length*np.sin(np.deg2rad(bed_legs_angle)) + bed_thickness*(1 - np.cos(np.deg2rad(bed_legs_angle)))
            x_Legs1R = legs1_bend_wrt_world_x - legs1_length*np.cos(np.deg2rad(bed_legs_angle))
            y_Legs1R = loc_right[1]
            z_Legs1R = frame_bend_world_z - legs1_length*np.sin(np.deg2rad(bed_legs_angle)) + bed_thickness*(1 - np.cos(np.deg2rad(bed_legs_angle)))

            #get the locations of the legs position at the very bottom of the legs e.g. lower corners of the mattress
            #first use laws of cosines and sines to get bed legs kinematic solutions
            l1 = 36.85 #segment attached to frame that bends up from primary legs angle
            l2 = 50.80 #segment of the lower legs between the connecting axles
            l3 = 19.05 #small bar at the bottom that the legs rotate about the frame by
            l4 = 69.85 #distance from small bar mounting to where the upper legs segment rotate about the frame by

            d1 = np.sqrt(l1*l1 + l4*l4 - 2*l1*l4*np.cos(np.deg2rad(bed_legs_angle + 5.7)))
            theta2_pp = np.rad2deg(np.arccos((l2*l2 + d1*d1 - l3*l3)/(2*l2*d1)))
            theta2 = theta2_pp + np.rad2deg(np.arccos((d1*d1 + l1*l1 - l4*l4)/(2*d1*l1)))
            theta2_p = theta2 - (90 - (bed_legs_angle))

            x_Legs2L = legs1_bend_wrt_world_x - legs1_length*np.cos(np.deg2rad(bed_legs_angle)) - legs2_length*np.sin(np.deg2rad(theta2_p)) - bed_thickness*np.cos(np.deg2rad(theta2_p))
            y_Legs2L = loc_left[1]
            z_Legs2L = frame_bend_world_z - legs1_length*np.sin(np.deg2rad(bed_legs_angle)) + legs2_length*np.cos(np.deg2rad(theta2_p)) + bed_thickness*(1 - np.sin(np.deg2rad(theta2_p)))
            x_Legs2R = legs1_bend_wrt_world_x - legs1_length*np.cos(np.deg2rad(bed_legs_angle)) - legs2_length*np.sin(np.deg2rad(theta2_p)) - bed_thickness*np.cos(np.deg2rad(theta2_p))
            y_Legs2R = loc_right[1]
            z_Legs2R = frame_bend_world_z - legs1_length*np.sin(np.deg2rad(bed_legs_angle)) + legs2_length*np.cos(np.deg2rad(theta2_p)) + bed_thickness*(1 - np.sin(np.deg2rad(theta2_p)))





            markers_bend.append([x_HeadL, y_HeadL, z_HeadL])
            markers_bend.append([x_HeadR, y_HeadR, z_HeadR])
            markers_bend.append([x_Legs1L, y_Legs1L, z_Legs1L])
            markers_bend.append([x_Legs1R, y_Legs1R, z_Legs1R])
            markers_bend.append([x_Legs2L, y_Legs2L, z_Legs2L])
            markers_bend.append([x_Legs2R, y_Legs2R, z_Legs2R])

            markers_bend_calib.append([x_HeadL_calib, y_HeadL_calib, z_HeadL_calib])
            markers_bend_calib.append([x_HeadR_calib, y_HeadR_calib, z_HeadR_calib])

            for marker in range(0, 6):
                u_p_bend[marker] = f_xc * markers_bend[marker][0] / markers_bend[marker][2] + c_xc
                v_p_bend[marker] = f_yc * markers_bend[marker][1] / markers_bend[marker][2] + c_yc

            for marker in range(0, 2):
                self.u_p_bend_calib[marker] = f_xc * markers_bend_calib[marker][0] / markers_bend_calib[marker][2] + c_xc
                self.v_p_bend_calib[marker] = f_yc * markers_bend_calib[marker][1] / markers_bend_calib[marker][2] + c_yc

        except: pass


        return pressure_im_size_required, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend, half_w_half_l




class VizLib():

    def assemble_A_2nx9(self, point, point_new, current = None):
        addition = np.array([[point[0], point[1], 1, 0, 0, 0, -point_new[0]*point[0], -point_new[0]*point[1], -point_new[0]],
                            [0, 0, 0, point[0], point[1], 1, -point_new[1]*point[0], -point_new[1]*point[1], -point_new[1]]])

        if current is None:
            return addition

        else:
            return np.concatenate((current, addition), axis = 0)

    def get_new_K_kin_homography(self, alpha_vert, alpha_horiz, new_K_kin, flip_vert = 1):
        #alpha_vert = 0.8
        #print alpha_vert, alpha_horiz, 'ALPHAs'

        p00 = [0, 0]
        p10 = [960, 0]
        p01 = [0, 540]
        p11 = [960, 540]

        #p00n = [(alpha_vert - 1) *480,  540 * (alpha_vert - 1)]
        #p10n = [960-(alpha_vert - 1) * 480, 540 * (alpha_vert - 1)]
        #p01n = [(1 - alpha_vert) * 480, 540 * alpha_vert]
        #p11n = [(1 + alpha_vert) * 480, 540 * alpha_vert]
        p00n = [(alpha_vert - 1) *480,  270*(alpha_horiz - 1)*flip_vert]
        p10n = [960-(alpha_vert - 1) * 480, 270*(1 - alpha_horiz)*flip_vert]
        p01n = [(1 - alpha_vert) * 480, 540 - 270*(alpha_horiz - 1)*flip_vert]
        p11n = [(1 + alpha_vert) * 480, 540 + 270*(alpha_horiz - 1)*flip_vert]

        #print p00n, p10n, p01n, p11n, 'homography points new'

        new_horiz_cam_center = new_K_kin[1, 2] + p00n[1]
        #print new_K_kin, new_horiz_cam_center, p00n

        A_2nx9 = self.assemble_A_2nx9(p00, p00n, None)
        A_2nx9 = self.assemble_A_2nx9(p10, p10n, A_2nx9)
        A_2nx9 = self.assemble_A_2nx9(p01, p01n, A_2nx9)
        A_2nx9 = self.assemble_A_2nx9(p11, p11n, A_2nx9)
        A_2nx9 = np.concatenate((A_2nx9, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1]])), axis=0)#
        x_vert, _, _, _  = np.linalg.lstsq(A_2nx9, np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.]))



        p00 = [0, 0]
        p10 = [960, 0]
        p01 = [0, 540]
        p11 = [960, 540]

        #p00n = [(1 - alpha_horiz) * 960,  0]
        #p10n = [960 +960* (alpha_horiz - 1), 540 * 0]
        #p01n = [(1 - alpha_horiz) * 960, 540]
        #p11n = [960 +960* (alpha_horiz - 1), 540]
        #p00n = [960*(alpha_horiz-1)*flip_vert,  270*(alpha_horiz - 1)]
        #p10n = [960+960*(alpha_horiz-1)*flip_vert, 270*(1 - alpha_horiz)]
        #p01n = [960*(alpha_horiz-1)*flip_vert, 540]
        #p11n = [960+960*(alpha_horiz-1)*flip_vert, 540]
        #p00n = [960*(alpha_horiz-1)*flip_vert,  0]
        #p10n = [960+960*(alpha_horiz-1)*flip_vert, 0]
        #p01n = [960*(alpha_horiz-1)*flip_vert, 540]
        #p11n = [960+960*(alpha_horiz-1)*flip_vert, 540]
        p00n = [0,  270*(alpha_horiz - 1)*flip_vert]
        p10n = [960, 270*(1 - alpha_horiz)*flip_vert]
        p01n = [0, 540 - 270*(alpha_horiz - 1)*flip_vert]
        p11n = [960, 540 + 270*(alpha_horiz - 1)*flip_vert]


        #new_vert_cam_center = new_K_kin[0, 2] + p00n[1]
        #print new_K_kin, new_horiz_cam_center, p00n

        A_2nx9 = self.assemble_A_2nx9(p00, p00n, None)
        A_2nx9 = self.assemble_A_2nx9(p10, p10n, A_2nx9)
        A_2nx9 = self.assemble_A_2nx9(p01, p01n, A_2nx9)
        A_2nx9 = self.assemble_A_2nx9(p11, p11n, A_2nx9)
        A_2nx9 = np.concatenate((A_2nx9, np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1]])), axis=0)#
        x_horiz, _, _, _  = np.linalg.lstsq(A_2nx9, np.array([0., 0., 0., 0., 0., 0., 0., 0., 1.]))

        #x = np.matmul(x_vert.reshape(3,3), x_horiz.reshape(3,3))
        #print x_vert.reshape(3,3), "HOMOGRAPHY"

        return x_vert.reshape(3,3)




    def color_image(self, color_orig, kcam, new_K_kin, u_c=None, v_c=None, u_c_drop=None, v_c_drop=None,
                    u_c_pmat=None, v_c_pmat=None, alpha_vert = 1.0, alpha_horiz = 1.0):
        # COLOR
        color_reshaped = cv2.undistort(color_orig, kcam.K, kcam.D, None, new_K_kin)
        h = self.get_new_K_kin_homography(alpha_vert, alpha_horiz, new_K_kin)
        color_reshaped = cv2.warpPerspective(color_reshaped, h, (color_reshaped.shape[1], color_reshaped.shape[0]))



        for marker in range(0, 4):
            try:
                pass
                #color in green squares where the AR tags are
                #color_reshaped[int(v_c[marker]) - 8:int(v_c[marker]) + 8, int(u_c[marker]) - 8:int(u_c[marker]) + 8, :] = 0
                #color_reshaped[int(v_c[marker]) - 5:int(v_c[marker]) + 5, int(u_c[marker]) - 5:int(u_c[marker]) + 5, 1] = 255

                #color in red markers where that are 10 cm below the AR tags at the level of a flat pressure mat
                #color_reshaped[int(v_c_drop[marker]) - 8:int(v_c_drop[marker]) + 8, int(u_c_drop[marker]) - 8:int(u_c_drop[marker]) + 8, :] = 0
                #color_reshaped[int(v_c_drop[marker]) - 5:int(v_c_drop[marker]) + 5, int(u_c_drop[marker]) - 5:int(u_c_drop[marker]) + 5, 2] = 255
            except:
                pass
        '''
        try:
            #color in a red rectangle outline where the pressure mat is
            color_reshaped[int(v_c_pmat[2]) - 2:int(v_c_pmat[3]) + 2, int(u_c_pmat[0]) - 2:int(u_c_pmat[0]) + 2, :] = 0
            color_reshaped[int(v_c_pmat[2]) - 2:int(v_c_pmat[3]) + 2, int(u_c_pmat[1]) - 2:int(u_c_pmat[1]) + 2, :] = 0
            color_reshaped[int(v_c_pmat[2]) - 2:int(v_c_pmat[2]) + 2, int(u_c_pmat[1]) - 2:int(u_c_pmat[0]) + 2, :] = 0
            color_reshaped[int(v_c_pmat[3]) - 2:int(v_c_pmat[3]) + 2, int(u_c_pmat[1]) - 2:int(u_c_pmat[0]) + 2, :] = 0
            color_reshaped[int(v_c_pmat[2]) - 1:int(v_c_pmat[3]) + 1, int(u_c_pmat[0]) - 1:int(u_c_pmat[0]) + 1, 2] = 255
            color_reshaped[int(v_c_pmat[2]) - 1:int(v_c_pmat[3]) + 1, int(u_c_pmat[1]) - 1:int(u_c_pmat[1]) + 1, 2] = 255
            color_reshaped[int(v_c_pmat[2]) - 1:int(v_c_pmat[2]) + 1, int(u_c_pmat[1]) - 1:int(u_c_pmat[0]) + 1, 2] = 255
            color_reshaped[int(v_c_pmat[3]) - 1:int(v_c_pmat[3]) + 1, int(u_c_pmat[1]) - 1:int(u_c_pmat[0]) + 1, 2] = 255
        except:
            pass
        '''
        color_reshaped = np.rot90(color_reshaped)

        return color_reshaped, color_reshaped.shape



    def depth_image(self, depth_r_orig, u_c = None, v_c = None):
        # DEPTH'
        if HENRY_EVANS == True:
            depth_r_reshaped = depth_r_orig / 3 - 300
        else:
            depth_r_reshaped = depth_r_orig / 4 - 300
        depth_r_reshaped = np.clip(depth_r_reshaped, 0, 255)
        depth_r_reshaped = depth_r_reshaped.astype(np.uint8)
        depth_r_reshaped = np.stack((depth_r_reshaped,) * 3, -1)

        for marker in range(0, 4):
            try:
                pass
                #depth_r_reshaped[int(v_c[marker]) - 8:int(v_c[marker]) + 8, int(u_c[marker]) - 8:int(u_c[marker]) + 8, :] = 0
                #depth_r_reshaped[int(v_c[marker]) - 5:int(v_c[marker]) + 5, int(u_c[marker]) - 5:int(u_c[marker]) + 5, 1] = 255
            except:
                pass

        depth_r_reshaped = np.rot90(depth_r_reshaped)
        depth_r_orig = np.rot90(depth_r_orig)
        return depth_r_reshaped, depth_r_reshaped.shape, depth_r_orig

    def thermal_image(self, thermal_orig, tcam, new_K_thr, new_K_kin, color_size, u_c, v_c, u_t=None, v_t=None, alpha_vert = 1.0, alpha_horiz = 1.0):
        # THERMAL

        mapx,mapy = cv2.initUndistortRectifyMap(tcam.K, tcam.D, None, new_K_thr,(thermal_orig.shape[1],thermal_orig.shape[0]),5)
        thermal_reshaped = cv2.remap(thermal_orig,mapx,mapy,cv2.INTER_AREA)


        factor_x = new_K_kin[0,0] / new_K_thr[0,0] * .99#2.04 #up and down person's body
        factor_y = new_K_kin[1,1] / new_K_thr[1,1] * .98 #across persons body


        thermal_reshaped = cv2.resize(thermal_reshaped,
                                      (int(factor_x * thermal_reshaped.shape[1]), int(factor_y * thermal_reshaped.shape[0])),
                                      interpolation=cv2.INTER_CUBIC).astype(np.uint8)

        thermal_reshaped = np.stack((thermal_reshaped,) * 3, -1)
        thermal_reshaped_temp = np.zeros((color_size[1], color_size[0], color_size[2])).astype(np.uint8)
        thermal_reshaped_temp[:, :, 1] = 50


        try:
            for marker in range(0, 4):
                if u_c[marker] is not None:
                    uc_samp = u_c[marker]
                    vc_samp = v_c[marker]
                    ut_samp = u_t[marker]*factor_x
                    vt_samp = v_t[marker]*factor_y

                    shift_x = uc_samp - ut_samp
                    shift_y = vc_samp - vt_samp

                    shift_x_int = np.rint(shift_x).astype(np.uint16)
                    shift_y_int = np.rint(shift_y).astype(np.uint16)

                    thermal_reshaped_temp[shift_y_int:thermal_reshaped.shape[0] + shift_y_int, shift_x_int:thermal_reshaped.shape[1] + shift_x_int, :] = thermal_reshaped
                    # thermal_reshaped_temp[0:thermal_reshaped.shape[0], 0:thermal_reshaped.shape[1], :] = thermal_reshaped
                    thermal_reshaped = thermal_reshaped_temp

                    for marker in range(0, 4):
                        try:
                            thermal_reshaped[
                            int(v_t[marker] * factor_y + shift_y) - 8:int(v_t[marker] * factor_y + shift_y) + 8,
                            int(u_t[marker] * factor_x + shift_x) - 8:int(u_t[marker] * factor_x + shift_x) + 8, :] = 0
                            thermal_reshaped[
                            int(v_t[marker] * factor_y + shift_y) - 5:int(v_t[marker] * factor_y + shift_y) + 5,
                            int(u_t[marker] * factor_x + shift_x) - 5:int(u_t[marker] * factor_x + shift_x) + 5,
                            1] = 255
                        except:
                            pass
                    break
        except: pass

        thermal_reshaped = thermal_reshaped_temp

        h = self.get_new_K_kin_homography(alpha_vert, alpha_horiz, new_K_kin)
        thermal_reshaped = cv2.warpPerspective(thermal_reshaped, h, (thermal_reshaped.shape[1], thermal_reshaped.shape[0]))
        thermal_reshaped = np.rot90(thermal_reshaped)

        return thermal_reshaped, thermal_reshaped.shape



    def pressure_image(self, pressure_orig, pressure_orig_size, pressure_im_size_required, color_size, u_c_drop, v_c_drop, u_c_pmat, v_c_pmat, u_p_bend, v_p_bend):



        # PRESSURE
        pressure_reshaped_temp = np.reshape(pressure_orig, pressure_orig_size)
        if HENRY_EVANS == True:
            pressure_reshaped_temp = np.flipud(np.fliplr(pressure_reshaped_temp))

        pressure_reshaped = np.zeros((pressure_reshaped_temp.shape[0], pressure_reshaped_temp.shape[1], 3))

        for j in range(0, pressure_reshaped.shape[0]):
            for i in range(0, pressure_reshaped.shape[1]):
                if pressure_reshaped_temp[j, i] > 50:
                    pressure_reshaped[j, i, 2] = 1.
                    pressure_reshaped[j, i, 0] = (100 - pressure_reshaped_temp[j, i]) * .9 / 50.
                else:
                    pressure_reshaped[j, i, 2] = pressure_reshaped_temp[j, i] / 40.
                    pressure_reshaped[j, i, 0] = 1.
                    pressure_reshaped[j, i, 1] = (30 - np.abs(pressure_reshaped_temp[j, i])) / 90.
        pressure_reshaped[:, :, 0] = np.clip(pressure_reshaped[:, :, 0] * 4, 0, 1)
        pressure_reshaped[:, :, 2] = np.clip(pressure_reshaped[:, :, 2] * 4, 0, 1)
        pressure_reshaped = (pressure_reshaped * 255).astype(np.uint8)
        pressure_reshaped = cv2.resize(pressure_reshaped, (pressure_im_size_required[1], pressure_im_size_required[0]),
                                       interpolation=cv2.INTER_CUBIC).astype(np.uint8)



        pressure_reshaped = np.rot90(pressure_reshaped, 3)
        pressure_reshaped_temp2 = np.zeros((color_size[1], color_size[0], color_size[2])).astype(np.uint8)
        pressure_reshaped_temp2[:, :, 0] = 50

        try:
            low_vert = np.rint(v_c_pmat[2]).astype(np.uint16)
            low_horiz = np.rint(u_c_pmat[1]).astype(np.uint16)
            pressure_reshaped_temp2[low_vert:low_vert+pressure_reshaped.shape[0],low_horiz:low_horiz+pressure_reshaped.shape[1],:] = pressure_reshaped

            pressure_reshaped = pressure_reshaped_temp2
            head_bend_loc = pressure_im_size_required[0]*HEAD_BEND_TAXEL/64 + low_horiz
            legs_bend_loc1 = pressure_im_size_required[0]*LEGS_BEND1_TAXEL/64 + low_horiz
            legs_bend_loc2 = pressure_im_size_required[0]*20/64 + low_horiz

            #cut off a section of the image above the bend between the frame and the head of the bed
            head_reshaped = np.copy(pressure_reshaped[:, head_bend_loc:, :])
            legs1_reshaped = np.copy(pressure_reshaped[:, legs_bend_loc2:legs_bend_loc1, :])
            legs2_reshaped = np.copy(pressure_reshaped[:, low_horiz:legs_bend_loc1, :])


            head_points2 = np.zeros((4, 2), dtype=np.float32)
            head_points1 = np.zeros((4, 2), dtype=np.float32)
            legs_points1 = np.zeros((4, 2), dtype=np.float32)
            legs_points2 = np.zeros((4, 2), dtype=np.float32)
            legs_points3 = np.zeros((4, 2), dtype=np.float32)
            legs_points4 = np.zeros((4, 2), dtype=np.float32)


            #original head view, square
            head_points1[0] = [0, low_vert] #happens at head bend
            head_points1[1] = [0, low_vert + pressure_im_size_required[1]] #happens at head bend
            head_points1[2] = [pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64, low_vert ]
            head_points1[3] = [pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64, low_vert + pressure_im_size_required[1]]

            #modified head view
            head_points2[0] = head_points1[0] #happens at head bend
            head_points2[1] = head_points1[1] #happens at head bend
            head_points2[2] = [np.rint(u_p_bend[0] - head_bend_loc - 3).astype(np.uint16), np.rint(v_p_bend[0]).astype(np.uint16) - 3]  # np.copy([head_points1[2][0] - decrease_from_orig_len, head_points1[2][1] - increase_across_pmat])
            head_points2[3] = [np.rint(u_p_bend[1] - head_bend_loc - 3).astype(np.uint16), np.rint(v_p_bend[1]).astype(np.uint16) + 4]  # np.copy([head_points1[3][0] - decrease_from_orig_len, head_points1[3][1] + increase_across_pmat])

            #original legs block1 view, square
            legs_points1[0] = [0, low_vert ] #happens at legs bend2
            legs_points1[1] = [0, low_vert + pressure_im_size_required[1]] #happens at legs bend2
            legs_points1[2] = [pressure_im_size_required[0]*(64 - LEGS_BEND2_TAXEL)/64-pressure_im_size_required[0]*(64 - LEGS_BEND1_TAXEL)/64, low_vert ] #happens at legs bend1
            legs_points1[3] = [pressure_im_size_required[0]*(64 - LEGS_BEND2_TAXEL)/64-pressure_im_size_required[0]*(64 - LEGS_BEND1_TAXEL)/64, low_vert + pressure_im_size_required[1]] #happens at legs bend1

            #modified legs block1 view
            legs_points2[0] = [np.rint(np.copy(legs_points1[2][0]) + u_p_bend[2] - legs_bend_loc1) - 2, np.rint(v_p_bend[2]).astype(np.uint16) - 3]
            legs_points2[1] = [np.rint(np.copy(legs_points1[3][0]) + u_p_bend[3] - legs_bend_loc1) - 2, np.rint(v_p_bend[3]).astype(np.uint16) + 4]
            legs_points2[2] = legs_points1[2]
            legs_points2[3] = legs_points1[3]

            #original legs block2 view, square
            legs_points3[0] = [0, low_vert ] #happens at legs bottom
            legs_points3[1] = [0, low_vert + pressure_im_size_required[1]] #happens at legs bottom
            legs_points3[2] = [pressure_im_size_required[0]*64/64-pressure_im_size_required[0]*(64 - LEGS_BEND2_TAXEL)/64, low_vert ] #happens at legs bend2
            legs_points3[3] = [pressure_im_size_required[0]*64/64-pressure_im_size_required[0]*(64 - LEGS_BEND2_TAXEL)/64, low_vert + pressure_im_size_required[1]] #happens at legs bend2

            #modified legs block2 view
            legs_points4[0] = [np.rint(np.copy(legs_points3[2][0]) + u_p_bend[4] - legs_bend_loc2) - 2, np.rint(v_p_bend[4]).astype(np.uint16) - 3] #happens at legs bottom
            legs_points4[1] = [np.rint(np.copy(legs_points3[2][0]) + u_p_bend[5] - legs_bend_loc2) - 2, np.rint(v_p_bend[5]).astype(np.uint16) + 4] #happens at legs bottom
            legs_points4[2] = [legs_points3[2][0] + legs_points2[0][0], legs_points2[0][1]] #happens at legs bend2
            legs_points4[3] = [legs_points3[2][0] + legs_points2[0][0], legs_points2[1][1]] #happens at legs bend2


            #reproject head of the bed
            h, mask = cv2.findHomography(head_points1, head_points2, cv2.RANSAC)
            height, width, channels = head_reshaped.shape
            head_warped = cv2.warpPerspective(head_reshaped, h, (width, height))
            pressure_reshaped[:, 960-head_warped.shape[1]:960,:] = head_warped



            #reproject upper legs part of the bed
            h, mask = cv2.findHomography(legs_points1, legs_points2, cv2.RANSAC)
            height, width, channels = legs1_reshaped.shape
            legs1_warped = cv2.warpPerspective(legs1_reshaped, h, (width, height))
            pressure_reshaped[:, legs_bend_loc1-legs1_warped.shape[1]:legs_bend_loc1,:] = legs1_warped
            legs_bend_loc2_new_moved = (np.rint(np.copy(legs_points1[2][0]) + u_p_bend[2] - legs_bend_loc1) - 2).astype(np.int16) #vertical bed location where the bending angle ends up in projected 2d space

        

            #reproject lower legs part of the bed that is dependent on the upper legs part of the bed
            h, mask = cv2.findHomography(legs_points3, legs_points4, cv2.RANSAC)
            height, width, channels = legs2_reshaped.shape
            legs2_warped = cv2.warpPerspective(legs2_reshaped, h, (width, height))
            legs2_warped = legs2_warped[:, 0:np.rint(legs_points4[2][0]).astype(np.uint16), :]
            pressure_reshaped[:, legs_bend_loc1-legs1_warped.shape[1]-legs2_warped.shape[1]+legs_bend_loc2_new_moved:legs_bend_loc1-legs1_warped.shape[1]+legs_bend_loc2_new_moved,:] = legs2_warped


            #fill in the black spot below when the lower legs move up too far
            if legs_points4[0][0] > 0:
                pressure_reshaped[:, :legs_bend_loc1-legs1_warped.shape[1]-legs2_warped.shape[1]+legs_bend_loc2_new_moved+(legs_points4[0][0]).astype(np.uint16), 0] = 50



            #Add in some markings for where the flat bed would be: bend of the head
            pressure_reshaped[int(low_vert)-8:int(low_vert)+8, head_bend_loc-1:head_bend_loc+1, :] = 255
            pressure_reshaped[int(low_vert) + pressure_im_size_required[1]-8:int(low_vert) + pressure_im_size_required[1]+8, head_bend_loc-1:head_bend_loc+1, :] = 255

            #Add in some markings for where the flat bed would be: corners of the head
            pressure_reshaped[int(low_vert)-1:int(low_vert)+1, head_bend_loc+pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64-10:head_bend_loc+pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64+1, :] = 255
            pressure_reshaped[int(low_vert)-1:int(low_vert)+10, head_bend_loc+pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64-1:head_bend_loc+pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64+1, :] = 255
            pressure_reshaped[int(low_vert) + pressure_im_size_required[1]- 1:int(low_vert) + pressure_im_size_required[1]+1, head_bend_loc+pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64-10:head_bend_loc+pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64+1, :] = 255
            pressure_reshaped[int(low_vert) + pressure_im_size_required[1]-10:int(low_vert) + pressure_im_size_required[1]+1, head_bend_loc+pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64-1:head_bend_loc+pressure_im_size_required[0]*(64 - HEAD_BEND_TAXEL)/64+1, :] = 255

            # Add in some markings for where the flat bed would be: corners of the leg rest
            pressure_reshaped[int(low_vert)-1:int(low_vert)+1, head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64-1:head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64+10, :] = 255
            pressure_reshaped[int(low_vert)-1:int(low_vert)+10, head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64-1:head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64+1, :] = 255
            pressure_reshaped[int(low_vert) + pressure_im_size_required[1]- 1:int(low_vert) + pressure_im_size_required[1]+1, head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64-1:head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64+10, :] = 255
            pressure_reshaped[int(low_vert) + pressure_im_size_required[1]-10:int(low_vert) + pressure_im_size_required[1]+1, head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64-1:head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64+1, :] = 255

            coords_from_top_left = [low_vert, head_bend_loc-pressure_im_size_required[0]*HEAD_BEND_TAXEL/64]
            print "CORNER VERTS", int(low_vert), 960 - (head_bend_loc - pressure_im_size_required[0] * HEAD_BEND_TAXEL / 64)


            #Add in some markings for where the flat bed would be: primary bend of the legs
            pressure_reshaped[int(low_vert)-8:int(low_vert)+8, legs_bend_loc1-1:legs_bend_loc1+1, :] = 255
            pressure_reshaped[int(low_vert) + pressure_im_size_required[1]-8:int(low_vert) + pressure_im_size_required[1]+8, legs_bend_loc1-1:legs_bend_loc1+1, :] = 255

            #Add in some markings for where the flat bed would be: secondary bend of the legs
            pressure_reshaped[int(low_vert)-8:int(low_vert)+8, legs_bend_loc2-1:legs_bend_loc2+1, :] = 255
            pressure_reshaped[int(low_vert) + pressure_im_size_required[1]-8:int(low_vert) + pressure_im_size_required[1]+8, legs_bend_loc2-1:legs_bend_loc2+1, :] = 255

            #print pressure_reshaped.shape, coords_from_top_left

            #add red markers that are dropped below the AR tags at the height of the bed
            for marker in range(0, 4):
                try:
                    pressure_reshaped[int(v_c_drop[marker]) - 8:int(v_c_drop[marker]) + 8, int(u_c_drop[marker]) - 8:int(u_c_drop[marker]) + 8, :] = 0
                    pressure_reshaped[int(v_c_drop[marker]) - 5:int(v_c_drop[marker]) + 5, int(u_c_drop[marker]) - 5:int(u_c_drop[marker]) + 5, 2] = 255
                except:
                    pass



        except:
            pressure_reshaped = pressure_reshaped_temp2
            coords_from_top_left = None


        pressure_reshaped = np.rot90(pressure_reshaped)
        return pressure_reshaped, pressure_reshaped.shape, coords_from_top_left
