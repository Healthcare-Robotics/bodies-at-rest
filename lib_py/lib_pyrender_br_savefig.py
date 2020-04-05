
try:
    import open3d as o3d
except:
    print "COULD NOT IMPORT 03D"
import trimesh
import pyrender
import pyglet
from scipy import ndimage

import numpy as np
import random
import copy
from smpl.smpl_webuser.serialization import load_model



from time import sleep

#ROS
#import rospy
#import tf
DATASET_CREATE_TYPE = 1

import cv2

import math
from random import shuffle
import torch
import torch.nn as nn

import tensorflow as tensorflow
import cPickle as pickle


#IKPY
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

#MISC
import time as time
import matplotlib.pyplot as plt
import matplotlib.cm as cm #use cm.jet(list)
#from mpl_toolkits.mplot3d import Axes3D

#hmr
from hmr.src.tf_smpl.batch_smpl import SMPL

import cPickle as pkl

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

import os


from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvasAgg




class pyRenderMesh():
    def __init__(self, render):

        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.render = render
        self.scene = pyrender.Scene()

        #self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 1.0 ,0.0])
        self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.8, 0.0], metallicFactor=0.6, roughnessFactor=0.5)#
        self.human_mat_gt = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.05, 0.05, 0.05, 0.0], metallicFactor=0.6, roughnessFactor=0.5)#

        self.human_mat_GT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.3, 0.0 ,0.0])
        self.human_arm_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.8 ,1.0])
        self.human_mat_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 0.3 ,0.5])
        self.human_bed_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.7, 0.7, 0.2 ,0.5])
        self.human_mat_D = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 1.0], alphaMode="BLEND")

        #if render == True:
        mesh_color_mult = 0.25

        self.mesh_parts_mat_list = [
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 166. / 255., mesh_color_mult * 206. / 255., mesh_color_mult * 227. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 31. / 255., mesh_color_mult * 120. / 255., mesh_color_mult * 180. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 251. / 255., mesh_color_mult * 154. / 255., mesh_color_mult * 153. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 227. / 255., mesh_color_mult * 26. / 255., mesh_color_mult * 28. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 178. / 255., mesh_color_mult * 223. / 255., mesh_color_mult * 138. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 51. / 255., mesh_color_mult * 160. / 255., mesh_color_mult * 44. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 253. / 255., mesh_color_mult * 191. / 255., mesh_color_mult * 111. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 255. / 255., mesh_color_mult * 127. / 255., mesh_color_mult * 0. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 202. / 255., mesh_color_mult * 178. / 255., mesh_color_mult * 214. / 255., 0.0]),
            pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 106. / 255., mesh_color_mult * 61. / 255., mesh_color_mult * 154. / 255., 0.0])]

        self.artag_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 1.0, 0.3, 0.5])
        self.artag_mat_other = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 0.0])
        #self.artag_r = np.array([[-0.055, -0.055, 0.0], [-0.055, 0.055, 0.0], [0.055, -0.055, 0.0], [0.055, 0.055, 0.0]])
        self.artag_r = np.array([[0.0, 0.0, 0.075], [0.0286*64*1.04/1.04, 0.0, 0.075], [0.0, 0.01, 0.075], [0.0286*64*1.04/1.04, 0.01, 0.075],
                                 [0.0, 0.0, 0.075], [0.0, 0.0286*27 /1.06, 0.075], [0.01, 0.0, 0.075], [0.01, 0.0286*27 /1.06, 0.075],
                                 [0.0,  0.0286*27 /1.06, 0.075], [0.0286*64*1.04/1.04, 0.0286*27 /1.06, 0.075], [0.0,  0.0286*27 /1.06+0.01, 0.075], [0.0286*64*1.04/1.04,  0.0286*27 /1.06+0.01, 0.075],
                                 [0.0286*64*1.04/1.04, 0.0, 0.075], [0.0286*64*1.04/1.04, 0.0286*27 /1.06, 0.075], [0.0286*64*1.04/1.04-0.01, 0.0, 0.075], [0.0286*64*1.04/1.04-0.01, 0.0286*27 /1.06, 0.075],
                                 ])
        #self.artag_f = np.array([[0, 1, 3], [3, 1, 0], [0, 2, 3], [3, 2, 0], [1, 3, 2]])
        self.artag_f = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 3], [1, 3, 2],
                                 [4, 5, 6], [4, 6, 5], [5, 6, 7], [5, 7, 6],
                                 [8, 9, 10], [8, 10, 9], [9, 10, 11], [9, 11, 10],
                                 [12, 13, 14], [12, 14, 13], [13, 14, 15], [13, 15, 14]])
        #self.artag_facecolors_root = np.array([[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0],[0.0, 1.0, 0.0]])
        self.artag_facecolors_root =  np.array([[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                [0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],[0.3, 0.3, 0.0],
                                                ])
        self.artag_facecolors_root_gt =  np.array([[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                                    [0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                                    [0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                                    [0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                                    ])
        #self.artag_facecolors = np.array([[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 0.0],])
        self.artag_facecolors = np.copy(self.artag_facecolors_root)
        self.artag_facecolors_gt = np.copy(self.artag_facecolors_root_gt)


        self.pic_num = 0


    def get_3D_pmat_markers(self, pmat, angle = 60.0, solidcolor = False):

        pmat_reshaped = pmat.reshape(64, 27)

        pmat_colors = cm.jet(pmat_reshaped/100)
        #print pmat_colors.shape
        pmat_colors[:, :, 3] = 1.0 #translucency

        if solidcolor == True:
            pmat_colors[:, :, 3] = 0.2#0.7 #translucency
            pmat_colors[:, :, 0] = 0.6
            pmat_colors[:, :, 1] = 0.6
            pmat_colors[:, :, 2] = 0.0


        pmat_xyz = np.zeros((65, 28, 3))
        pmat_faces = []
        pmat_facecolors = []

        for j in range(65):
            for i in range(28):

                pmat_xyz[j, i, 1] = i * 0.0286 /1.06# * 1.02 #1.0926 - 0.02
                pmat_xyz[j, i, 0] = ((64 - j) * 0.0286) * 1.04 /1.04#1.1406 + 0.05 #only adjusts pmat NOT the SMPL person
                pmat_xyz[j, i, 2] = 0.075#0.12 + 0.075
                #if j > 23:
                #    pmat_xyz[j, i, 0] = ((64 - j) * 0.0286 - 0.0286 * 3 * np.sin(np.deg2rad(angle)))*1.04 + 0.15#1.1406 + 0.05
                #    pmat_xyz[j, i, 2] = 0.12 + 0.075
                #    # print marker.pose.position.x, 'x'
                #else:

                #    pmat_xyz[j, i, 0] = ((41) * 0.0286 + (23 - j) * 0.0286 * np.cos(np.deg2rad(angle)) \
                #                        - (0.0286 * 3 * np.sin(np.deg2rad(angle))) * 0.85)*1.04 + 0.15#1.1406 + 0.05
                #    pmat_xyz[j, i, 2] = -((23 - j) * 0.0286 * np.sin(np.deg2rad(angle))) * 0.85 + 0.12 + 0.075
                    # print j, marker.pose.position.z, marker.pose.position.y, 'head'

                if j < 64 and i < 27:
                    coord1 = j * 28 + i
                    coord2 = j * 28 + i + 1
                    coord3 = (j + 1) * 28 + i
                    coord4 = (j + 1) * 28 + i + 1

                    pmat_faces.append([coord1, coord2, coord3]) #bottom surface
                    pmat_faces.append([coord1, coord3, coord2]) #top surface
                    pmat_faces.append([coord4, coord3, coord2]) #bottom surface
                    pmat_faces.append([coord2, coord3, coord4]) #top surface
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])
                    pmat_facecolors.append(pmat_colors[j, i, :])

        #print np.min(pmat_faces), np.max(pmat_faces), 'minmax'


        pmat_verts = list((pmat_xyz).reshape(1820, 3))

        #print "len faces: ", len(pmat_faces)
        #print "len verts: ", len(pmat_verts)
        #print len(pmat_faces), len(pmat_facecolors)

        return pmat_verts, pmat_faces, pmat_facecolors




    def get_human_mesh_parts(self, smpl_verts, smpl_faces, viz_type = None, segment_limbs = False):

        if segment_limbs == True:
            if viz_type == 'arm_penetration':
                segmented_dict = load_pickle('segmented_mesh_idx_faces_larm.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['l_arm_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['l_arm_face_list']]
            elif viz_type == 'leg_correction':
                segmented_dict = load_pickle('segmented_mesh_idx_faces_rleg.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['r_leg_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['r_leg_face_list']]
            else:
                segmented_dict = load_pickle('segmented_mesh_idx_faces.p')
                human_mesh_vtx_parts = [smpl_verts[segmented_dict['l_lowerleg_idx_list'], :],
                                        smpl_verts[segmented_dict['r_lowerleg_idx_list'], :],
                                        smpl_verts[segmented_dict['l_upperleg_idx_list'], :],
                                        smpl_verts[segmented_dict['r_upperleg_idx_list'], :],
                                        smpl_verts[segmented_dict['l_forearm_idx_list'], :],
                                        smpl_verts[segmented_dict['r_forearm_idx_list'], :],
                                        smpl_verts[segmented_dict['l_upperarm_idx_list'], :],
                                        smpl_verts[segmented_dict['r_upperarm_idx_list'], :],
                                        smpl_verts[segmented_dict['head_idx_list'], :],
                                        smpl_verts[segmented_dict['torso_idx_list'], :]]
                human_mesh_face_parts = [segmented_dict['l_lowerleg_face_list'],
                                         segmented_dict['r_lowerleg_face_list'],
                                         segmented_dict['l_upperleg_face_list'],
                                         segmented_dict['r_upperleg_face_list'],
                                         segmented_dict['l_forearm_face_list'],
                                         segmented_dict['r_forearm_face_list'],
                                         segmented_dict['l_upperarm_face_list'],
                                         segmented_dict['r_upperarm_face_list'],
                                         segmented_dict['head_face_list'],
                                         segmented_dict['torso_face_list']]
        else:
            human_mesh_vtx_parts = [smpl_verts]
            human_mesh_face_parts = [smpl_faces]

        return human_mesh_vtx_parts, human_mesh_face_parts




    def render_mesh_pc_bed_pyrender_everything(self, smpl_verts, smpl_faces, camera_point, bedangle, RESULTS_DICT,
                                    pc = None, pmat = None, smpl_render_points = False, markers = None,
                                    dropout_variance=None, color_im = None, tf_corners = None, current_pose_type_ct = None,
                                    participant = None):

        pmat *= 0.75
        pmat[pmat>0] += 10

        #print np.min(smpl_verts[:, 0])
        #print np.min(smpl_verts[:, 1])

        shift_estimate_sideways = np.min([-0.15, np.min(smpl_verts[:, 1])])
        #print shift_estimate_sideways
        shift_estimate_sideways = 0.8 - shift_estimate_sideways

        top_smpl_vert = np.max(smpl_verts[:, 0])
        extend_top_bottom  = np.max([np.max(smpl_verts[:, 0]), 64*.0286]) - 64*.0286
        print extend_top_bottom, 'extend top bot'


        shift_both_amount = np.max([0.9, np.max(smpl_verts[:, 1])]) #if smpl is bigger than 0.9 shift less
        shift_both_amount = 1.5 - shift_both_amount + (0.15 + np.min([-0.15, np.min(smpl_verts[:, 1])]))

        #print np.max(smpl_verts[:, 1]), 'max smpl'

        #shift_both_amount = 0.6
        #smpl_verts[:, 2] += 0.5
        #pc[:, 2] += 0.5

        pc[:, 0] = pc[:, 0] # - 0.17 - 0.036608
        pc[:, 1] = pc[:, 1]# + 0.09

        #adjust the point cloud


        #segment_limbs = True

        #if pmat is not None:
         #   if np.sum(pmat) < 5000:
         #       smpl_verts = smpl_verts * 0.001


        smpl_verts_quad = np.concatenate((smpl_verts, np.ones((smpl_verts.shape[0], 1))), axis = 1)
        smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)

        #print smpl_verts_quad.shape

        transform_A = np.identity(4)
        transform_A[1, 3] = shift_both_amount

        transform_B = np.identity(4)
        transform_B[1, 3] = shift_estimate_sideways + shift_both_amount#4.0 #move things over
        smpl_verts_B = np.swapaxes(np.matmul(transform_B, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_C = np.identity(4)
        transform_C[1, 3] = 2.0#2.0 #move things over
        smpl_verts_C = np.swapaxes(np.matmul(transform_C, smpl_verts_quad), 0, 1)[:, 0:3]



        from matplotlib import cm


        human_mesh_vtx_all, human_mesh_face_all = self.get_human_mesh_parts(smpl_verts_B, smpl_faces, segment_limbs=False)

        #GET MESH WITH PMAT
        tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all[0]), faces = np.array(human_mesh_face_all[0]))
        tm_list = [tm_curr]
        original_mesh = [tm_curr]



        mesh_list = []
        mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.human_mat, smooth=True))#wireframe = False)) #this is for the main human



        print np.shape(color_im)
        print tf_corners
        top_idx = float(tf_corners[0,1])
        bot_idx = float(tf_corners[2,1])
        perc_total = (bot_idx-top_idx)/880.
        print perc_total

        fig = plt.figure()
        if self.render == True:

            #print m.r
            #print artag_r
            #create mini meshes for AR tags
            artag_meshes = []
            if markers is not None:
                for marker in markers:
                    if markers[2] is None:
                        artag_meshes.append(None)
                    elif marker is None:
                        artag_meshes.append(None)
                    else:
                        #print marker - markers[2]
                        if marker is markers[2]:
                            print "is markers 2", marker
                            #artag_tm = trimesh.base.Trimesh(vertices=self.artag_r, faces=self.artag_f, face_colors = self.artag_facecolors_root)
                            #artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))
                        else:
                            artag_tm = trimesh.base.Trimesh(vertices=self.artag_r + [0.0, shift_estimate_sideways + shift_both_amount, 0.0], faces=self.artag_f, face_colors = self.artag_facecolors)
                            artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))



            if pmat is not None:
                pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat, bedangle)
                pmat_verts = np.array(pmat_verts)
                pmat_verts = np.concatenate((np.swapaxes(pmat_verts, 0, 1), np.ones((1, pmat_verts.shape[0]))), axis = 0)
                pmat_verts = np.swapaxes(np.matmul(transform_A, pmat_verts), 0, 1)[:, 0:3]
                pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors = pmat_facecolors)
                pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth = False)

                pmat_verts2, _, pmat_facecolors2 = self.get_3D_pmat_markers(pmat, bedangle, solidcolor = True)
                pmat_verts2 = np.array(pmat_verts2)
                pmat_verts2 = np.concatenate((np.swapaxes(pmat_verts2, 0, 1), np.ones((1, pmat_verts2.shape[0]))), axis = 0)
                pmat_verts2 = np.swapaxes(np.matmul(transform_B, pmat_verts2), 0, 1)[:, 0:3]
                pmat_tm2 = trimesh.base.Trimesh(vertices=pmat_verts2, faces=pmat_faces, face_colors = pmat_facecolors2)
                pmat_mesh2 = pyrender.Mesh.from_trimesh(pmat_tm2, smooth = False)

            else:
                pmat_mesh = None
                pmat_mesh2 = None


            #print "Viewing"
            if self.first_pass == True:

                for mesh_part in mesh_list:
                    self.scene.add(mesh_part)
                if pmat_mesh is not None:
                    self.scene.add(pmat_mesh)

                if pmat_mesh2 is not None:
                    self.scene.add(pmat_mesh2)

                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        self.scene.add(artag_mesh)


                lighting_intensity = 20.

                #self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                #                              point_size=2, run_in_thread=True, viewport_size=(1200, 1200))



                self.first_pass = False

                self.node_list = []
                for mesh_part in mesh_list:
                    for node in self.scene.get_nodes(obj=mesh_part):
                        self.node_list.append(node)



                self.artag_nodes = []
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh):
                            self.artag_nodes.append(node)
                if pmat_mesh is not None:
                    for node in self.scene.get_nodes(obj=pmat_mesh):
                        self.pmat_node = node
                if pmat_mesh2 is not None:
                    for node in self.scene.get_nodes(obj=pmat_mesh2):
                        self.pmat_node2 = node

                camera_pose = np.eye(4)
                # camera_pose[0,0] = -1.0
                # camera_pose[1,1] = -1.0

                camera_pose[0, 0] = np.cos(np.pi/2)
                camera_pose[0, 1] = np.sin(np.pi/2)
                camera_pose[1, 0] = -np.sin(np.pi/2)
                camera_pose[1, 1] = np.cos(np.pi/2)
                rot_udpim = np.eye(4)

                rot_y = 180*np.pi/180.
                rot_udpim[1,1] = np.cos(rot_y)
                rot_udpim[2,2] = np.cos(rot_y)
                rot_udpim[1,2] = np.sin(rot_y)
                rot_udpim[2,1] = -np.sin(rot_y)
                camera_pose = np.matmul(rot_udpim,  camera_pose)

                camera_pose[0, 3] = 64*0.0286/2  # -1.0
                camera_pose[1, 3] = 1.2
                camera_pose[2, 3] = -1.0


                # self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True,
                #                              lighting_intensity=10.,
                #                              point_size=5, run_in_thread=True, viewport_size=(1000, 1000))
                # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

                magnify =(64*.0286)*0.5/perc_total

                camera = pyrender.OrthographicCamera(xmag=magnify, ymag = magnify)

                self.scene.add(camera, pose=camera_pose)


                light = pyrender.SpotLight(color=np.ones(3), intensity=250.0, innerConeAngle=np.pi / 10.0,
                                           outerConeAngle=np.pi / 2.0)
                light_pose = np.copy(camera_pose)
                # light_pose[1, 3] = 2.0
                light_pose[0, 3] = 0.8
                light_pose[1, 3] = -0.5
                light_pose[2, 3] = -2.5

                light_pose2 = np.copy(camera_pose)
                light_pose2[0, 3] = 2.5
                light_pose2[1, 3] = 1.0
                light_pose2[2, 3] = -5.0

                light_pose3 = np.copy(camera_pose)
                light_pose3[0, 3] = 1.0
                light_pose3[1, 3] = 5.0
                light_pose3[2, 3] = -4.0

                #light_pose2[0, 3] = 1.0
                #light_pose2[1, 3] = 2.0 #across
                #light_pose2[2, 3] = -1.5
                # light_pose[1, ]

                self.scene.add(light, pose=light_pose)
                self.scene.add(light, pose=light_pose2)
                self.scene.add(light, pose=light_pose3)




            else:
                #self.viewer.render_lock.acquire()

                #reset the human mesh
                for idx in range(len(mesh_list)):
                    self.scene.remove_node(self.node_list[idx])
                    self.scene.add(mesh_list[idx])
                    for node in self.scene.get_nodes(obj=mesh_list[idx]):
                        self.node_list[idx] = node

                #reset the artag meshes
                for artag_node in self.artag_nodes:
                    self.scene.remove_node(artag_node)
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        self.scene.add(artag_mesh)
                self.artag_nodes = []
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh):
                            self.artag_nodes.append(node)


                #reset the pmat mesh
                if pmat_mesh is not None:
                    self.scene.remove_node(self.pmat_node)
                    self.scene.add(pmat_mesh)
                    for node in self.scene.get_nodes(obj=pmat_mesh):
                        self.pmat_node = node


                #reset the pmat mesh
                if pmat_mesh2 is not None:
                    self.scene.remove_node(self.pmat_node2)
                    self.scene.add(pmat_mesh2)
                    for node in self.scene.get_nodes(obj=pmat_mesh2):
                        self.pmat_node2 = node



                #print self.scene.get_nodes()
                #self.viewer.render_lock.release()
            #time.sleep(100)


        r = pyrender.OffscreenRenderer(880, 880)
        # r.render(self.scene)
        color_render, depth = r.render(self.scene)
        # plt.subplot(1, 2, 1)
        plt.axis('off')

        if 880.-bot_idx > top_idx:
            print 'shift im down by', 880.-bot_idx - top_idx
            downshift = int((880.-bot_idx)/2 - top_idx/2 + 0.5)
            color_im[downshift:880] = color_im[0:880 - downshift]

        elif top_idx > (880. - bot_idx):
            print 'shift im up by', top_idx - (880.-bot_idx)
            upshift = int(top_idx/2 - (880.-bot_idx)/2 + 0.5)
            color_im[0:880-upshift]= color_im[upshift:880]

        print tf_corners
        print np.shape(color_render), np.shape(color_im)
        color_im = np.concatenate((color_im[:, :, 2:3], color_im[:, :, 1:2], color_im[:, :, 0:1] ), axis = 2)
        color_im = color_im[:, int(tf_corners[0,0]-10):int(tf_corners[1,0]+10), :]


        im_to_show = np.concatenate((color_render, color_im), axis = 1)


        im_to_show = im_to_show[130-int(extend_top_bottom*300):750+int(extend_top_bottom*300), :, :]

        #plt.imshow(color)
        plt.imshow(im_to_show)
        # plt.subplot(1, 2, 2)
        # plt.axis('off')
        # plt.imshow(depth, cmap=plt.cm.gray_r) >> > plt.show()

        fig.set_size_inches(15., 10.)
        fig.tight_layout()
        #save_name = 'f_hbh_'+'{:04}'.format(self.pic_num)

        save_name = participant+'_'+current_pose_type_ct

        print "saving!"
        fig.savefig('/media/henry/multimodal_data_2/CVPR2020_study/'+participant+'/estimated_poses_camready/'+save_name+'_v2.png', dpi=300)
        #fig.savefig('/media/henry/multimodal_data_2/CVPR2020_study/'+participant+'/natural_est_poses/'+save_name+'.png', dpi=300)
        #fig.savefig('/media/henry/multimodal_data_2/CVPR2020_study/TEST.png', dpi=300)

        #plt.savefig('test2png.png', dpi=100)

        self.pic_num += 1
        #plt.show()
        #if self.pic_num == 20:
        #    print "DONE"
        #    time.sleep(1000000)
        #print "got here"

        #print X.shape


        return RESULTS_DICT

    def render_mesh_pc_bed_pyrender_everything_synth(self, smpl_verts, smpl_faces, camera_point, bedangle, RESULTS_DICT,
                                    smpl_verts_gt = None, pmat = None, smpl_render_points = False, markers = None,
                                    dropout_variance=None, tf_corners = None, save_name = 'test_synth'):

        pmat *= 0.75
        pmat[pmat>0] += 10

        viz_popup = False

        #print np.min(smpl_verts[:, 0])
        #print np.min(smpl_verts[:, 1])

        shift_estimate_sideways = np.min([-0.15, np.min(smpl_verts[:, 1])])
        #print shift_estimate_sideways
        shift_estimate_sideways = 0.8 - shift_estimate_sideways

        top_smpl_vert = np.max(smpl_verts[:, 0])
        extend_top_bottom  = np.max([np.max(smpl_verts[:, 0]), 64*.0286]) - 64*.0286
        print extend_top_bottom, 'extend top bot'


        shift_both_amount = np.max([0.9, np.max(smpl_verts[:, 1])]) #if smpl is bigger than 0.9 shift less
        shift_both_amount = 1.5 - shift_both_amount + (0.15 + np.min([-0.15, np.min(smpl_verts[:, 1])]))

        smpl_verts_quad = np.concatenate((smpl_verts, np.ones((smpl_verts.shape[0], 1))), axis = 1)
        smpl_verts_quad = np.swapaxes(smpl_verts_quad, 0, 1)


        smpl_verts_quad_gt = np.concatenate((smpl_verts_gt, np.ones((smpl_verts_gt.shape[0], 1))), axis = 1)
        smpl_verts_quad_gt = np.swapaxes(smpl_verts_quad_gt, 0, 1)

        #print smpl_verts_quad.shape

        shift_ground_truth = 1.3

        transform_A = np.identity(4)
        transform_A[1, 3] = shift_both_amount

        transform_B = np.identity(4)
        transform_B[1, 3] = shift_estimate_sideways + shift_both_amount#4.0 #move things over
        smpl_verts_B = np.swapaxes(np.matmul(transform_B, smpl_verts_quad), 0, 1)[:, 0:3]

        transform_C = np.identity(4)
        transform_C[1, 3] = shift_estimate_sideways + shift_both_amount+shift_ground_truth #move things over
        smpl_verts_C = np.swapaxes(np.matmul(transform_C, smpl_verts_quad_gt), 0, 1)[:, 0:3]



        from matplotlib import cm


        human_mesh_vtx_all, human_mesh_face_all = self.get_human_mesh_parts(smpl_verts_B, smpl_faces, segment_limbs=False)

        #GET MESH WITH PMAT
        tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all[0]), faces = np.array(human_mesh_face_all[0]))
        tm_list = [tm_curr]
        original_mesh = [tm_curr]

        mesh_list = []
        mesh_list.append(pyrender.Mesh.from_trimesh(tm_list[0], material = self.human_mat, smooth=True))#wireframe = False)) #this is for the main human


        human_mesh_vtx_all_gt, human_mesh_face_all_gt = self.get_human_mesh_parts(smpl_verts_C, smpl_faces, segment_limbs=False)

        #GET MESH GT WITH PMAT
        tm_curr_gt = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_all_gt[0]), faces = np.array(human_mesh_face_all_gt[0]))
        tm_list_gt = [tm_curr_gt]
        original_mesh_gt = [tm_curr_gt]

        mesh_list_gt = []
        mesh_list_gt.append(pyrender.Mesh.from_trimesh(tm_list_gt[0], material = self.human_mat_gt, smooth=True))#wireframe = False)) #this is for the main human


        fig = plt.figure()
        if self.render == True:


            artag_meshes = []
            artag_tm = trimesh.base.Trimesh(vertices=self.artag_r + [0.0, shift_estimate_sideways + shift_both_amount, 0.0], faces=self.artag_f, face_colors = self.artag_facecolors)
            artag_meshes.append(pyrender.Mesh.from_trimesh(artag_tm, smooth = False))


            artag_meshes_gt = []
            artag_tm_gt = trimesh.base.Trimesh(vertices=self.artag_r + [0.0, shift_estimate_sideways + shift_both_amount+shift_ground_truth, 0.0], faces=self.artag_f, face_colors = self.artag_facecolors_gt)
            artag_meshes_gt.append(pyrender.Mesh.from_trimesh(artag_tm_gt, smooth = False))



            if pmat is not None:
                pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat, bedangle)
                pmat_verts = np.array(pmat_verts)
                pmat_verts = np.concatenate((np.swapaxes(pmat_verts, 0, 1), np.ones((1, pmat_verts.shape[0]))), axis = 0)
                pmat_verts = np.swapaxes(np.matmul(transform_A, pmat_verts), 0, 1)[:, 0:3]
                pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors = pmat_facecolors)
                pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth = False)

                pmat_verts2, _, pmat_facecolors2 = self.get_3D_pmat_markers(pmat, bedangle, solidcolor = True)
                pmat_verts2 = np.array(pmat_verts2)
                pmat_verts2 = np.concatenate((np.swapaxes(pmat_verts2, 0, 1), np.ones((1, pmat_verts2.shape[0]))), axis = 0)
                pmat_verts2 = np.swapaxes(np.matmul(transform_B, pmat_verts2), 0, 1)[:, 0:3]
                pmat_tm2 = trimesh.base.Trimesh(vertices=pmat_verts2, faces=pmat_faces, face_colors = pmat_facecolors2)
                pmat_mesh2 = pyrender.Mesh.from_trimesh(pmat_tm2, smooth = False)

            else:
                pmat_mesh = None
                pmat_mesh2 = None


            #print "Viewing"
            if self.first_pass == True:

                for mesh_part in mesh_list:
                    self.scene.add(mesh_part)

                for mesh_part_gt in mesh_list_gt:
                    self.scene.add(mesh_part_gt)

                if pmat_mesh is not None:
                    self.scene.add(pmat_mesh)

                if pmat_mesh2 is not None:
                    self.scene.add(pmat_mesh2)

                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        self.scene.add(artag_mesh)

                for artag_mesh_gt in artag_meshes_gt:
                    if artag_mesh_gt is not None:
                        self.scene.add(artag_mesh_gt)


                lighting_intensity = 20.

                #self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                #                              point_size=2, run_in_thread=True, viewport_size=(1200, 1200))



                self.first_pass = False

                self.node_list = []
                for mesh_part in mesh_list:
                    for node in self.scene.get_nodes(obj=mesh_part):
                        self.node_list.append(node)

                self.node_list_gt = []
                for mesh_part_gt in mesh_list_gt:
                    for node in self.scene.get_nodes(obj=mesh_part_gt):
                        self.node_list_gt.append(node)



                self.artag_nodes = []
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh):
                            self.artag_nodes.append(node)
                self.artag_nodes_gt = []
                for artag_mesh_gt in artag_meshes_gt:
                    if artag_mesh_gt is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh_gt):
                            self.artag_nodes_gt.append(node)
                if pmat_mesh is not None:
                    for node in self.scene.get_nodes(obj=pmat_mesh):
                        self.pmat_node = node
                if pmat_mesh2 is not None:
                    for node in self.scene.get_nodes(obj=pmat_mesh2):
                        self.pmat_node2 = node

                camera_pose = np.eye(4)
                # camera_pose[0,0] = -1.0
                # camera_pose[1,1] = -1.0

                camera_pose[0, 0] = np.cos(np.pi/2)
                camera_pose[0, 1] = np.sin(np.pi/2)
                camera_pose[1, 0] = -np.sin(np.pi/2)
                camera_pose[1, 1] = np.cos(np.pi/2)
                rot_udpim = np.eye(4)

                rot_y = 180*np.pi/180.
                rot_udpim[1,1] = np.cos(rot_y)
                rot_udpim[2,2] = np.cos(rot_y)
                rot_udpim[1,2] = np.sin(rot_y)
                rot_udpim[2,1] = -np.sin(rot_y)
                camera_pose = np.matmul(rot_udpim,  camera_pose)

                camera_pose[0, 3] = 64*0.0286/2  # -1.0
                camera_pose[1, 3] = 1.2 + 0.8
                camera_pose[2, 3] = -1.0


                if viz_popup == True:
                    self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True,
                                                  lighting_intensity=10.,
                                                  point_size=5, run_in_thread=True, viewport_size=(1000, 1000))
                    #camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

                magnify =(64*.0286)

                camera = pyrender.OrthographicCamera(xmag=magnify, ymag = magnify)

                self.scene.add(camera, pose=camera_pose)


                light = pyrender.SpotLight(color=np.ones(3), intensity=250.0, innerConeAngle=np.pi / 10.0,
                                           outerConeAngle=np.pi / 2.0)
                light_pose = np.copy(camera_pose)
                # light_pose[1, 3] = 2.0
                light_pose[0, 3] = 0.8
                light_pose[1, 3] = -0.5
                light_pose[2, 3] = -2.5

                light_pose2 = np.copy(camera_pose)
                light_pose2[0, 3] = 2.5
                light_pose2[1, 3] = 1.0
                light_pose2[2, 3] = -5.0

                light_pose3 = np.copy(camera_pose)
                light_pose3[0, 3] = 1.0
                light_pose3[1, 3] = 5.0
                light_pose3[2, 3] = -4.0

                #light_pose2[0, 3] = 1.0
                #light_pose2[1, 3] = 2.0 #across
                #light_pose2[2, 3] = -1.5
                # light_pose[1, ]

                self.scene.add(light, pose=light_pose)
                self.scene.add(light, pose=light_pose2)
                self.scene.add(light, pose=light_pose3)




            else:
                if viz_popup == True:
                    self.viewer.render_lock.acquire()

                #reset the human mesh
                for idx in range(len(mesh_list)):
                    self.scene.remove_node(self.node_list[idx])
                    self.scene.add(mesh_list[idx])
                    for node in self.scene.get_nodes(obj=mesh_list[idx]):
                        self.node_list[idx] = node

                #reset the human mesh
                for idx in range(len(mesh_list_gt)):
                    self.scene.remove_node(self.node_list_gt[idx])
                    self.scene.add(mesh_list_gt[idx])
                    for node in self.scene.get_nodes(obj=mesh_list_gt[idx]):
                        self.node_list_gt[idx] = node

                #reset the artag meshes
                for artag_node in self.artag_nodes:
                    self.scene.remove_node(artag_node)
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        self.scene.add(artag_mesh)
                self.artag_nodes = []
                for artag_mesh in artag_meshes:
                    if artag_mesh is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh):
                            self.artag_nodes.append(node)

                #reset the artag meshes
                for artag_node_gt in self.artag_nodes_gt:
                    self.scene.remove_node(artag_node_gt)
                for artag_mesh_gt in artag_meshes_gt:
                    if artag_mesh_gt is not None:
                        self.scene.add(artag_mesh_gt)
                self.artag_nodes_gt = []
                for artag_mesh_gt in artag_meshes_gt:
                    if artag_mesh_gt is not None:
                        for node in self.scene.get_nodes(obj=artag_mesh_gt):
                            self.artag_nodes_gt.append(node)


                #reset the pmat mesh
                if pmat_mesh is not None:
                    self.scene.remove_node(self.pmat_node)
                    self.scene.add(pmat_mesh)
                    for node in self.scene.get_nodes(obj=pmat_mesh):
                        self.pmat_node = node


                #reset the pmat mesh
                if pmat_mesh2 is not None:
                    self.scene.remove_node(self.pmat_node2)
                    self.scene.add(pmat_mesh2)
                    for node in self.scene.get_nodes(obj=pmat_mesh2):
                        self.pmat_node2 = node



                #print self.scene.get_nodes()
                if viz_popup == True:
                    self.viewer.render_lock.release()
            #time.sleep(100)


        if viz_popup == False:
            r = pyrender.OffscreenRenderer(880, 880)
            # r.render(self.scene)
            color_render, depth = r.render(self.scene)
            # plt.subplot(1, 2, 1)
            plt.axis('off')


            #im_to_show = np.concatenate((color_render, color_im), axis = 1)
            im_to_show = np.copy(color_render)


            im_to_show = im_to_show[130-int(extend_top_bottom*300):750+int(extend_top_bottom*300), :, :]

            #plt.imshow(color)
            plt.imshow(im_to_show)
            # plt.subplot(1, 2, 2)
            # plt.axis('off')
            # plt.imshow(depth, cmap=plt.cm.gray_r) >> > plt.show()

            fig.set_size_inches(15., 10.)
            fig.tight_layout()
            #save_name = 'f_hbh_'+'{:04}'.format(self.pic_num)


            print "saving!"
            fig.savefig('/media/henry/multimodal_data_2/CVPR2020_study/'+save_name+'_v2.png', dpi=300)


            self.pic_num += 1
            #plt.show()
            #if self.pic_num == 20:
            #    print "DONE"
            #    time.sleep(1000000)
            #print "got here"

            #print X.shape


        return RESULTS_DICT