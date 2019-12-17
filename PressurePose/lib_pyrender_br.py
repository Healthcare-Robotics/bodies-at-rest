#!/usr/bin/env python

#Bodies at Rest: Code to do rendering.
#(c) Henry M. Clever
#Major updates made for CVPR release: December 10, 2019


try:
    import open3d as o3d
except:
    print "CANNOT IMPORT 03D. POINT CLOUD PROCESSING WON'T WORK"

import trimesh
import pyrender
import pyglet

import numpy as np
import random
import copy
from time import sleep

import math
from random import shuffle


import time as time
import matplotlib.cm as cm #use cm.jet(list)

import cPickle as pickle
def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)



import sys
sys.path.insert(0, '../lib_py')


class pyRenderMesh():
    def __init__(self, render):

        # terms = 'f', 'frustum', 'background_image', 'overdraw', 'num_channels'
        # dterms = 'vc', 'camera', 'bgcolor'
        self.first_pass = True
        self.render = render
        if True:# render == True:
            self.scene = pyrender.Scene()

            self.human_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.0, 0.0 ,0.0])#[0.05, 0.05, 0.8, 0.0])#
            self.human_mat_GT = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.0, 0.3, 0.0 ,0.0])
            self.human_arm_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.8 ,1.0])
            self.human_mat_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.3, 0.3, 0.3 ,0.5])
            self.human_bed_for_study = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.7, 0.7, 0.2 ,0.5])
            self.human_mat_D = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.1, 0.1, 0.1, 1.0], alphaMode="BLEND")

            mesh_color_mult = 0.25

            self.mesh_parts_mat_list = [
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 0. / 255., mesh_color_mult * 0. / 255., mesh_color_mult * 0. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 31. / 255., mesh_color_mult * 120. / 255., mesh_color_mult * 180. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 251. / 255., mesh_color_mult * 154. / 255., mesh_color_mult * 153. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 227. / 255., mesh_color_mult * 26. / 255., mesh_color_mult * 28. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 178. / 255., mesh_color_mult * 223. / 255., mesh_color_mult * 138. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 51. / 255., mesh_color_mult * 160. / 255., mesh_color_mult * 44. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 253. / 255., mesh_color_mult * 191. / 255., mesh_color_mult * 111. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 255. / 255., mesh_color_mult * 127. / 255., mesh_color_mult * 0. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 202. / 255., mesh_color_mult * 178. / 255., mesh_color_mult * 214. / 255., 0.0]),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=[mesh_color_mult * 106. / 255., mesh_color_mult * 61. / 255., mesh_color_mult * 154. / 255., 0.0])]

        self.pic_num = 0


    def get_3D_pmat_markers(self, pmat):

        pmat_reshaped = pmat.reshape(64, 27)

        pmat_colors = cm.jet(pmat_reshaped/100)
        #print pmat_colors.shape
        pmat_colors[:, :, 3] = 0.7 #pmat translucency

        pmat_xyz = np.zeros((65, 28, 3))
        pmat_faces = []
        pmat_facecolors = []

        for j in range(65):
            for i in range(28):

                pmat_xyz[j, i, 1] = i * 0.0286# /1.06# * 1.02 #1.0926 - 0.02
                pmat_xyz[j, i, 0] = ((64 - j) * 0.0286) #* 1.04 #/1.04#1.1406 + 0.05 #only adjusts pmat NOT the SMPL person
                pmat_xyz[j, i, 2] = 0.075#0.12 + 0.075

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

        return pmat_verts, pmat_faces, pmat_facecolors



    def reduce_by_cam_dir(self, vertices, faces, camera_point, transform):

        vertices = np.array(vertices)
        faces = np.array(faces)


        #kill everything thats hanging off the side of the bed
        vertices[vertices[:, 0] < 0 + transform[0], 2] = 0
        vertices[vertices[:, 0] > (0.0286 * 64  + transform[0])*1.04, 2] = 0
        vertices[vertices[:, 1] < 0 + transform[1], 2] = 0
        vertices[vertices[:, 1] > 0.0286 * 27 + transform[1], 2] = 0

        # find normal of every mesh triangle
        tri_norm = np.cross(vertices[faces[:, 1], :] - vertices[faces[:, 0], :],
                            vertices[faces[:, 2], :] - vertices[faces[:, 0], :])

        # convert normal to a unit vector
        tri_norm = tri_norm/np.linalg.norm(tri_norm, axis = 1)[:, None]

        tri_norm[tri_norm[:, 2] == -1, 2] = 1

        tri_to_cam = camera_point - vertices[faces[:, 0], :] ## triangle to camera vector
        tri_to_cam = tri_to_cam/np.linalg.norm(tri_to_cam, axis = 1)[:, None]

        angle_list = tri_norm[:, 0]*tri_to_cam[:, 0] + tri_norm[:, 1]*tri_to_cam[:, 1] + tri_norm[:, 2]*tri_to_cam[:, 2]
        angle_list = np.arccos(angle_list) * 180 / np.pi

        angle_list = np.array(angle_list)

        faces = np.array(faces)
        faces_red = faces[angle_list < 90, :]

        return list(faces_red)



    def downspl_pc_get_normals(self, pc, camera_point):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        #print("Downsample the point cloud with a voxel of 0.01 m)
        downpcd = o3d.geometry.voxel_down_sample(pcd, voxel_size=0.01)

        o3d.geometry.estimate_normals(
            downpcd,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,
                                                              max_nn=30))

        o3d.geometry.orient_normals_towards_camera_location(downpcd, camera_location=np.array(camera_point))
        points = np.array(downpcd.points)
        normals = np.array(downpcd.normals)

        return points, normals


    def get_human_mesh_parts(self, smpl_verts, smpl_faces, segment_limbs = False):

        if segment_limbs == True:
            segmented_dict = load_pickle('../lib_py/segmented_mesh_idx_faces.p')
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




    def render_3D_data(self, camera_point, pmat, pc = None, smpl_verts_gt = None, smpl_faces = None, segment_limbs = False):

        #this is for the pressure mat
        if pmat is not None:
            pmat_verts, pmat_faces, pmat_facecolors = self.get_3D_pmat_markers(pmat)
            pmat_tm = trimesh.base.Trimesh(vertices=pmat_verts, faces=pmat_faces, face_colors = pmat_facecolors)
            pmat_mesh = pyrender.Mesh.from_trimesh(pmat_tm, smooth = False)

        else:
            pmat_mesh = None

        #this is for the real data ground truth - point cloud
        if pc is not None:
            #downsample the point cloud and get the normals using open3D libraries
            pc_red, pc_red_norm = self.downspl_pc_get_normals(pc, camera_point)
            pc_greysc_color = 0.0 * (pc_red[:, 2:3] - np.max(pc_red[:, 2])) / (np.min(pc_red[:, 2]) - np.max(pc_red[:, 2]))
            pc_mesh = pyrender.Mesh.from_points(pc_red, colors=np.concatenate((pc_greysc_color, pc_greysc_color, pc_greysc_color), axis=1))
        else:
            pc_mesh = None


        #this is for the ground truth mesh. But you can also use this on a network output mesh.
        mesh_list_seg = []
        if smpl_verts_gt is not None:
            smpl_verts_quad_GT = np.concatenate((smpl_verts_gt, np.ones((smpl_verts_gt.shape[0], 1))), axis=1)
            smpl_verts_quad_GT = np.swapaxes(smpl_verts_quad_GT, 0, 1)

            transform_A = np.identity(4) #modify this if you want to move the mesh around

            smpl_verts_A_GT = np.swapaxes(np.matmul(transform_A, smpl_verts_quad_GT), 0, 1)[:, 0:3] #gt over pressure mat

            human_mesh_vtx_GT, human_mesh_face_GT = self.get_human_mesh_parts(smpl_verts_A_GT, smpl_faces, segment_limbs=segment_limbs)


            # only use the vertices that are facing the camera
            if camera_point is not None:
                camera_point_A = np.matmul(transform_A, np.array([camera_point[0], camera_point[1], camera_point[2], 1.0]))[0:3]
                human_mesh_face_GT_red = []
                for part_idx in range(len(human_mesh_vtx_GT)):
                    human_mesh_face_GT_red.append(self.reduce_by_cam_dir(human_mesh_vtx_GT[part_idx],
                                                                            human_mesh_face_GT[part_idx],
                                                                            camera_point_A, transform_A[0:3, 3]))
            else:
                human_mesh_face_GT_red = human_mesh_face_GT



            #GET SEGMENTED LIMBS
            tm_list_seg = []
            for idx in range(len(human_mesh_vtx_GT)):
                tm_curr = trimesh.base.Trimesh(vertices=np.array(human_mesh_vtx_GT[idx]),
                                               faces = np.array(human_mesh_face_GT_red[idx]))#,
                                               #vertex_colors = np.array(norm_colors[idx]))
                tm_list_seg.append(tm_curr)

            if segment_limbs == True:
                wf = False
            else:
                wf = True

            for idx in range(len(tm_list_seg)):
                mesh_list_seg.append(pyrender.Mesh.from_trimesh(tm_list_seg[idx], material = self.mesh_parts_mat_list[idx], wireframe = wf))
                #mesh_list_seg.append(pyrender.Mesh.from_trimesh(tm_list_seg[idx], smooth = False))





        #print "Viewing"
        if self.first_pass == True:

            if pc_mesh is not None:
                self.scene.add(pc_mesh)

            if pmat_mesh is not None:
                self.scene.add(pmat_mesh)

            if smpl_verts_gt is not None:
                for mesh_part_seg in mesh_list_seg:
                    self.scene.add(mesh_part_seg)
                if len(mesh_list_seg) > 1:
                    for i in range(10 - len(mesh_list_seg)):
                        self.scene.add(mesh_part_seg)  # add fillers in


            lighting_intensity = 20.

            self.viewer = pyrender.Viewer(self.scene, use_raymond_lighting=True, lighting_intensity=lighting_intensity,
                                          point_size=2, run_in_thread=True, viewport_size=(1200, 1200))


            self.first_pass = False

            if pc_mesh is not None:
                for node in self.scene.get_nodes(obj=pc_mesh):
                    self.point_cloud_node_mesherr2 = node

            if pmat_mesh is not None:
                for node in self.scene.get_nodes(obj=pmat_mesh):
                    self.pmat_node = node

            if smpl_verts_gt is not None:
                self.node_list_seg = []
                for mesh_part_seg in mesh_list_seg:
                    for node in self.scene.get_nodes(obj=mesh_part_seg):
                        self.node_list_seg.append(node)
                if len(mesh_list_seg) > 1:
                    for i in range(10 - len(mesh_list_seg)):
                        for node in self.scene.get_nodes(obj=mesh_part_seg):
                            self.node_list_seg.append(node)


        else:
            self.viewer.render_lock.acquire()

            #reset the pmat mesh
            if pmat_mesh is not None:
                self.scene.remove_node(self.pmat_node)
                self.scene.add(pmat_mesh)
                for node in self.scene.get_nodes(obj=pmat_mesh):
                    self.pmat_node = node

            #reset the point cloud
            if pc_mesh is not None:
                self.scene.remove_node(self.point_cloud_node_mesherr2)
                self.scene.add(pc_mesh)
                for node in self.scene.get_nodes(obj=pc_mesh):
                    self.point_cloud_node_mesherr2 = node

            # reset the segmented human mesh
            if smpl_verts_gt is not None:
                for idx in range(len(mesh_list_seg)):
                    self.scene.remove_node(self.node_list_seg[idx])
                    self.scene.add(mesh_list_seg[idx])
                    for node in self.scene.get_nodes(obj=mesh_list_seg[idx]):
                        self.node_list_seg[idx] = node

            self.viewer.render_lock.release()