# Author: Deepak Pathak (c) 2016

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals
import numpy as np
from PIL import Image
import time
import argparse
from pyflow import pyflow
from OF_utils import *
import cv2
import os

parser = argparse.ArgumentParser(
    description='Demo for python wrapper of Coarse2Fine Optical Flow')
parser.add_argument(
    '-viz', dest='viz', action='store_true',
    help='Visualize (i.e. save) output of flow.')
args = parser.parse_args()

root = '../../data'
gt_path = os.path.join(root, 'data_stereo_flow/training/flow_noc')
ground_truth = read_OF(os.path.join(gt_path, '000045_10.png'))

im1 = np.array(Image.open('../../data/data_stereo_flow/training/colored_0/000045_10.png'))
im2 = np.array(Image.open('../../data/data_stereo_flow/training/colored_0/000045_11.png'))
im1 = im1.astype(float) / 255.
im2 = im2.astype(float) / 255.

# Flow Options:
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

s = time.time()
u, v, im2W = pyflow.coarse2fine_flow(
    im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
    nSORIterations, colType)
e = time.time()
print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
    e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
flow = np.concatenate((u[..., None], v[..., None]), axis=2)
np.save('outFlow.npy', flow)

print(flow.shape)

flow_occ = np.ones((flow.shape[0], flow.shape[1], flow.shape[2]+1))
flow_occ[:,:,:-1] = flow

draw_OF_magnitude_direction(flow_occ)
OF_quiver_visualize(im1,flow_occ,10)
nocc_error, error = compute_vector_dif(ground_truth=ground_truth, detection=flow)

msen = compute_msen(nocc_error)
pepn = compute_pepn(nocc_error)

print(msen,pepn)