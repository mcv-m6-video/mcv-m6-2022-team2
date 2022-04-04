import numpy as np
import cv2
from tqdm import tqdm
from OF_utils import draw_OF_magnitude_direction, OF_quiver_visualize
import pickle
import os
import matplotlib.pyplot as plt

frame0 = cv2.imread('../data_stereo_flow/training/image_0/000045_10.png')
frame1 = cv2.imread('../data_stereo_flow/training/image_0/000045_11.png')

frame0_gray = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY).astype(np.uint8)
frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY).astype(np.uint8)
                                                         
window_size = 32
search_area = 32
step = 10
method = 'euclidean'


w, h = frame1_gray.shape

mask = np.zeros((w + 2*search_area,h + 2*search_area))
mask[search_area:w + search_area, search_area:h + search_area] = 1

canvas0 = np.zeros((w + 2*search_area,h + 2*search_area))
canvas0[search_area:w + search_area, search_area:h + search_area] = frame0_gray

canvas1 = np.zeros((w + 2*search_area,h + 2*search_area))
canvas1[search_area:w + search_area, search_area:h + search_area] = frame1_gray

canvas0 = canvas0.astype(np.uint8)
canvas1 = canvas1.astype(np.uint8)
mask = mask.astype(np.uint8)
    
optical_flow = np.zeros((w,h,3))
optical_flow[:,:,2] = np.ones((w,h))

w_new, h_new = canvas0.shape

if not os.path.exists('./optical_flow_block.pickle'):
    for i1 in tqdm(range(search_area,w_new - search_area)):
        for j1 in tqdm(range(search_area,h_new - search_area),leave=False):
            window1 = canvas1[i1:i1+window_size,j1:j1+window_size]
            
            search_window = canvas0[i1 - search_area:np.min((i1 + window_size + search_area, w_new)),j1 - search_area:np.min((j1 + window_size + search_area, h_new))]
            mask = mask[i1 - search_area:np.min((i1 + window_size + search_area, w_new)),j1 - search_area:np.min((j1 + window_size + search_area, h_new))]
            
            if method == 'euclidean':
                res = cv2.matchTemplate(search_window,window1,cv2.TM_SQDIFF)
                _, _, (i_min,j_min), _ = cv2.minMaxLoc(res)

            i_flow = ((search_area + window_size)/2 - i_min)
            j_flow = ((search_area + window_size)/2 - j_min)
            optical_flow[i1 - search_area,j1 - search_area,:-1] = [i_flow,j_flow]

    with open('./optical_flow_block.pickle','wb') as f:
        pickle.dump(optical_flow,f)
        
else:
    with open('./optical_flow_block.pickle','rb') as f:
        optical_flow = pickle.load(f)
            
draw_OF_magnitude_direction(optical_flow)
OF_quiver_visualize(frame0_gray,optical_flow,10)