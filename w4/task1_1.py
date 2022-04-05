import numpy as np
import cv2
from tqdm import tqdm
from OF_utils import *
import pickle
import os
import matplotlib.pyplot as plt


root = '../'
gt_path = os.path.join(root, 'data_stereo_flow/training/flow_noc')
ground_truth = read_OF(os.path.join(gt_path, '000045_10.png'))

frame0 = cv2.imread('../data_stereo_flow/training/image_0/000045_10.png')
frame1 = cv2.imread('../data_stereo_flow/training/image_0/000045_11.png')

frame0_gray = cv2.cvtColor(frame0,cv2.COLOR_BGR2GRAY).astype(np.uint8)
frame1_gray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY).astype(np.uint8)
                                                         
window_size = 32
search_area = 32
method = 'euclidean'

window_sizes = [4,8,16,32,64,128]
search_areas = [4,8,16,32,64,128]

msen = []
pepn = []
times = []
for i in tqdm(range(len(window_sizes))):
    for j in tqdm(range(len(search_areas)),leave=False):
        with tqdm(total=100) as t: #initialize clock
            
            window_size = window_sizes[i]
            
            search_area = search_areas[j]
            print(search_area)
            
            w, h = frame1_gray.shape

            canvas0 = np.zeros((w + 2*search_area,h + 2*search_area))
            canvas0[search_area:w + search_area, search_area:h + search_area] = frame0_gray

            canvas1 = np.zeros((w + 2*search_area,h + 2*search_area))
            canvas1[search_area:w + search_area, search_area:h + search_area] = frame1_gray

            canvas0 = canvas0.astype(np.uint8)
            canvas1 = canvas1.astype(np.uint8)
                
            optical_flow = np.zeros((w,h,3))
            optical_flow[:,:,2] = np.ones((w,h))

            w_new, h_new = canvas0.shape
            
            
            """ if not os.path.exists('./optical_flow_block.pickle'): """
            for i1 in tqdm(range(search_area,w_new - search_area),leave=False):
                for j1 in range(search_area,h_new - search_area):
                    window1 = canvas1[i1:i1+window_size,j1:j1+window_size]
                    
                    search_window = canvas0[i1 - search_area:np.min((i1 + window_size + search_area, w_new)),j1 - search_area:np.min((j1 + window_size + search_area, h_new))]
                    
                    if method == 'euclidean':
                        res = cv2.matchTemplate(search_window,window1,cv2.TM_SQDIFF)
                        _, _, (i_min,j_min), _ = cv2.minMaxLoc(res)

                    i_flow = ((search_area + window_size)/2 - i_min)
                    j_flow = ((search_area + window_size)/2 - j_min)
                    optical_flow[i1 - search_area,j1 - search_area,:-1] = [i_flow,j_flow]
                
        

            """ with open('./optical_flow_block.pickle','wb') as f:
                pickle.dump(optical_flow,f) """
                    
            """ else:
                with open('./optical_flow_block.pickle','rb') as f:
                    optical_flow = pickle.load(f) """
                        
            """ draw_OF_magnitude_direction(optical_flow)
            OF_quiver_visualize(frame0_gray,optical_flow,10) """

            # Save computation times
            t.update()
            times.append(t.format_dict['elapsed'])

            # Compute the vector difference between the ground truth and the detection OF
            nocc_error, error = compute_vector_dif(ground_truth=ground_truth, detection=optical_flow)

            msen.append(compute_msen(nocc_error))
            pepn.append(compute_pepn(nocc_error))
            
with open('./times.pickle','wb') as f:
    pickle.dump(times,f)
    
with open('./msen.pickle','wb') as f:
    pickle.dump(msen,f)
    
with open('./pepn.pickle','wb') as f:
    pickle.dump(pepn,f)
            
