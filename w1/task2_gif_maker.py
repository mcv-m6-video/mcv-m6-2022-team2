import imageio
import numpy as np  
import glob
from PIL import Image

det_file_paths = ['det_mask_rcnn.txt', 'det_ssd512.txt', 'det_yolo3.txt']
for det in det_file_paths:
    for type in ['bboxes']:
        type_aux = type
        if type == 'meanIoU':
            type_aux = 'meanIou'
            
        frames = []
        for i in range(800,1000):
            index = '0' + str(i) if (i < 1000) else str(i)
            img = Image.open(f'{det.split(".txt")[0]}_{type}/{type_aux}_{index}.png')
            new_img = img.resize((640, 360))
            frames.append(new_img)
            
        frame_one = frames[0]
        frame_one.save(f'{det.split(".txt")[0]}_{type}.gif', format="GIF", append_images=frames,
                    save_all=True, duration=70, loop=0)

#Create reader object for the gif
det_file_paths = ['det_mask_rcnn.txt', 'det_ssd512.txt', 'det_yolo3.txt']

for det in det_file_paths:
    gif1 = imageio.get_reader(f'{det.split(".txt")[0]}_bboxes.gif')
    gif2 = imageio.get_reader(f'{det.split(".txt")[0]}_meanIoU.gif')

    #If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length()) 

    #Create writer object
    new_gif = imageio.get_writer(f'{det.split(".txt")[0]}.gif')

    for frame_number in range(number_of_frames):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        #here is the magic
        new_image = np.vstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()    
    new_gif.close()