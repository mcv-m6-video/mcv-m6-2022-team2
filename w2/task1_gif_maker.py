import imageio
import numpy as np  
import glob
from PIL import Image

det_file_paths = ['frames', 'plot_mean']
for det in det_file_paths:
   
    frames = []
    for i in range(0,100):
        index = '0' + str(i) if (i < 10) else str(i)
        img = Image.open("task1_plots/" + det + "/frame_" + index + '.png')
        if det=='frames':
            img = img.resize((640, 360))
        frames.append(img)
        
    frame_one = frames[0]
    frame_one.save(det + ".gif", format="GIF", append_images=frames,
                save_all=True, duration=140, loop=0)

#Create reader object for the gif
det_file_paths = ['frames', 'plot_mean']

for det in det_file_paths:
    gif1 = imageio.get_reader('frames.gif')
    gif2 = imageio.get_reader('plot_mean.gif')

    #If they don't have the same number of frame take the shorter
    number_of_frames = min(gif1.get_length(), gif2.get_length()) 

    #Create writer object
    new_gif = imageio.get_writer('frames_plot_mean.gif')

    for frame_number in range(number_of_frames):
        img1 = gif1.get_next_data()
        img2 = gif2.get_next_data()
        #here is the magic
        new_image = np.vstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()    
    new_gif.close()