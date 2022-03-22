import imageio
import numpy as np  
import glob
from PIL import Image

def generate_gifs(paths):
    for path in paths:
    
        frames = []
        for i in range(0,100):
            index = '0' + str(i) if (i < 10) else str(i)
            img = Image.open(path + "/frame_" + index + '.png')
            if path=='frames':
                img = img.resize((640, 360))
            frames.append(img)
            
        frame_one = frames[0]
        frame_one.save(path + ".gif", format="GIF", append_images=frames,
                    save_all=True, duration=140, loop=0)
        
det_file_paths = ['frames', 'plot_mean']
generate_gifs(det_file_paths)

#Create reader object for the gif
det_file_paths = ['frames', 'plot_mean']

def join_gif_vertically(gif1_name,gif2_name):
    gif1 = imageio.get_reader(gif1_name)
    gif2 = imageio.get_reader(gif2_name)

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