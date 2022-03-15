import numpy as np

def noise_bboxes(BBs,mean = 0, std = 1.0, dropout = 0.5, generate = 0.5):
    """
    This function corrupts to a list of bboxes in three ways:
    1. Adds Gaussian Noise to the size and position
    2. Drops bboxes from the list
    3. Generates new random bboxes 

    :param BBs: is a list of dictionaries of bboxes. Structure of the dict elements: {'name': 'car', 'bbox': [558.36, 94.45, 663.49, 169.31], 'confidence': 1.0}
    :param mean: mean of the Gaussian Noise
    :param std: standard deviation of the Gaussian Noise
    :param dropout: the probability of a bbox to be dropped from the list
    :param generate: for each bbox, the probability to create an extra random bbox
    :return: newBBs: list of coordinates of the corrupted bboxes. Each item is of the form [xmin, ymin, xmax, ymax]
    """

    newBBs = []
    for BB in BBs:

        dropped = np.random.choice([True,False],1,p = [dropout, 1-dropout])
        if not dropped: # Add Gaussian Noise to the bbox if not dropped
            [xmin, ymin, xmax, ymax] = BB['bbox']

            w, h = [xmax-xmin,ymax-ymin]

            xmin = xmin + np.random.normal(mean,std)
            ymin = ymin + np.random.normal(mean,std)

            xmax = xmin + (w + np.random.normal(mean,std))
            ymax = ymin + (h + np.random.normal(mean,std))

            BB['bbox'] = [xmin, ymin, xmax, ymax]
            newBBs.append(BB)
        
        generation = np.random.choice([True,False],1,p = [generate, 1-generate])
        if generation: # Generate new bbox
            w_max = 1920 # Width of the frame
            h_max = 1080 # Height of the frame

            xmin = np.random.uniform(0,1000)
            ymin = np.random.uniform(0,1000)

            xmax = xmin + np.random.uniform(0,w_max - xmin)
            ymax = ymin + np.random.uniform(0,h_max - ymin)

            confidence = np.round(np.random.uniform(0,1),3) # Random uniform confidence
            
            randomBB = {'name': BB['name'], 'bbox': [xmin, ymin, xmax, ymax], 'confidence': confidence}
            newBBs.append(randomBB)

    
    return newBBs