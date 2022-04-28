## Week 1

The task corresponding to week 1 were devoted to explore the available data in the datasets, as well as the metrics for evaluation.
We make use of the ground truth anotations available in AI City dataset to asses and understand the Avergage Precission and 
Intersection over Union metrics. Furthermore, we add Gaussian noise to the ground truth to explore how the metrics evolve.
We also evaluate detectors using the given predictions made by state-of-the-art techniques such as Mask RCNN, SSD and YOLOV3.

Then we move to the task of optical flow estimation with the Kitty dataset. For this task we aim to comprehensively assess the 
detections given by the Lucas-Kanade algorithm in some instances of the dataset. We use MSEN and PEPN evaluation metrics and 
then perform some visual analysis of the results. One of the goals is to find a well suited method to visualize the optical flow.

The division of the tasks is the following:
- Task 1: Intersection over Union (IoU) and mean Average Precision (mAP)
- Task 2: Evolution of IoU over time
- Task 3.1: Mean Square Error in Non-occluded areas (MSEN) and Percentage of Erroneous Pixels in Non-occluded areas (PEPN) metrics to evaluate Optical Flow
- Task 3.3: Analysis and Visualization of Optical Flow errors
- Task 4: Optical Flow visualization
 
 ![OF visualization](OF_plot.png)
 
All the infromation regarding the experiments done during this week can be found at: [Slides](https://docs.google.com/presentation/d/1--gSyRbA2TWpcgvf9KUmqyU1-4Lp5N8DZkfvTHhmimQ/edit?usp=sharing)

To run each section explained in the slides, we have created many files denoted as task_{id_task_{...}}.py, where {...} has some further information. These files are to be run as follows:

```
$ python week1/task_{id_task_{...}}.py
```