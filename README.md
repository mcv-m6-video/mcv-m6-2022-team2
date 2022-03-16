## Master in Computer Vision, Barcelona 2021-2022 - Module 6: Video Analysis

This module is devoted to explore Computer Vision techniques to the problem of Road Traffic Monitoring, 
using video sequences of vehicles in roads. Throuought the weeks, we present different strategies for estimating 
background information, detecting foreground objects, estimating optical flow and tracking objects.

Mainly two datasets are used for this project:

- *KITTI dataset*: This dataset offers pairs of images corresponding to a video sequence of a road, as well as the optical flow ground truth.
  Train set: 194 instances, Test set: 195 instances (image pairs + Optical flow GT)

- *AI City dataset*: This dataset offers 3.25 hours of videos showing road interections and driving vehicles. 40 cameras were used in 10 different intersections.
The dataset offers the frame-by-frame bounding boxes of each vehicle, giving a total of 229,680 bounding boxes for 666 different vehicles.

## Team 2

| Members | Contact |
| :---         |   :---    | 
| Igor Ugarte Molinet | igorugarte.cvm@gmail.com | 
| Juan Antonio Rodríguez García | juanantonio.rodriguez@upf.edu  |
| Francesc Net Barnès | francescnet@gmail.com  |
| David Serrano Lozano | 99d.serrano@gmail.com |

## Week 1

This week tasks are devoted to explore the data available in the datasets, as well as the metrics for evaluation.
We make use of the ground truth anotations available in AI City dataset to asses and understand the Avergage Precission and 
Intersection over Union metrics. Furthermore, we add Gaussian noise to the ground truth to explore how the metrics evolve.
We also evaluate detectors using the given predictions made by state-of-the-art techniques such as Mask RCNN, SSD and YOLOV3.

Then we move to the task of optical flow estimation with the Kitty dataset. For this task we aim to comprehensively assess the 
detections given by the Lucas-Kanade algorithm in some instances of the dataset. We use MSEN and PEPN evaluation metrics and 
then perform some visual analysis of the results. One of the goals is to find a well suited method to visualize the optical flow.
