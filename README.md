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

## Week 2
This week tasks present the problem of background estimation from video sequences, by means of Gaussian modelling methods.
This family of methods pose the problem as a statistical model, where each pixel of each frame is modelled as a Random Variable
with additive Gaussian noise. The mean and variance of the noise is set according to the mean and std over a set of frames, 
specifically the initial 25% of frames. This results in a mean and std matrix, with the shape of the input images, 
that will be our model for the background. By comparing consecutive new frames with the mean, we can detect sudden 
changes in the pixel values, and hence moving objects. Furthermore, the detections are converted into connected components 
to perform foreground estimation.

We test different versions of the model by applying an adaptive scheme for updating the mean and std, and perform hyperparameter search
for tuning the alpha and rho parameters. The method is compared with state of the art models. Finally, we test the 3D Gaussian model using different color spaces.

The division of the tasks is the following: 
⋅⋅* Task1: Modeling of the background with a single gaussian estimation
⋅⋅* Task2: Modeling of the background with a single gaussian estimation adaptatively
⋅⋅* Task3: Background removal algorithms in comparison with our method
⋅⋅* Task4: Modeling of the background with a single gaussian estimation adaptatively (RGB case)

To execute tasks (1 to 4) on week2:
```
python task{id_task}.py
```
For example to execute task 3:
```
python task3.py
```
