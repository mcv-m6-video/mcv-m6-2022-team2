# Week 3 - Object Detection and Tracking
The tasks corresponding to week 3 were devoted to explore both Object Detection and tracking algorithms in order to detect and identify vehicles in the *AI City dataset*.
To implement the Object Detection algorithms we have used the Detectron2 library using and testing the performance of both Faster R-CNN and RetinaNet models with ResNet-101 and ResNeXt-101 as backbone, respectively. The models were tested using the weights of the COCO-dataset, fine-tuned with the first 25% of the video sequence and 4-fold cross-validation in the same sequence.
Regarding the tracking, we have tested the Maximum Overlap method and the Kalman Filter using the IDF1 metric.

The division of the task is the following:
- Task 1.1: Object Detection with off-the-shelf weights
- Task 1.2: Object Detection fine-tuning the models with the first 25% of the sequence
- Task 1.3: Object Detection with a 4-fold cross-validation over the same sequence
- Task 2.1: Tracking by Maximum Overlap
- Task 2.2: Tracking by Kalman Filter

All the infromation regarding the experiments done during this week can be found at: [Slides](https://docs.google.com/presentation/d/1iI8YRSMnAx5lvk0_UCn_JQF1Z2GEYZMhchprLHi-dgc/edit?usp=sharing)

To run each section explained in the slides, we have created many files denoted as task_{id_task_{...}}.py, where {...} has some further information. These files are to be run as follows:

```
$ python week3/task_{id_task_{...}}.py
```
