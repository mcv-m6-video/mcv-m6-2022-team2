# Week 2
The tasks corresponding to week 2 presented the problem of background estimation from video sequences, by means of Gaussian modelling methods.
This family of methods pose the problem as a statistical model, where each pixel of each frame is modelled as a Random Variable
with additive Gaussian noise. The mean and variance of the noise is set according to the mean and std over a set of frames, 
specifically the initial 25% of frames. This results in a mean and std matrix, with the shape of the input images, 
that will be our model for the background. By comparing consecutive new frames with the mean, we can detect sudden 
changes in the pixel values, and hence moving objects. Furthermore, the detections are converted into connected components 
to perform foreground estimation.

We tested different versions of the model by applying an adaptive scheme for updating the mean and std, and perform hyperparameter search
for tuning the alpha and rho parameters. The method is compared with state of the art models. Finally, we test the 3D Gaussian model using different color spaces.

The division of the tasks is the following: 
- Task1: Modeling of the background with a single gaussian estimation
- Task2: Modeling of the background with a single gaussian estimation adaptatively
- Task3: Background removal algorithms in comparison with our method
- Task4: Modeling of the background with a single gaussian estimation adaptatively (RGB case)

All the infromation regarding the experiments done during this week can be found at: [Slides](https://docs.google.com/presentation/d/1PknD9ThP7xNblwPMWfg3HDnbSZza3tuVdbl8uXHcQ94/edit?usp=sharing)

To run each section explained in the slides, we have created many files denoted as task_{id_task_{...}}.py, where {...} has some further information. These files are to be run as follows:

```
$ python week2/task_{id_task_{...}}.py
```