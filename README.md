# CarND-VehicleDetection

#Vehicle Detection and Tracking using Computer Vision
Same report on Medium
https://medium.com/@arnaldogunzi/vehicle-detection-and-tracking-using-computer-vision-baea4df65906#.9wmy0vv0i

##1. Introduction

In Project 5 of the great Udacity Self Driving car nanodegree, the goal is to use computer vision techniques to detect vehicles in a road.

Visually, to do something like this:
![](https://cdn-images-1.medium.com/max/873/1*cbZHwe7kGKkxn9RNVy9mNA.jpeg)


Part of the final video:
https://youtu.be/UUzWRUmykxc
Partial video of Vehicle Detection Project

##2. How to do this?

In the project, computer vision methods are used. It is not the only technique — deep learning could be used instead. The advantage of computer vision is that we can analyze each step, in a straightforward way. Deep learning, in contrast, is more like a black box.

Steps taken:

    Analysis of data
    HOG feature extraction to find the features of images
    Train a Support Vector Machine classifier
    Implement a sliding-window and use the classifier to search the vehicle
    Generation of Heatmap with detection and bound box on vehicles
    Smooth the results and eliminate false positives in video

##3. Analysis of data

Which of the two pictures is a car?
![](https://cdn-images-1.medium.com/max/873/1*5r5CM1CqalNYzEulS44IbQ.jpeg)

We, humans, are very good in detecting a car in a picture.

But an algorithm must be learned to do so. The best way is to train the algorithm with a lot of images, labeled “cars” and “non-cars”.

These images have to be extracted from real world videos and images, and correctly labeled. Udacity provided 8.792 images of car and 8.968 images of non-cars, from sources listed in the attachments. The images have 64 x 64 pixels.
![](https://cdn-images-1.medium.com/max/873/1*KPR1eCUpoqTfIk3zQ7LaEA.jpeg)
Examples of data

The quantity and quality of these sample images is critical to the process. Bad quality images will make the classifier do wrong predictions.

These data are separated in training (80%) and validation sets (20%), and their order is randomized.

The code of data analysis is in file “Data_Exploration.ipynb”, in Github link.

#4. The HOG feature extractor

The HOG extractor is the heart of the method described here. It is a way to extract meaningful features of a image. It captures the “general aspect” of cars, not the “specific details” of it. It is the same as we, humans, do: in a first glance, we locate the car, not the make, the plate, the wheel, or other small detail.

HOG stands for “Histogram of Oriented Gradients”. Basically, it divides an image in several pieces. For each piece, it calculates the gradient of variation in a given number of orientations.
Example of HOG detector — the idea is the image on the right to capture the essence of original image
![](https://cdn-images-1.medium.com/max/873/1*lDQ20tXlqcMcxvD9l-MyPg.jpeg)



The following gif shows the effect of the number of pixels per cell.
![](https://cdn-images-1.medium.com/max/873/1*bzzdKbayJjG1f_X2XNc52Q.gif)



The gradient is the variation of colors: the greater the variation, the greater the gradient. It’s a partial derivative. And direction is controlled by the number of orientations.

The following gif shows the effect of the number of orientations.
![](https://cdn-images-1.medium.com/max/873/1*ar8UjeWdrWvoZgjxF1KCPw.gif)

The final step of the algorithm is to take a histogram of directions and orientations, make a block regularization and return an single dimension array of data to be fed in a classifier. This post is a good introduction: http://www.learnopencv.com/histogram-of-oriented-gradients/

All of this is packed inside the HOG opencv function.

The less the number of pixels per cells (and other parameters), more general the data, and the more, more specific. By playing with the parameters, I found that orientations above 7, and 8 pixels per cell are enough to identify a car.

The HOG algorithm is robust for small variations and different angles. But, on the other way, it can detect also some image that has the same general aspect of the car, but it not a car at all — the so called “False positives”.

The code of this section is in “Data_Exploration.ipynb”, in the Github link.

The same could be made with a color detector, in addition to HOG detector. Because the HOG only classifier was good enough, I used it in the rest of project.

A brief digression: In previous project with deep learning (P3), I used image augmentation (flipping, rotating) to make the algorithm more robust. In this case, it is not necessary, because the HOG detector already is robust in the sense that captures the “general aspect” of the car. The image augmentation would be necessary only if the database were unbalanced (more left images than right, more cars than non-cars).

##5. The classifier

The next step is to train a classifier. It receives the cars / non-cars data transformed with HOG detector, and returns if the sample is or is not a car.

![](https://cdn-images-1.medium.com/max/873/1*MStS2dBWSZo8iJPiL2_uXg.png)
Support Vector Machine

In this case, I used a Support Vector Machine Classifier (SVC), with linear kernel, based on function SVM from scikit-learn.

In a simplified way, a SVM finds a line that better divides two sets.
Following link from opencv is a good introduction. http://docs.opencv.org/2.4/doc/tutorials/ml/introduction_to_svm/introduction_to_svm.html

Some of the tests made to define parameters are in table below:
![](https://cdn-images-1.medium.com/max/873/1*3chREro9oIGwCYn_wGqZ8A.jpeg)
In the end, I used color space YCrCb, all channels.

To do the training of the classifier, the data was scaled using SkLearn RobustScaler. It is important to scale the images, because some of them can be to bright or too dark, distorting the classifier.

This classifier and the scaler were saved using pickle library, to be used later in the classification of the video image.

The code for classifier is in file Hog_classifier.py, in Github.

##6. Applying the classifier in a image frame

Up to now, we can feed a classifier with an 64 x 64 pixels image and get a result from it: car or non-car.

In order to do this in an entire image (720 x 1280), we use a sliding window.

First, I cropped just the interest region. Then sliced the image in small frames, resized it to the right size (64x64), and applied the classification algorithm.

It’s possible to define the size of the window in the image, or to resize the original image and crop a fixed window. I used this later option. To have an idea of the size of the window in the image, I plotted some of them.
Example of scale 1.1
![](https://cdn-images-1.medium.com/max/873/1*csYbKRv4wtfTweOKWjF3JA.jpeg)

The car can appear in different sizes. Then, we apply different windows sizes over the image. In the end, I used following scales.

scales = [1.1,1.4, 1.8, 2.4, 2.9, 3.4]
![](https://cdn-images-1.medium.com/max/873/1*m9GdHB2oJiKy1S8zjnbzeA.jpeg)

Example of scale 3.4

It is expected bigger images near the bottom of the image. And smaller images in the top of it. Then, I divided the image frame in two, with smaller windows in the upper half, and bigger ones in the lower.

It is analogous as what we do when looking for something. We scan the image with our eyes, till we find something that fits what we are expecting to find.

The problem of this method is that it is computationally expensive. I was thinking on some weak pre-filter to do a fast evaluation of each frame, then passing it to the more robust filter. But I couldn’t figured out a pre-filter fast enough to compensate this extra step.

##7. Smoothing

One problem of the method described so far is that it detects a lot of false positives: images that are not cars but fool the SVC. The image below is an example of it.

![](https://cdn-images-1.medium.com/max/873/1*UiVpHu7Ae8xg_PvibgnFUQ.jpeg)

The false positives to the left are not so that false, because actually there are cars there

To avoid false positives, we do an average over 10 frames of images. A real car is probable to keep appearing in the image. A false positive will disappear.


The code for sliding window and smoothing is in jupyter notebook file VehicleDetection_Video.ipynb, in Github.

##8. Final video

Here is the link of final video, also with Advanced Lane Finding from previous Project.
https://youtu.be/f2OJ2ePPC2k

The Github link is https://github.com/asgunzi/CarND-VehicleDetection. I used Ubuntu 16.04, 64 bits. Python 3, OpenCV and Scikit-learn to do this project.

##9. Conclusion and Discussion

Detection of cars is a difficult problem. The sliding window method is expensive, in the sense that it takes too long to process (10 min to process 1 min). For a real-time application, it has to be optimized, say using parallel processing. And this method is likely to find a lot of false positives, even averaging frames.

The described method is likely to fail in cases where it wasn’t trained for: a bicycle, a pedestrian, a dog crossing the street, a cone in the middle of the road.

In real world car driving, we use the rear mirror to look behind. In the same way, we should have a rear camera to identify cars behind us.

I would like to thanks Udacity for providing this high level challenge and valuable guidance on this project.

##10. Tips and Tricks

The code is largely based on classroom examples, with some adaptations. Here follows some tricks not in the classroom examples.

Interact

I used Jupyter notebook to do the image analysis. It has a good addin, the Interact (http://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html). It makes very easy to understand the effect of different parameters.

Moviepy
To save a subclip

clip1 = VideoFileClip(“project_video.mp4”).subclip(20,24) 
Process from 20 to 24 seconds

To save a (problematic) image of the clip

#Save clip 
myclip = clip1.to_ImageClip(t=’00:00:06') # frame at t=1 hour.
myclip.save_frame(“frame06.jpeg”)

Average frames

To do the average of different frames, I used global variables.
One to count the frame: 1, 2, …. Other to save the rectangles detected in each frame.
To make calculation easier, I used a simple python list to save the rectangles. If I want to store 10 frames, this list had 10 positions. Then, by doing a simple module of the count of frame, I replaced the old information of the list with the new one.

241 % 10 = 1
231 % 10 = 1
221 % 10 = 1

Example: frame 241 equals 1 mod 10. Then, I replaced the position 1 of the list (containg previously frame 231) with the new information of this frame.

Creating a diagnostic view
The diagnostic window, with some informations on what the algorithm is generating, is a good way to calibrate parameters.

It is easy to create this. Because an image is just an rectangular array of numbers for each color dimension. An array can be deleted, copied, inserted, resized, and so on. 
 
For the diagnostic view, I just resized a grayscale version of the frame, added the heatmap information, and inserted it in a position it doesn’t disturb the movie.

Portion of the code:

dimg[0:sizeY,0:sizeX,0]=res_img_gray_R +heat3
dimg[0:sizeY,0:sizeX,1]=res_img_gray
dimg[0:sizeY,0:sizeX,2]=res_img_gray

##Attachments

Previous projects:
- Advanced Lane Finding

- Teaching a car to drive

Links:
- Udacity (http://www.udacity.com)
- OpenCV (http://opencv-python-tutroals.readthedocs.io/en/latest/)
- Scikit learn (http://scikit-learn.org)
- Moviepy (http://zulko.github.io/moviepy/)
- Interact (http://ipywidgets.readthedocs.io/)

Sources of images:

GTI vehicle image database — http://www.gti.ssr.upm.es/data/Vehicle_database.html
KITTI vision benchmark suite — http://www.cvlibs.net/datasets/kitti/

Udacity labeled set https://github.com/udacity/self-driving-car/tree/master/annotations
