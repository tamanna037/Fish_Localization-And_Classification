# Fish Localization And Classification

**Project overview**

Nearly half of the world depends on seafood for their main source of protein. In the Western and Central Pacific, where 60% of the world’s tuna is caught, illegal, unreported, and unregulated fishing practices are threatening marine ecosystems, global seafood supplies and local livelihoods. The Nature Conservancy is working with local, regional and global partners to preserve this fishery for the future.

Currently, the Conservancy is looking to the future by using cameras to dramatically scale the monitoring of fishing activities to fill critical science and compliance monitoring data gaps. Although these electronic monitoring systems work well and are ready for wider deployment, the amount of raw data produced is cumbersome and expensive to process manually.

We have to develop algorithms to automatically detect and classify species of tunas, sharks and more that fishing boats catch, which will accelerate the video review process. Faster review and more reliable data will enable countries to reallocate human capital to management and enforcement activities which will have a positive impact on conservation and our planet.

Machine learning has the ability to transform what we know about our oceans and how we manage them.  


**Problem statement**

The main task is to localize and detect which species of fish appears on a fishing boat, based on images captured from boat cameras of various angles. 

The goal is to predict the likelihood that a fish is from a certain class from the provided classes, thus making it a multiclass classification problem in machine learning terms.

Eight target classes are provided in this dataset : Albacore tuna, Bigeye tuna, Yellowfin tuna, Mahi Mahi, Opah, Sharks, Other (meaning that there are fish present but not in the above categories), and No Fish (meaning that no fish is in the picture).

The goal is to train a CNN that would be able to classify fishes into these eight classes.




**Performance metrics**

The metric used for this Kaggle competition is multiclass logarithmic loss (also known as
categorical cross entropy).   

In multi-class classification (M>2), we take the sum of log loss values for each class prediction in the observation.


Here each image has been labeled with one true class and for each image a set of predicted
probabilities should be submitted. N is the number of images in the test set, M is the number of
image class labels,log is the natural logarithm, yij is 1 if observation belongs to class and 0
otherwise, and pij is the predicted probability that observation belongs to class .


Further we have also measured accuracy as a metric as we had to output accuracy of the model as required.

**Data description**

We are using the data set of “The Nature Conservancy Fisheries Monitoring” for this project.
Data Set link: https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data
Eight target categories are available in this dataset: 
1. Albacore tuna
2. Bigeye tuna
3. Yellowfin tuna
4. Mahi Mahi, 
5. Opah
6. Sharks
7. Other (meaning that there are fish present but not in the above categories) 
8. No Fish (meaning that no fish is in the picture). 

Each image has only one fish category, except that there are sometimes very small fish in the pictures that are used as bait. The train data is labeled according to fish species labels. We have to predict labels of test data.

There are 3777 images in the training set.
Among those,
1. 1719 images of ALB
2. 200 images of BET
3. 117 images of DOL
4. 67 images of LAG
5. 465 images of NoF
6. 299 images of Others
7. 176 images of Sharks
8. 734 images of YFT


**Data preprocessing**


1) Fish Localization: 

 We resize training files to 145x145x3. Training data was splited to train : validation : test data in the proportion of 80:10:10.  Then we apply various augmentations on the images such as rotation, width shift, height shift, shear, zoom, horizontal flip, rescale.  Moreover, we shuffle images in the respective directories.


2) Fish Classification:

All the images are resized to 145x145x3 as inceptionv3 model takes this specific sized files generally.
The localization part gives output a mask that is then ‘AND’ed  with the image to find the image of fish only in the real image. This masked image is then send to InceptionV3 to classify the fish found in the image.

**Model description**

1) Fish Localization:

We used a vgg16 model to localize fish in the image. This model was created by  VGG (Visual Geometry Group, University of Oxford) for the ILSVRC-2014.

2) Fish Classification:

We used inceptionV3 model to categorize the fish localized in the image.


**Result description**

Intersection Over Union(IoU)  (for model 1)
0.11

Log loss: (for model 2)
2.5134533

The result is not as satisfactory as expected because

Our devices do not have GPU, so we couldn’t run enough epochs to reach minima.
Images are large in size, take a lot of time to run even small number of epochs.
We had to train with a very small number of training data as it takes huge amount of time to train large dataset.
For SGD optimizer and learning rate 0.1, model did not converge.

**Conclusion**

We hope and believe that we will be able to get better accuracy if we use tensorflow object detection api such as yolo, ssd etc to detect fish images. We will further carry out this project to get satisfactory result. If we could run the project with bigger dataset and more epochs the result would be much better.


