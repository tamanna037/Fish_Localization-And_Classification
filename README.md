# Fish Localization And Classification

**Project overview**

Nearly half of the world depends on seafood for their main source of protein. In the Western and Central Pacific, where 60% of the world’s tuna is caught, illegal, unreported, and unregulated fishing practices are threatening marine ecosystems, global seafood supplies and local livelihoods. The Nature Conservancy is working with local, regional and global partners to preserve this fishery for the future.

Currently, the Conservancy is looking to the future by using cameras to dramatically scale the monitoring of fishing activities to fill critical science and compliance monitoring data gaps. Although these electronic monitoring systems work well and are ready for wider deployment, the amount of raw data produced is cumbersome and expensive to process manually.

We have to develop algorithms to automatically detect and classify species of tunas, sharks and more that fishing boats catch, which will accelerate the video review process. Faster review and more reliable data will enable countries to reallocate human capital to management and enforcement activities which will have a positive impact on conservation and our planet.

Machine learning has the ability to transform what we know about our oceans and how we manage them.  





Figure: Images from training set


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
