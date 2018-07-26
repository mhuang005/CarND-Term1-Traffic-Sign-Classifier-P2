## **Traffic Sign Recognition** 


### Overview
---

A convolutional neural network model is designed and trained on the [German traffic sign dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) to classify traffic signs. The model is required to achieve accuracy of 0.93 or greater on the validation set.   The steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/train_images.png "Randomly chosen images from the training set:"
[image2]: ./images/distr_train.png "distribution of training classes "
[image3]: ./images/distr_valid.png "distribution of validation classes "
[image4]: ./images/distr_test.png "distribution of test classes "
[image5]: ./images/new_images.png "5 new traffic Signs"
[image6]: ./images/five_softmax_probs.png "Top 5 softmax probabilities"




You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration
---
#### 1. The basic sumary of the traffic signs data set

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set:  **34799**
* The size of the validation set:  **4410**
* The size of test set: **12630**
* The shape of a traffic sign image: **(32, 32, 3)**
* The number of unique classes/labels in the data set: **43**

#### 2. An exploratory visualization of the dataset.

Here are 5 randomly chosen images from the German traffic sign training set:

![alt text][image1]


The following bar charts show three distributions of classes in the training, validation and test sets, respectively.

![alt text][image2]
![alt text][image3]
![alt text][image4]


As shown above, the classes have similar distributions across the training, validation and test sets.


### Design and Test a Model Architecture
---
#### 1. Data Preprocessing

I only use the normalization

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   **(pixel - 128) / 128**
   
to map the data to [-1, 1].   Note that the large values in the data set are more likely to produce large weights/parameters of the model, which may result in divergence of the model.  prefers samll weights. This normalization helps generate very large weights during the training. 


#### 2. The convolutional Neural Network Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding,  outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x16 	|
| Dropout   	      	| keep_drop: 0.6    							|
| Convolution 3x3     	| 1x1 stride, same padding,  outputs 14x14x32 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding,  outputs 14x14x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 7x7x32 	|
| Dropout               | keep_drop: 0.6    							|
| Flatten	        	| inputs 7x7x32, outputs 1568     				|
| Fully connected		| inputs 1568,   outputs 400     				|
| Fully connected		| inputs 400,    outputs 400     				|
| Dropout               | keep_drop: 0.6    							|
| Softmax				| inputs 400,    outputs 43     				|
|						|												|

The Adam optimizer (with default parameters in Tensorflow) is used to train the model.  The following are the hyperparameters used in the training:

* Epochs: **40**
* Batch size: **64**
* Learning rate: **0.001**


#### 3. The approach for the final model

As suggested, I started with the Lenet which gave decent results on the traffic date set.
Notice that the Lenet is a quite simple CNN model (e.g., 5 layers, samll filter numbers for convolutinal layers, small number of neurons in fully connected layers). The basic idea is to inrease the model capacity by increasing the number of layers, the number of filters and the number of neurons in the fully conncected layers. Also, I used the dropout technique to avoid potential overfitting. I tested the **VGG-like architecture** (see above) with the dropout layers. As shown above, I did increase those numbers I mentioned. The **VGG-like architecture** is well-known and gives good performance for image recognition. The task we need to do here is similar to the tasks the **VGG-like architecture** did (e.g., classification on ImageNet).   This is the main reason I chose this model, which actually gave very good results (e.g., > 0.94 accuracy on the validation set).   

The following results are achieved by the model:

* The accuracy on the training set: **1.000** 
* The accuracy on the validation set: **0.979**
* The accuracy on the test set: **0.970**
 

### Test a Model on New Images
---
#### 1. Five new German traffic sign images

Here are five German traffic signs that I found on the web:

![alt text][image5] 


The first four images are chosen from [Wikipedia](https://en.wikipedia.org/wiki/Road_signs_in_Germany). The qualities of the images look good, but the images seem different from those in the training set. The images in the training set (as well as the validation or test set) are blur with noisy background. In contrast, the first four images here look more clear (but) with dark background. I choose the fifth image to add a little bit of diversity in the set. In addition, the five classes of the traffic signs chosen here are not among the classes with large number of samples, which make the classification a little chanllenging.    

#### 2. Model performance on the new test set. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy road      		| Bumpy road   									| 
| Double curve    		| Double curve 									|
| Children crossing		| Children crossing								|
| Slippery Road			| Slippery Road      							|
| Stop          		| Stop       									|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97%.

#### 3. Top five softmax probabilities

The code for making predictions on my final model is located in the 19th cell of the Ipython notebook. Here are top 5 softmax probabilites for each of five traffic signs.


![alt text][image6] 





