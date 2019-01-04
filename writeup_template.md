# **Traffic Sign Recognition** 

## Building a Network to classify traffic Networks

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


### Step 0: Load the Data

In a first step I load in the provided Project data. Store training calidation and test sets.

To see the dimensions of the data I printed out the shape of the test and training data set. Also I had a look into the numbers of images provided.

Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630

### Step 1: Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is: 32, 32, 3
* The size of the validation set is: 32, 32, 3
* The size of test set is: 32, 32, 3
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Vizualization of a random trafic sign from the training data and mapping it to the name which was provided in the csv file: 

[image8]: ./examples/Sign.png "Traffic Sign"
[image8]: ./examples/Bildschirmfoto1.png "Traffic Sign"

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in the training set.

[alt text]: ./examples/Histogram.png "Histogram"

Additionally I counted the number of signs in the test data to get a feeling for the provided signs.

Many signs:
- Speed limit (50km/h)''  train samples: 2010
- Speed limit (30km/h)''  train samples: 1980
- Yield''  train samples: 1920
- Priority road''  train samples: 1890
- Keep right''  train samples: 1860

Fewer signs:
- Go straight or left''  train samples: 180
- Dangerous curve to the left''  train samples: 180
- Speed limit (20km/h)''  train samples: 180
- End of no passing''  train samples: 210
- End of no passing by vehicles over 3.5 metric tons''  train samples: 210


### Step 2: Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
 
Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and so it can be used in this project.

I also converted the images to grayscale.


In the def function the different steps (normalize and grayscale) are conducted. Image that shows the transforming steps:


[image8]: ./examples/Transformation.png "Transformation"
[image8]: ./examples/Bildschirmfoto2.png "Transformation"

I did this step with the tes, validation and training data.
The final shape of the data is as follows:

Training set Image data shape = (34799, 32, 32)
Validating set Image data shape = (4410, 32, 32)
Testing set Image data shape = (12630, 32, 32)

#### Defenition Architecture

Shuffle data. This is important because otherwise the ordering of the data might have a big impact in how well the network trends.

EPOCHS number defines how many times we want to run the training data through the network.
BATCH SIZE variable defines how many training images run through the network at a time.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
Hyper parameters to inizialize our weights.
I used the suggested Model which was presented in the training lesson.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1   image   							| 
| Convolution Layer1   	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution Layer2   	| 1x1 stride,  outputs 10x10x16             	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flatten       	    | output= 400 									|
| connected Layer3 		| output= 120  									|
| RELU					|												|
| Connected Layer4		| output= 84  									|
| RELU					|												|
| Connected  Layer5		| output= 43  									|

 The last three layers are fully connected layers. The fully connected convolutional layer map the x feature maps of each size 1x1.
 
 (source: https://engmrk.com/lenet-5-a-classic-cnn-architecture/)


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


#### Training Pipeline- trains model
I choose the following parameters:
EPOCHS=30
EPOCHS number defines how many times we want to run the training data through the network.
BATCH SIZE=64
BATCH SIZE variable defines how many training images run through the network at a time.

Tensor flow variables
x- as a placeholder that stores the input batches- (any size possible, 32,32)
y- stores the lables

Learning rate: 0.001
Learning rate sets how quckly to update the network´s weights.

Calculation of Logits by passing the input data to the LeNet function.

#### Model Evaluation -evaluates how godd the model is
Another Pipeline to evaluate the model.
Meassueres if a given prediction is correct: Compares the logit prediction to the one-hot encoded ground trouth lable.
Calculates overal accuracy of the model.

I set up the learning pilelines, as describes in Lesson 15 (5 to 6).

#### Training the Model

Function to train and validate the Model- also as described in the lesson.
First initialize variables and train over the number of set EPOCHS. 
Train Model on the broken down batches.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
I tested many different parameters to get a good result on my data.


My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.981 
* test set accuracy of 0.916

If a well known architecture was chosen:
* What architecture was chosen? LeNet as tought in the classroom
* Why did you believe it would be relevant to the traffic sign application? I had some issues with the arry sizes thats why i had to print out the results to adjust my input sizes. I adapted the layers and bias as it was suggested in the classroom.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? the accuracy is almost improving
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

[image4]: ./test_images/Sign1.png "Traffic Sign 1"
[image5]: ./test_images/Sign2.png "Traffic Sign 2"
[image6]: ./test_images/Sign3.png "Traffic Sign 3"
[image7]: ./test_images/Sign4.png "Traffic Sign 4"
[image8]: ./test_images/Sign5.png "Traffic Sign 5"
I chose easy pictures. Because I want to prove that my code works. It can still be improved in many ways(color filters, cut out of area of interest...) before starting the analysation.
I focused on the most important steps: normalization and grayscale as well as cropping

[image8]: ./examples/TBildschirmfoto4.png "Transformed pictures"

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn left ahead     	| Turn left ahead 								 | 
| No entry     			| No entry 										|
| Priority road			| Priority road									|
| Construction	   		| Priority road	  								|
| Speed limit (80km/h)	| Priority road									|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

Looking at the first colum I get [ 0.17483388  0.24703826  0.25325593  0.18915817  0.16582201]. These are the three largest probabilities. The corresponding indices are: [34, 17, 12, 12, 12]. the numbers stand for the corresponding traffic sign. I could get the name from the .csv file

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .17         			| Turn left ahead  								| 
| .24     				| No entry 										|
| .25					| Priority road									|
| .18	      			| Priority road					 				|
| .16				    | Priority road      							|



