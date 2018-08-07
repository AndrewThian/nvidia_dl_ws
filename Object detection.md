
<a href="https://www.nvidia.com/en-us/deep-learning-ai/education/"> <img src="DLI Header.png" alt="Header" style="width: 400px;"/> </a>

# Object Detection with DIGITS

In [Image Classification with DIGITS](https://nvidia.qwiklab.com/focuses/preview/1579), you learned to successfully *train* a neural network. You saw that while traditional programming was inadequate for classifying images, deep learning makes it not only possible, but fairly straightforward. You can now create an image classifier *when given a deep neural network and thousands of labeled images.* 

You then explored how you could improve your model. You increased its confidence and accuracy by running more epochs, adding more data, and being thoughtful about which network was best for your data. You expanded the types of digits your model could classify by augmenting images prior to training to represent a more diverse dataset. 

By the end of the lab, you had taken an off-the-shelf neural network and an open source dataset (handwritten digits) and created a model that could correctly classify new unlabeled digits that were both like and unlike your initial dataset. Pretty powerful!

In **this** lab, you'll learn to solve problems beyond image classification with deep learning by:

- Combining deep learning with traditional computer vision
- Modifying the internals (the actual math) of neural networks
- Choosing the right network for the job

to solve a second deep learning challenge:

**Can we *detect* and *localize* whale faces from aerial images?**

![Right whale example](right_whale_example.png)
<h4 align="center">Mother Whale and calf</h4> 

### Beyond Image Classification

With image classification, our network took an *image* and generated a *classification.* With object detection,  we want to take an *image* and generate a *new image.* 

More specifically, our input was a 28X28 matrix of pixel values and our output was a 10-dimensional vector (since we had 10 classes) of probabilities.

![](classify.png)

Image from Stanford's [CS231 Course](http://cs231n.github.io/classification/)

For *object detection* we want to take any size matrix (image) as an input and generate a new image of the same size where the whale faces (if any) are identified. *Detection* reports whether an object is present in an image. *Localization* reports where in an image an object is present. Some useful outputs could be heatmaps or bounding boxes. 

Heatmaps:

|   What we have     |   What we want |
|:------------------:|:--------------:|
|![](aerialwhale.png)|![](heatmap.png)|

Bounding Boxes:  

![](boundingbox.png)

## Approach 1: Building *around* deep learning 

In the first part of this lab, we will write traditional code (python) around an image classification model to create an object detection tool. If you have python (or other CS) experience, feel free to dig in and infer what the code is doing. If you aren't yet comfortable with the code, don't let it bog you down; we'll highlight and explain the most important bits.

The most important takeaway of this first part (that will allow you to solve problems way beyond this specific example) is:

#### **A *learned* function is still just a function. Before the input and after the output, you can build anything you want. **

Before we walk you through **one** way to solve this problem, take a minute to think through a possible solution. Assuming you could train a network to identify images like we did with handwritten digits, how might you use that to [detect](#det "find") whale faces in large aerial images.
Record your thoughts here:
Your solution may or may not match the way this first section of this lab tackles the problem. That's ok.

### 1.1 Sliding Window Approach

In the sliding window approach, we feed each 256X256 [grid square](#gs "portion") of the larger image through an image classifier. We're going to classify each *part* of an aerial image as either: *whale face* or *not whale face.*

**We need data:** Thanks to the [NOAA Right Whale Recognition Kaggle competition](https://www.kaggle.com/c/noaa-right-whale-recognition), we've got 9000 color 256X256 images in a folder on the GPU where this instance is running. 4500 of them are whale faces and 4500 of them are random (most likely "not whale faces"). This is a good place to start.

**We need a neural network:** Since the images are 256X256 and we've worked with it before, let's use AlexNet.

Test yourself: After opening DIGITS, your job is to:  

1) Load a new classification dataset - Training images are in the folder <code>/home/ubuntu/data/whale/data/train</code>  
2) Train a new classification model - Use <code>AlexNet</code> for <code>5 epochs</code>  

**Ignore the datasets and models that are already loaded and trained (you'll use them later).**

Note, you can find step-by-step guidance below.

# [Open DIGITS](/digits/) and start here:

![](datasetload.PNG)

You should get a binary classifier that can *determine* whether a 256X256 patch of an image is of a whale face or not a whale face with more than 90% accuracy. If you need a review on training image classification networks using DIGITS, click [here](#review).
<a id='digitsreturn'></a>

We now have a classifier whose input is a 256X256X3 color image and whose output is a set of probabilities of whether the image is or is not a whale face. 

We're going to write some python code that moves across each larger image, cutting it into data shape that our network expects, 256X256X3. After our output, we'll want to track the results of each classification. We can then use the results to communicate, through a heatmap, where in the image our classifier identified a whale face and where it did not.

### 1.2 Getting your model from DIGITS
In order to do anything with your trained model, you first have to get it from DIGITS. DIGITS creates "Job Directories" for both the model and the dataset. They contain [everything you'd need](#need "A description of the untrained network architecture, the weights the network learned during training,  and the mean image from the dataset used to 'normalize' the data) to deploy your model to the real world. We'll go deeper into the contents of these directories and this *deployment* workflow in "Neural Network Deployment with DIGITS and TensorRT," the next lab in this series.

DIGITS' "Job Page" for the model is what you see as soon as you create the model, when it is training, and/or if you select the model under DIGITS' "model" tab. The Job Directory is in the top left.

![](ModelJobView.PNG)

**Copy the job directory (highlighted above) and replace ##FIXME## in the code block below. Once you've copied the directory, execute the cell (Shift+Enter) to store it to the variable <code>MODEL_JOB_DIR</code>**


```python
MODEL_JOB_DIR = '##FIXME##'  ## Remember to set this to be the job directory for your model
print('Got it.')
```

The job directory for the **dataset** you just trained from is found by selecting the dataset from the model page (whale_faces in the above image) and/or if you select the dataset under DIGITS' "dataset tab. It's in the same place it was for the model, but should be a different number.

![](datasetjobdir.PNG)

Replace ##FIXME## with it and execute the code below:


```python
DATASET_JOB_DIR = '##FIXME##'  ## Remember to set this to be the job directory for your dataset
if MODEL_JOB_DIR == DATASET_JOB_DIR:
    print ('Make sure you have distinct directories for your model and your dataset.')
else:
    print('Got it.')
```

Next, we can import libraries, load the files we need, and select a random aerial image from the dataset. Each file is described in the comments, but there is more detail about what's happening in this step in the next lab in this series, "Neural Network Deployment with DIGITS and TensorRT." 

Execute the cell (Shift+Enter) but don't worry too much about it. It might take a minute or two and is done when 'In[X]' becomes 'In[number].'


```python
%matplotlib inline

import time
import numpy as np #Data is often stored as "Numpy Arrays"
import matplotlib.pyplot as plt #matplotlib.pyplot allows us to visualize results
import caffe #caffe is our deep learning framework, we'll learn a lot more about this later in the lab

MODEL_FILE = MODEL_JOB_DIR + '/deploy.prototxt'                 # This file contains the description of the network architecture
PRETRAINED = MODEL_JOB_DIR + '/snapshot_iter_270.caffemodel'    # This file contains the *weights* that were "learned" during training
MEAN_IMAGE = DATASET_JOB_DIR + '/mean.jpg'                      # This file contains the mean image of the entire dataset. Used to preprocess the data.

# Tell Caffe to use the GPU so it can take advantage of parallel processing. 
# If you have a few hours, you're welcome to change gpu to cpu and see how much time it takes to deploy models in series. 
caffe.set_mode_gpu()
# Initialize the Caffe model using the model trained in DIGITS
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# load the mean image from the file
mean_image = caffe.io.load_image(MEAN_IMAGE)

# Choose a random image to test against
RANDOM_IMAGE = str(np.random.randint(10))
IMAGE_FILE = 'data/samples/w_' + RANDOM_IMAGE + '.jpg'
input_image= caffe.io.load_image(IMAGE_FILE)
print("Ready to predict.")
```

### 1.3 Using our *learned* function: Forward Propagation

This is what we care about. Let's take a look at the function:   
<code>prediction = net.predict([grid_square])</code>.

Like any [function](https://www.khanacademy.org/computing/computer-programming/programming#functions), <code>net.predict</code> passes an input, <code>grid_square</code>, and returns an output, <code>prediction</code>. Unlike other functions, this function isn't following a list of steps, instead, it's performing layer after layer of matrix math to transform an image into a vector of probabilities.  

Run the cell below to see the prediction from the top left 256X256 <code>grid_square</code> of our random aerial image.


```python
X = 0
Y = 0

grid_square = input_image[X*256:(X+1)*256,Y*256:(Y+1)*256]
# subtract the mean image (because we subtracted it while we trained the network - more on this in next lab)
grid_square -= mean_image
# make prediction
prediction = net.predict([grid_square])
print prediction
```

Use what you know about the inputs and outputs of image classifiers to interpret this output.
Record your thoughts here:
The last layer of most classification networks is a function called "softmax." Softmax creates a vector where the higher the value of each cell, the higher the likelihood of the image belonging to that class. For example, above, if the first cell of the vector is higher than the second (and the network has been trained), the image likely contains a *whale face*. 

Optional: [Explore the math of softmax.](https://en.wikipedia.org/wiki/Softmax_function)

Next, let's implement some code around our function to iterate over the image and classify each grid_square to create a heatmap. Note that the key lesson of this section is that we could have built ANYTHING around this function, limited only by your creativity.


```python
# Load the input image into a numpy array and display it
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.show()

# Calculate how many 256x256 grid squares are in the image
rows = input_image.shape[0]/256
cols = input_image.shape[1]/256

# Initialize an empty array for the detections
detections = np.zeros((rows,cols))

# Iterate over each grid square using the model to make a class prediction
start = time.time()
for i in range(0,rows):
    for j in range(0,cols):
        grid_square = input_image[i*256:(i+1)*256,j*256:(j+1)*256]
        # subtract the mean image
        grid_square -= mean_image
        # make prediction
        prediction = net.predict([grid_square]) 
        detections[i,j] = prediction[0].argmax()
end = time.time()
        
# Display the predicted class for each grid square
plt.imshow(detections)

# Display total time to perform inference
print 'Total inference time: ' + str(end-start) + ' seconds'
```

It will take about 30 seconds to run.  Again, this is the output of a randomly chosen wide area test image along with an array showing the predicted class for each non-overlapping 256x256 grid square when input in to the model.

The code block below just groups everything we did into one block. You can run it as many times as you want to run multiple aerial images through your new object detection tool.


```python
# Choose a random image to test against
RANDOM_IMAGE = str(np.random.randint(10))
IMAGE_FILE = 'data/samples/w_' + RANDOM_IMAGE + '.jpg' 

# Load the input image into a numpy array and display it
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.show()

# Calculate how many 256x256 grid squares are in the image
rows = input_image.shape[0]/256
cols = input_image.shape[1]/256

# Initialize an empty array for the detections
detections = np.zeros((rows,cols))

# Iterate over each grid square using the model to make a class prediction
start = time.time()
for i in range(0,rows):
    for j in range(0,cols):
        grid_square = input_image[i*256:(i+1)*256,j*256:(j+1)*256]
        # subtract the mean image
        grid_square -= mean_image
        # make prediction
        prediction = net.predict([grid_square]) 
        detections[i,j] = prediction[0].argmax()
end = time.time()
        
# Display the predicted class for each grid square
plt.imshow(detections)

# Display total time to perform inference
print 'Total inference time: ' + str(end-start) + ' seconds'
```

As you run the above code multiple times you will see that in some cases this baseline model and sliding window approach is able to locate the whale's face. However, more often it will find a larger amount of the whale or get easily confused by breaking waves or sunlight reflecting from the ocean surface.

Before we address what we could do better, take a minute to reflect on what we just did by responding to the following question:

**In your own words, how did we just use an image classifier to detect objects?**
Record your thoughts here:
The intent of this section was to show that any deep learning solution is still just a mapping from input to output. This realization brings us one step closer to being able to understand how more complex solutions get built. For example, voice assistants like Alexa, Google Assistant, and Siri have to convert raw sound data into text, text into understanding, and understanding into a desired task like playing your favorite song.

<a id='question1'></a>

The complexity can be as far reaching as anything that has or can been built with computer science. As a coding/problem solving challenge, what are some factors we didn't think about when building this solution and how could we account for them?
Record your thoughts here:
Potential Answer: [Click here](#answer1)

### 1.4 Challenging optional exercises:

<a id='question-optional-exercise'></a>

1. Keeping the grid square size as 256x256, modify the code to increase the overlap between grid squares and obtain a finer classification map.

2. Modify the code to batch together multiple grid squares to pass in to the network for prediction.

Answer: [Click here](#answer-optional-exercise)

As we have now seen, the advantage of this sliding window approach is that we can train a detector using only patch-based training data (which is more widely available) and that with our current skillset, we can start working to solve the problem.

We could keep tweaking our code and see if we can continue to solve our problems (inacurate, slow) with brute force, OR, we could learn a bit more about deep learning. Let's go with the latter.

## Approach 2 - Rebuilding from an existing neural network

Remember, the the network is analogous to the brain. In this section, we'll show that there are different types of artificial "brains" that are designed to *learn* different types of things.  [Deeplearning4J](https://deeplearning4j.org/neuralnetworktable) has a great table that maps network types to use cases.

The intent of this next section isn't to dive into the details of *why* each type of network is ideal for each type of problem. At the end of the day, networks are *made of math.* Each layer is simply an operation. The more math you know, the more layer types will make sense. In subsequent labs, we'll build an understanding of common and powerful layer types.

Instead, this section will show HOW to change a network and prove that there is a lot of room for improvement once we start experimenting with network internals. Let's start with an exploration of our current network, AlexNet.

### 2.1 Our current network

The structure of these networks are described in some type of *framework,* in this case, Caffe. There are [many different frameworks](https://developer.nvidia.com/deep-learning-frameworks), each of which allow a practitioner to describe a network at a high level vs. having to physically program a GPU to do tensor math.

Let's take a look at AlexNet's Caffe description in DIGITS.

Go back to DIGITS' home screen by selecting "DIGITS" on the top left corner of the screen and select the model you just created.

Select "Clone Job" (on the top right in blue) to copy all of the settings to a new network. Cloning networks is a great way to iterate as you experiment.

At the bottom of the model settings, where you selected AlexNet, select "Customize."

![](customize.PNG)

You'll see some *prototext.* That is the Caffe description of the AlexNet network. Feel free to explore a bit to see how it's structured. In other frameworks, you would see something similar, it would just have a different structure.

Next, at the top of the network, select "Visualize" to, you guessed it, visualize the AlexNet network.

![](visualize.PNG)

While what's in here might not mean much to you...yet, you can see that what's written in the prototext directly informs what's displayed in the visualization. Prove it to yourself. Change the *name* of the first layer, *train-data* to *your name* and select *visualize* again. 

There you go...full control!

Now put it back to *train-data* and we'll change something that matters, the math.

We're going to replace some layers of the AlexNet network to create new functionality AND make sure that the network still trains. 

### 2.2 Brain Surgery

Let's start by replacing one layer and making sure that everything is still hooked up correctly. 

In the prototext, find the layer with the name "fc6." You can see that there's a lot of information about *what* this layer does. We'll get there. For now, let's just replace it. Replace all of the prototext between the word "layer" and the closing [bracket](#bracket "}") before the next "layer" with the following:

```
layer {
  name: "conv6"
  type: "Convolution"
  bottom: "pool5"
  top: "conv6"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 4096
    pad: 0
    kernel_size: 6
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
```

Let's *visualize* the network to compare our new network to the original AlexNet. For reference, this was the beginning of the original:

![](AlexNet beginning.PNG)

and this was the end:

![](AlexNetEnd.PNG)

Why might this indicate that we've broken something?
Record your thoughts here:
What do you think we could do to fix what we've broken?
Record your thoughts here:
### 2.3 Finishing the job

#### Rule 1: Data must be able to flow

We removed the fc6 layer, but it's clear that it's still present when we visualize our network. This is because *there are other layers that reference fc6:* 

- fc7
- drop6
- relu6

Take a look at the Caffe prototext to see where those layers are *told* to connect to fc6 and connect them instead to your new layer, <code>conv6</code>.

**Visualize again.** Your network should again show that data flows from and to the same inputs and outputs as the original AlexNet.

As the data flows from input to output, there are many "hidden layers," that perform some operation (math). The output of each layer needs to be defined as the input of the next. These create massively compound functions, which allow deep learning to model increasingly complex real world relationships between inputs and outputs. 

On one hand, we can use ANY math we want, as long as we keep everything connected so the data can flow from input to output. On the other:

#### Rule 2: The math matters

Let's take a second to compare the layer we removed with the layer we added. 

We removed a "fully connected" layer, which is a traditional matrix multiplication. While there is a lot to know about matrix multiplication, the one characteristic that we'll address is that matrix multiplication only works with specific matrix sizes. The implications of this for our problem is that we are limited to fixed size images as our input (in our case 256X256).

We added a "convolutional" layer, which is a "filter" function that moves over an input matrix. There is also a lot to know about convolutions, but the one characteristic that we'll take advantage of is that they **do not** only work with specific matrix sizes. The implications of this for our problem are that if we replace all of our fully connected layers with convolutional layers, we can accept any size image. 

**By converting AlexNet to a "Fully Convolutional Network," we'll be able to feed our entire aerial images to our network without first splitting them into grid squares.**

Let's convert AlexNet into a Fully-Convolutional Network (somewhat confusingly referred to as an FCN). Replace `fc7` through `fc8` with equivalent convolutional layers using the following Caffe text. 

**Important: This replacement should END at the end of the 'fc8' definition.**

```
layer {
  name: "conv7"
  type: "Convolution"
  bottom: "conv6"
  top: "conv7"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 4096
    kernel_size: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "conv7"
  top: "conv7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "conv7"
  top: "conv7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "conv8"
  type: "Convolution"
  bottom: "conv7"
  top: "conv8"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 2
    kernel_size: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
```

Is there anything else that you must change in your network to make sure that everything is still hooked up properly? If you're not sure, *visualize* your network again to help assess. If you still need a hint, hover [here](#rewiring "Remember to change the `bottom` blob of the `loss`, `accuracy` and `softmax` layers from 'fc8' to `conv8`.").

Once you feel confident that everything is hooked up properly, let's train a new model!

![](createmodel.PNG)

Give your model a name and select *create,* and while it trains, watch the video below to learn more about the convolutions you just added to your network.

[![What are convolutions?](Convolutions.png)](https://youtu.be/BBYnIoGrf8Y?t=13"What are convolutions.")

If you get no errors, great job! If you get one, read it. It will likely let you know where you've missed something. Commonly, the mistake lies [here](#rewiring "Remember to change the `bottom` blob of the `loss`, `accuracy` and `softmax` layers from 'fc8' to `conv8`."). If you just want the solution, click [here](#gu).

<a id='fcn'></a>

**Once you have a trained model, replace the ##FIXME## below with your new "Model Job Directory" like you did in the first section.


```python
JOB_DIR = '##FIXME'  ## Remember to set this to be the job directory for your model
```

Feel free to examine the code to see the difference between using this model and using AlexNet out of the box, but the main takeaway is that we feed the *entire* image through our trained model instead of grid squares.


```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import caffe
import copy
from scipy.misc import imresize
import time

MODEL_FILE = JOB_DIR + '/deploy.prototxt'                 # Do not change
PRETRAINED = JOB_DIR + '/snapshot_iter_270.caffemodel'    # Do not change

# Choose a random image to test against
RANDOM_IMAGE = str(np.random.randint(10))
IMAGE_FILE = 'data/samples/w_' + RANDOM_IMAGE + '.jpg'                  

# Tell Caffe to use the GPU
caffe.set_mode_gpu()

# Load the input image into a numpy array and display it
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.show()

# Initialize the Caffe model using the model trained in DIGITS
# This time the model input size is reshaped based on the randomly selected input image
net = caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
net.blobs['data'].reshape(1, 3, input_image.shape[0], input_image.shape[1])
net.reshape()
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data', 255.0)

# This is just a colormap for displaying the results
my_cmap = copy.copy(plt.cm.get_cmap('jet')) # get a copy of the jet color map
my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values

# Feed the whole input image into the model for classification
start = time.time()
out = net.forward(data=np.asarray([transformer.preprocess('data', input_image)]))
end = time.time()

# Create an overlay visualization of the classification result
im = transformer.deprocess('data', net.blobs['data'].data[0])
classifications = out['softmax'][0]
classifications = imresize(classifications.argmax(axis=0),input_image.shape,interp='bilinear').astype('float')
classifications[classifications==0] = np.nan
plt.imshow(im)
plt.imshow(classifications,alpha=.5,cmap=my_cmap)
plt.show()

# Display total time to perform inference
print 'Total inference time: ' + str(end-start) + ' seconds'
```

As you run the above code multiple times you will see that in many cases the Fully Convolutional Network is able to locate the whale's face with greater precision than the sliding window approach. Also, the total inference time for the FCN is about 1.5 seconds, whereas for the sliding window approach it took 10 seconds.  We get a better quality result in a fraction of the time!  This is a great benefit if we wish to deploy our detector in a real-time application, e.g. on board an aircraft, as we obtain a single model that can perform detection and classification in one efficient pass.

However, often it will still find a larger amount of the whale and is sometimes confused by breaking waves or sunlight reflecting from the ocean surface. Again, the false alarms caused by background clutter and the whale's body could be mitigated using strategies like appropriate data augmentation and other data work outside of the neural network.

Can we solve more of this problem within the network itself? 

The short answer is: yes.

The long answer is: by using an end-to-end object detection network trained to detect and localize whale faces. We'll use Whale_DetectNet.

## Approach 3: DetectNet

We started this lab with AlexNet, an elegant solution to the very specific problem of *image classification*. We built some python around it and then performed minor brain surgery to accomplish a task that we wouldn't have been able to prior to our work in deep learning, *object detection.*

However, is there an elegant solution to *object detection?* Could we build a model that maps directly from the input you have: aerial photos, to the output you want: localization and detection information?

It would also be nice if it has these benefits:

* Simple one-shot detection, classification and bounding box
* Very low latency
* Very low false alarm rates due to strong, voluminous background training data
* Handles varying number of objects in an image

**First,** you need specialized training data where all objects of interest are labeled with accurate bounding boxes (much rarer and costly to produce).

The figure below shows an example of a labeled training sample for a vehicle detection scenario.

![kespry example](kespry_example.png)

**Next,** you need a network that is purpose built to solve your specific problem. Luckily, there are many places to find these:

1. DIGITS has many networks pre-loaded (currently focused on Image Classification, Object Detection, and Image Segmentation)
2. Each framework (in this case Caffe) host "Model Zoos." These are repositiories of both untrained and trained networks that are designed to do everything from language translation to emotion detection.

After this lab, you are encouraged to explore different workflows and models in DIGITS and online and see what you can work with and what will require more learning. You will likely come away knowing that you are more capable than you thought and that there is a lot more to learn. 

We're going to use the DetectNet model in DIGITS to detect and localize whale faces. A complete training run of DetectNet on this dataset takes several hours, so we have provided a trained model to experiment with.  

We have also already prepared the full-size aerial images of the ocean in the appropriate folders and format for training DetectNet with the specific goal of training an end-to-end object detection model.

Before we start, let's clear the memory of the GPU by executing the following cell:


```python
try:
    del mean
    del transformer
    del classifier
except:
    print "Memory already freed"
```

### 3.1 Training an Object Detection Network in DIGITS

**We first need to import this data into DIGITS.**  Use the Datasets->Images dropdown and select "object detection" dataset. When the "New Object Detection Dataset" panel opens, use the preprocessing options from the image below:

Training image folder - /home/ubuntu/data/whale/data_336x224/train/images  
Training label folder - /home/ubuntu/data/whale/data_336x224/train/labels  
Validation image folder - /home/ubuntu/data/whale/data_336x224/val/images  
Validation label folder - /home/ubuntu/data/whale/data_336x224/val/labels  
Delete the default 1248 and 384 from *Pad Image*
Name - whales_detectnet

![OD data ingest](OD_ingest.png)

**Return to the main DIGITS screen and use the Models tab.**  Open the "whale_detectnet" model and use it as a pretrained model by selecting "Make Pretrained Model" towards the bottom of the screen.

![](pretrained.png)

Next, return to the main DIGITS screen and create a new model, this time, selecting "Object Detection" as the type. 

Make the following changes:

* select your newly created "whales_detectnet" dataset
* change the number of training epochs to 1 (We've already done most of the training.)  
* change the batch size to 5
* change the base learning rate to .0001
* Select pretrained networks and then "whale_detectnet"

![](pretrainedselect.png)

Find out why we didn't walk through how to build this network from scratch by exploring the network architecture. Select "Customize" and then "Visualize." Don't mess with it, but take a second to appreciate it.   

DetectNet is actually an Fully Convolutional Network (FCN), as we built above, but configured to produce precisely this data representation as its output.  The bulk of the layers in DetectNet are identical to the well-known GoogLeNet network.

When you're ready to train, give the model a new name like "whale_detectnet_2" and click "create".  Training this model for just 1 epoch will still take about 3 minutes, and you won't see much change in it's performance, but take a look at how performance is measured.

You will also see a different measure of performance than was reported for Image Classification models. The mean Average Precision (mAP) is a combined measure of how well the network can detect the whale faces and how accurate its bounding box (localization) estimates were for the validation dataset.

While the model is training return to the pre-trained "whale_detectnet" model.  You will see that after 100 training epochs this model had not only converged to low training and validation loss values, but also a high mAP score.  Let's test this trained model against a validation image to see if it can find the whale face.

**Set the visualization method to "Bounding boxes"** and paste the following image path in:  `/home/ubuntu/data/whale/data_336x224/val/images/000000118.png`.  Click "Test One".  You should see DetectNet successfully detects the whale face and draws a bounding box, like this:

![detectnet success](detectnet_success.png)

To access a list of other images to try, run the following cell and note that the '!' in Jupyter notebooks makes these cells act like the command line.


```python
!ls /home/ubuntu/data/whale/data_336x224/val/images
```

Feel free to test other images from the `/home/ubuntu/data/whale/data_336x224/val/images/` folder.  You will see that DetectNet is able to accurately detect most whale faces with a tightly drawn bounding box and has a very low false alarm rate.

There are a few key takeaways here:

1. The right network and data for the job at hand are vastly superior to hacking your own solution.
2. The right network (and sometimes even pretrained model) might be available in DIGITS or on the Model Zoo of your framework.

So one problem solving framework you can use is:

- determine if someone else has already solved your problem, use their model
- if they have not, determine if someone else has already solved a problem *like* yours, use their network and your data
- if they have not, determine if you can identify the shortcomings of someone else's solution in solving your problem and work to design a new network
- if you can not, use an existing solution other problem solving techniques (eg. python) to solve your problem
- either way, continue to experiment and take labs to increase your skill set!

Before moving to what's next, it is worth celebrating what you've accomplished.

You now know:

- How to train multiple types of neural networks using DIGITS
- What types of changes you can make to *how* you train networks to improve performance
- How to combine deep learning with traditional programming to solve new problems
- How to modify the internal layers of a network to change functionality and limitations
- to see who else has worked on problems like yours and what parts of their solution you can use

Next, we will dive into how to *deploy* deep neural networks to production to solve the problems you've set out to solve. We'll go deeper into what information you'll need, performance metrics in deployment, and the tools that maintain (or increase) the power and speed of deep learning in the hands of the end user!

### Thanks and data attribution

We would like to thank NOAA for their permission to re-use the Right Whale imagery for the purposes of this lab.

Images were collected under MMPA Research permit numbers:

MMPA 775-1600,      2005-2007

MMPA 775-1875,      2008-2013 

MMPA 17355,         2013-2018

Photo Credits: NOAA/NEFSC

<a href="https://www.nvidia.com/en-us/deep-learning-ai/education/"> <img src="DLI Header.png" alt="Header" style="width: 400px;"/> </a>

# Appendix


<a id='review'></a>
## Training Image Classifier Step-by-step

1. If DIGITS is not yet open, open it [here](/digits/).

#### Create dataset
2. From the main screen, select "Dataset" on the top left.
3. Select "New Dataset" and "Classification."
4. In the top field, "Training Images", copy and paste the filename: <code>/home/ubuntu/data/whale/data/train</code>.
5. In the bottom field, "Name", name your dataset "whale_faces".
6. Leaving all other defaults, select "Create."

#### Create model
7. Go back to the home screen by selecting DIGITS in the top left corner.
8. Select "New Model" and "Classification."
9. On the left hand side, select the dataset you just created, "whale_faces."
10. Change the number of "epochs" to 5.
11. Leaving all other defaults, select the AlexNet network.
12. In the bottom field, "Name", name your model "baseline."
13. Select "Create."

For more information (including images, etc.) retake [Image Classification with DIGITS](https://nvidia.qwiklab.com/focuses/preview/1579)


#### When you have your classifier, go [back to the lab](#digitsreturn)

## Answers to questions:

<a id='answer1'></a>
### Answer 1

The fact that we used a sliding window with non-overlapping grid squares means that it is very likely that some of our grid squares will only partially contain a whale face and this can lead to misclassifications. However, as we increase the overlap in the grid squares we will rapidly increase the computation time for this sliding window approach. We can make up for this increase in computation by *batching* inputs, a strategy that takes advantage of the GPU's intrinsic strength of parallel processing.

[Click here](#question1) to return to the lab.

<a id='answer-optional-exercise'></a>

### Answer to optional exercise



```python
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import caffe
import time

MODEL_JOB_NUM = '20160920-092148-8c17'  ## Remember to set this to be the job number for your model
DATASET_JOB_NUM = '20160920-090913-a43d'  ## Remember to set this to be the job number for your dataset

MODEL_FILE = '/home/ubuntu/digits/digits/jobs/' + MODEL_JOB_NUM + '/deploy.prototxt'                 # Do not change
PRETRAINED = '/home/ubuntu/digits/digits/jobs/' + MODEL_JOB_NUM + '/snapshot_iter_270.caffemodel'    # Do not change
MEAN_IMAGE = '/home/ubuntu/digits/digits/jobs/' + DATASET_JOB_NUM + '/mean.jpg'                      # Do not change

# load the mean image
mean_image = caffe.io.load_image(MEAN_IMAGE)

# Choose a random image to test against
RANDOM_IMAGE = str(np.random.randint(10))
IMAGE_FILE = 'data/samples/w_' + RANDOM_IMAGE + '.jpg' 

# Tell Caffe to use the GPU
caffe.set_mode_gpu()
# Initialize the Caffe model using the model trained in DIGITS
net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

# Load the input image into a numpy array and display it
input_image = caffe.io.load_image(IMAGE_FILE)
plt.imshow(input_image)
plt.show()

# Calculate how many 256x256 grid squares are in the image
rows = input_image.shape[0]/256
cols = input_image.shape[1]/256

# Subtract the mean image
for i in range(0,rows):
    for j in range(0,cols):
        input_image[i*256:(i+1)*256,j*256:(j+1)*256] -= mean_image
        
# Initialize an empty array for the detections
detections = np.zeros((rows,cols))
        
# Iterate over each grid square using the model to make a class prediction
start = time.time()
for i in range(0,rows):
    for j in range(0,cols):
        grid_square = input_image[i*256:(i+1)*256,j*256:(j+1)*256]
        # make prediction
        prediction = net.predict([grid_square])
        detections[i,j] = prediction[0].argmax()
end = time.time()
        
# Display the predicted class for each grid square
plt.imshow(detections)
plt.show()

# Display total time to perform inference
print 'Total inference time (sliding window without overlap): ' + str(end-start) + ' seconds'

# define the amount of overlap between grid cells
OVERLAP = 0.25
grid_rows = int((rows-1)/(1-OVERLAP))+1
grid_cols = int((cols-1)/(1-OVERLAP))+1

print "Image has %d*%d blocks of 256 pixels" % (rows, cols)
print "With overlap=%f grid_size=%d*%d" % (OVERLAP, grid_rows, grid_cols)

# Initialize an empty array for the detections
detections = np.zeros((grid_rows,grid_cols))

# Iterate over each grid square using the model to make a class prediction
start = time.time()
for i in range(0,grid_rows):
    for j in range(0,grid_cols):
        start_col = j*256*(1-OVERLAP)
        start_row = i*256*(1-OVERLAP)
        grid_square = input_image[start_row:start_row+256, start_col:start_col+256]
        # make prediction
        prediction = net.predict([grid_square])
        detections[i,j] = prediction[0].argmax()
end = time.time()
        
# Display the predicted class for each grid square
plt.imshow(detections)
plt.show()

# Display total time to perform inference
print ('Total inference time (sliding window with %f%% overlap: ' % (OVERLAP*100)) + str(end-start) + ' seconds'

# now with batched inference (one column at a time)
# we are not using a caffe.Classifier here so we need to do the pre-processing
# manually. The model was trained on random crops (256*256->227*227) so we
# need to do the cropping below. Similarly, we need to convert images
# from Numpy's Height*Width*Channel (HWC) format to Channel*Height*Width (CHW) 
# Lastly, we need to swap channels from RGB to BGR
net = caffe.Net(MODEL_FILE, PRETRAINED, caffe.TEST)
start = time.time()
net.blobs['data'].reshape(*[grid_cols, 3, 227, 227])

# Initialize an empty array for the detections
detections = np.zeros((rows,cols))

for i in range(0,rows):
    for j in range(0,cols):
        grid_square = input_image[i*256:(i+1)*256,j*256:(j+1)*256]
        # add to batch
        grid_square = grid_square[14:241,14:241] # 227*227 center crop        
        image = np.copy(grid_square.transpose(2,0,1)) # transpose from HWC to CHW
        image = image * 255 # rescale
        image = image[(2,1,0), :, :] # swap channels
        net.blobs['data'].data[j] = image
    # make prediction
    output = net.forward()[net.outputs[-1]]
    for j in range(0,cols):
        detections[i,j] = output[j].argmax()
end = time.time()
        
# Display the predicted class for each grid square
plt.imshow(detections)
plt.show()

# Display total time to perform inference
print 'Total inference time (batched inference): ' + str(end-start) + ' seconds'
```

When you're done, go [back to the lab.](#question-optional-exercise)

<a id='gu'></a>
# Fully Convolutional Network Solution

You can replace **all** of the text in AlexNet with the following to successfully convert it to a fully convolutional network. After copying, return to the lab by clicking [here](#fcn).

```# AlexNet
name: "AlexNet"
layer {
  name: "train-data"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    mirror: true
    crop_size: 227
  }
  data_param {
    batch_size: 128
  }
  include { stage: "train" }
}
layer {
  name: "val-data"
  type: "Data"
  top: "data"
  top: "label"
  transform_param {
    crop_size: 227
  }
  data_param {
    batch_size: 32
  }
  include { stage: "val" }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "norm1"
  type: "LRN"
  bottom: "conv1"
  top: "norm1"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "norm1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "norm2"
  type: "LRN"
  bottom: "conv2"
  top: "norm2"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "norm2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv6"
  type: "Convolution"
  bottom: "pool5"
  top: "conv6"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 4096
    pad: 0
    kernel_size: 6
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "conv6"
  top: "conv6"
}
layer {
  name: "drop6"
  type: "Dropout"
  bottom: "conv6"
  top: "conv6"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "conv7"
  type: "Convolution"
  bottom: "conv6"
  top: "conv7"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 4096
    kernel_size: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "conv7"
  top: "conv7"
}
layer {
  name: "drop7"
  type: "Dropout"
  bottom: "conv7"
  top: "conv7"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "conv8"
  type: "Convolution"
  bottom: "conv7"
  top: "conv8"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  param {
    lr_mult: 2.0
    decay_mult: 0.0
  }
  convolution_param {
    num_output: 2
    kernel_size: 1
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "accuracy"
  type: "Accuracy"
  bottom: "conv8"
  bottom: "label"
  top: "accuracy"
  include { stage: "val" }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "conv8"
  bottom: "label"
  top: "loss"
  exclude { stage: "deploy" }
}
layer {
  name: "softmax"
  type: "Softmax"
  bottom: "conv8"
  top: "softmax"
  include { stage: "deploy" }
}
```

After copying, return to the lab by clicking [here](#fcn).
