
<a href="https://www.nvidia.com/en-us/deep-learning-ai/education/"> <img src="files/DLI Header.png" alt="Header" style="width: 400px;"/> </a>

# Deep Learning Network Deployment

By Jon Barker and Ryan Olson

Before we begin, let's verify [WebSockets](http://en.wikipedia.org/wiki/WebSocket) are working on your system.  To do this, execute the cell block below by giving it focus (clicking on it with your mouse), and hitting Ctrl-Enter, or pressing the play button in the toolbar above.  If all goes well, you should see get some output returned below the grey cell.  If not, please consult the [Self-paced Lab Troubleshooting FAQ](https://developer.nvidia.com/self-paced-labs-faq#Troubleshooting) to debug the issue.


    print "The answer should be three: " + str(1+2)

    The answer should be three: 3


Let's execute the cell below to display information about the GPUs running on the server.


    !nvidia-smi

    Fri Jul 27 08:07:35 2018       
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 367.44                 Driver Version: 367.44                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GRID K520           On   | 0000:00:03.0     Off |                  N/A |
    | N/A   31C    P8    18W / 125W |     38MiB /  4036MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                                   
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |    0      3108    C   /usr/bin/python                                 36MiB |
    +-----------------------------------------------------------------------------+


---
The following video will explain the infrastructure we are using for this self-paced lab, as well as give some tips on it's usage.  If you've never taken a lab on this system before, it's highly recommended you watch this short video first.<br><br>
[![Intro](http://img.youtube.com/vi/ZMrDaLSFqpY/0.jpg)](http://www.youtube.com/watch?v=ZMrDaLSFqpY "Intro to NVIDIA's Self-Paced Lab Infrastructure")

## Introduction

Welcome to NVIDIA's deep learning network deployment lab.  In previous labs, you learned to *train* deep neural networks. You also performed basic *inference* for visualization and testing. In this lab, you'll gain the skills to deploy your network for production. 

You'll use DIGITS, Caffe and TensorRT to deploy networks trained in DIGITS. You will experiment with the effect of batch size on throughput and latency while using Caffe for object detection. You'll also assess the performance difference between Caffe and TensorRT by using both for efficient image classification.

Whether you're classifying images to help fight cancer or detecting objects to avoid pedestrians in self-driving cars, deployment is where you'll solve the problem you set out to solve. Your network **uses what it learned** during training **to make decisions** in the real world.

## GPU acceleration for training and deployment

GPU acceleration improves performance in both phases of machine learning solutions: training and deployment.

### Training

In [Getting Started with Deep Learning,](https://nvidia.qwiklab.com/focuses/preview/1579?locale=en) you learned how to create, train, and test a deep neural network. You saw how GPUs greatly accelerate training. Due to the size of datasets and the depth of neural networks, training is typically very resource-intensive. What can take weeks or months on traditional compute architectures takes days or hours using GPUs.

![](files/dnn.png)

### Deployment

*Inference* is the process of using a trained model to make predictions from new data. *Deployment* is *inference* in a production environment such as a data center, an automobile, or an embedded platform. Due to the depth of deep neural networks, inference also requires significant compute resources to process in real-time on imagery or other high-volume sensor data. 

GPU acceleration can also be utilized during inference. [NVIDIA Jetson's TX1 and TX2](http://www.nvidia.com/object/embedded-systems-dev-kits-modules.html) integrated NVIDIA GPU deployed onboard embedded platforms and [NVIDIA Drive PX2](http://www.nvidia.com/object/drive-px.html) enable real-time inference in robotics applications like picking, autonomous navigation, agriculture, and industrial inspection.

## Completing a deep learning workflow: Inference and Deployment

The typical training and inference cycle is depicted below.

<img src="files/digits_workflow.png" alt="Drawing" style="width: =800px;"/>

This lab will focus on the right-hand side of the diagram. We'll deploy two networks, *GoogLeNet* and *pedestrian_detectNet:*

- *GoogLeNet* is a well known convolutional neural network (CNN) architecture that has been trained for image classification using  the [ilsvrc12 Imagenet](http://www.image-net.org/challenges/LSVRC/2012/) dataset.  This network can assign one of 1000 class labels to an entire image based on the dominant object present. The classes this model can recognize can be found [here](http://image-net.org/challenges/LSVRC/2012/browse-synsets).

- *pedestrian_detectNet* is another CNN architecture that is not only able to assign a global classification to an image but can go further and detect multiple objects within the image and draw bounding boxes around them.  The pre-trained model provided has been trained for the task of pedestrian detection using a large dataset of pedestrians in a variety of indoor and outdoor scenes.

After reviewing inference in DIGITS, we'll deploy the network in two ways:
1. *Using the trained model as-is* using pycaffe, a python based deep learning framework API. 
2. *Optimizing the model for production* using Nvidia TensorRT, a high-performance deep learning inference engine.

You'll compare and contrast the process for different networks, tasks and deployment frameworks and identify the impact of batch size in optimizing for throughput vs. latency.

You'll always start with a model definition and trained weights. In this lab, we'll find those directly in DIGITS. 

### Part 1: DIGITS

DIGITS is an open-source project contributed by NVIDIA.

As you saw in [Getting Started with Deep Learning,](https://nvidia.qwiklab.com/focuses/preview/1579?locale=en) DIGITS allows anyone to easily get started and interactively *train* their networks with GPU acceleration. 

You also saw that any pre-trained model in DIGITS can immediately be applied to a new test image through the same browser interface in which it was trained. This is *inference.* *Inference* is a network's main job once deployed, so we'll use DIGITS as a first step.

---
Click [here](/digits/) to open DIGITS in a separate tab.  If at any time DIGITS asks you to login you can just make up a username and proceed.

The DIGITS server you will see running contains our two neural networks, *GoogLeNet* and *pedestrian_detectNet*, listed under the `"Models"` tab:

![](files/pretrained.png)

Click on the name of one of the models and scroll to the bottom to see the "Trained Models" view.  Here, you can *apply* the model to a single test image, a whole LMDB database of test images or a list of files referenced in a text file. Note that you have different options depending on the model you're using.

**Exercise:** Follow the instructions in the image below to test the following single images that reside on the same server as the running DIGITS instance:
1. For *GoogLeNet:* /home/ubuntu/deployment_lab/test_images/GN_test.png
2. For *pedestrian_detectNet:* /home/ubuntu/deployment_lab/test_images/DN_test.png

![](files/digits_inference.png)

If you successfully follow these steps you should see one of the inference outputs below:

![](files/digits_inference_outputs.png)

(**Optional Exercise - The power of additional computation**) 

Carry out the same inference task again using DetectNet but this time select a version of the model from an earlier epoch in the training process using `"Select Model`" dropdown.  You should find that models from earlier epochs produce less accurate detection results. Note that this is usually, but [not always](http://docs.aws.amazon.com/machine-learning/latest/dg/model-fit-underfitting-vs-overfitting.html) the case. 


We've now used our trained network to make inferences on new data. Next, we'll deploy our model to production to carry out inference programatically. We'll start with a simple Python application that uses pycaffe, the same framework that DIGITS just used to infer. Through this process, we'll identify the files you need from DIGITS for deployment. You'll also learn to optimize your deployment for either throughput or latency by manipulating batch size.

## Part 2: Inference using pycaffe

We will now walk through a simple example Python application that uses pycaffe to detect pedestrians in a test image using the pretrained DetectNet model introduced in Part 1.  Note that this example is representative of the way that inference is carried out in the majority of deep learning frameworks.

Run the following code to import libraries and configure Caffe to use the GPU for inference.


    # Import required Python libraries
    %pylab inline
    pylab.rcParams['figure.figsize'] = (15, 9)
    import caffe
    import numpy as np
    import time
    import os
    import cv2
    from IPython.display import clear_output
    
    # Configure Caffe to use the GPU for inference
    caffe.set_mode_gpu()

    Populating the interactive namespace from numpy and matplotlib


A model trained in DIGITS can be imported into pycaffe as an instance of the `caffe.Net` class.  You can find the path of the directory containing the model definition files and trained model weights listed under **Job Directory
** at the top of the model's page in DIGITS.  For example:

<img src="files/digits_job.png" alt="Drawing" style="width: 600px;"/>

Similarly if you look at the corresponding dataset in DIGITS you can find the **Job Directory** containing the dataset mean image. The mean image was used to normalize training data, so it must also be use for inference as well.

**Exercise:** Find the job directory for the pedestrian_detectnet model and the pedestrian_dummy_dataset and use them as the `MODEL_JOB_DIR` and `DATA_JOB_DIR` parameters in the code cells below to apply DetectNet to a sequence of test frames from the video `melbourne.mp4` using pycaffe.


    # Set the model job directory from DIGITS here
    MODEL_JOB_DIR='/home/ubuntu/digits/digits/jobs/20160905-143028-2f08'
    # Set the data job directory from DIGITS here
    DATA_JOB_DIR='/home/ubuntu/digits/digits/jobs/20160905-135347-01d5'

DIGITS makes the model weights after each epoch available as the "snapshot_iter_X.caffemodel" file. Below, we find the iteration number, X, of the last epoch and then instantiate both the model architecture ("deploy.prototxt") and trained model weights in GPU memory using "caffe.Net".


    # We need to find the iteration number of the final model snapshot saved by DIGITS
    for root, dirs, files in os.walk(MODEL_JOB_DIR):
        for f in files:
            if f.endswith('.solverstate'):
                last_iteration = f.split('_')[2].split('.')[0]
    print 'Last snapshot was after iteration: ' + last_iteration

    Last snapshot was after iteration: 70800



    # Instantiate a Caffe model in GPU memory
    # The model architecture is defined in the deploy.prototxt file
    # The pretrained model weights are contained in the snapshot_iter_<number>.caffemodel file as detirmined above.
    classifier = caffe.Net(os.path.join(MODEL_JOB_DIR,'deploy.prototxt'), 
                           os.path.join(MODEL_JOB_DIR,'snapshot_iter_' + last_iteration + '.caffemodel'),
                           caffe.TEST)


    # Load the dataset mean image file
    mean = np.load(os.path.join(DATA_JOB_DIR,'train_db','mean.npy'))
    
    # Instantiate a Caffe Transformer object that will preprocess test images before inference
    transformer = caffe.io.Transformer({'data': classifier.blobs['data'].data.shape}) 
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data',mean.mean(1).mean(1)/255)
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))
    
    BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = classifier.blobs['data'].data[...].shape
    
    print 'The input size for the network is: (' + \
            str(BATCH_SIZE), str(CHANNELS), str(HEIGHT), str(WIDTH) + \
             ') (batch size, channels, height, width)'

    The input size for the network is: (1 3 512 1024) (batch size, channels, height, width)


We can now deploy our model programmatically. *pedestrian_detectNet* is designed to place bounding boxes around pedestrians. Our code needs to do three things:

1. Make our actual input look like the input layer of our network.  

    Our input will be frames from a video stream. As shown above, our network will expect a height of 512 and a width of 1024. This is done using <code>frame = cv.resize(frame, (1024, 512), 0, 0)</code>.

2. Infer

    <code>bounding_boxes = classifier.forward()['bbox-list'][0]" - classifier.forward </code>is where the *actual* inference happens. 
    
3. Make our network's output match our intended output.

    In this case, our model outputs an array of bounding box predictions where (bounding_box[0],bounding_box[1]) are the coordinates of one corner of each box and (bounding_box[2],bounding_box[3]) are the coordinates of the opposite corner. Our deployment then simply overlays rectangles using those coordinates to display where the pedestrians are.  

Run the following code block to deploy your model.


    # Create opencv video object
    vid = cv2.VideoCapture('/home/ubuntu/deployment_lab/melbourne.mp4')
    
    # We will just use every n-th frame from the video
    every_nth = 5
    counter = 0
    
    try:
        while(True):
            # Capture video frame-by-frame
            ret, frame = vid.read()
            counter += 1
            
            if not ret:
                
                # Release the Video Device if ret is false
                vid.release()
                # Mesddage to be displayed after releasing the device
                print "Released Video Resource"
                break
            if counter%every_nth == 0:
                
                # Resize the captured frame to match the DetectNet model
                frame = cv2.resize(frame, (1024, 512), 0, 0)
                
                # Use the Caffe transformer to preprocess the frame
                data = transformer.preprocess('data', frame.astype('float16')/255)
                
                # Set the preprocessed frame to be the Caffe model's data layer
                classifier.blobs['data'].data[...] = data
                
                # Measure inference time for the feed-forward operation
                start = time.time()
                # The output of DetectNet is an array of bounding box predictions
                bounding_boxes = classifier.forward()['bbox-list'][0]
                end = (time.time() - start)*1000
                
                # Convert the image from OpenCV BGR format to matplotlib RGB format for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create a copy of the image for drawing bounding boxes
                overlay = frame.copy()
                
                # Loop over the bounding box predictions and draw a rectangle for each bounding box
                for bbox in bounding_boxes:
                    if  bbox.sum() > 0:
                        cv2.rectangle(overlay, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (255, 0, 0), -1)
                        
                # Overlay the bounding box image on the original image
                frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
                
                # Display the inference time per frame
                cv2.putText(frame,"Inference time: %dms per frame" % end,
                            (10,500), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    
                # Display the frame
                imshow(frame)
                show()
                # Display the frame until new frame is available
                clear_output(wait=True)
                
    # At any point you can stop the video playback and inference by  
    # clicking on the stop (black square) icon at the top of the notebook
    except KeyboardInterrupt:
        # Release the Video Device
        vid.release()
        # Message to be displayed after releasing the device
        print "Released Video Resource"            

    Released Video Resource


** Make sure to release the video resource by clicking on the stop (black square) icon at the top of the notebook.**

As you will see pycaffe can carry out DetectNet inference on a 1024x512 video frame in a time of about 73ms, or about 13.6 frames per second. Is this fast enough for your application?

The overhead of reading frames from the video and drawing the output frame with bounding box overlays slows down video playback. There is also a startup cost associated with initializing the model in memory.  In a production system you would often try to have the data ingest and output postprocessing taking place in a separate parallel thread to the Caffe inference.  You would also keep the model initialized in GPU memory to accept new data at any time without incurring the startup overhead again.

### Batch size effect on latency vs. throughput

In some applications, we are less worried about the inference time for an individual frame (latency) and more concerned with how many frames we can process in a unit of time (throughput).  In these situations, we can carry out inference as a batch process, just like during training.  In the case of video, this would mean buffering frames until a batch is full and then carrying out inference.

**Exercise:** Modify the code cell below to carry out inference on the same video as before but with batch sizes 1, 2, 5 and 8.  Compare the per frame inference time and throughput.  


    NEW_BATCH_SIZE = 2
    
    # Resize the input data layer for the new batch size
    OLD_BATCH_SIZE, CHANNELS, HEIGHT, WIDTH = classifier.blobs['data'].data[...].shape
    classifier.blobs['data'].reshape(NEW_BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    classifier.reshape()
    
    # Create opencv video object
    vid = cv2.VideoCapture('/home/ubuntu/deployment_lab/melbourne.mp4')
    
    counter = 0
    
    batch = np.zeros((NEW_BATCH_SIZE, CHANNELS, HEIGHT, WIDTH))
    
    try:
        while(True):
            # Capture video frame-by-frame
            ret, frame = vid.read()
            
            if not ret:
                # Release the Video Device if ret is false
                vid.release()
                # Message to be displayed after releasing the device
                print "Released Video Resource"
                break
                
            # Resize the captured frame to match the DetectNet model
            frame = cv2.resize(frame, (WIDTH, HEIGHT), 0, 0)
            
            # Add transformed frame to batch array
            batch[counter%NEW_BATCH_SIZE,:,:,:] = transformer.preprocess('data', frame.astype('float16')/255)
            counter += 1
            
            if counter%NEW_BATCH_SIZE==0:
                
                # Set the preprocessed frame to be the Caffe model's data layer
                classifier.blobs['data'].data[...] = batch
                
                # Measure inference time for the feed-forward operation
                start = time.time()
                # The output of DetectNet is now an array of bounding box predictions
                # for each image in the input batch
                bounding_boxes = classifier.forward()['bbox-list']
                end = (time.time() - start)*1000
                
                print 'Inference time: %dms per batch, %dms per frame, output size %s' % \
                        (end, end/NEW_BATCH_SIZE, bounding_boxes.shape)
                
    # At any point you can stop the video playback and inference by  
    # clicking on the stop (black square) icon at the top of the notebook
    except KeyboardInterrupt:
        # Release the Video Device
        vid.release()
        # Message to be displayed after releasing the device
        print "Released Video Resource"      

    Inference time: 921ms per batch, 460ms per frame, output size (2, 50, 5)
    Inference time: 5271ms per batch, 2635ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 139ms per batch, 69ms per frame, output size (2, 50, 5)
    Inference time: 139ms per batch, 69ms per frame, output size (2, 50, 5)
    Inference time: 139ms per batch, 69ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 139ms per batch, 69ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 139ms per batch, 69ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 139ms per batch, 69ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Inference time: 140ms per batch, 70ms per frame, output size (2, 50, 5)
    Released Video Resource


NOTE: If you try a batch size greater than 8 you will run out of GPU memory and see an error from the notebook.  This is fine to do, but you will have to re-execute the previous code cells to get back to this point.

You should have found that per frame inference time decreases as batch size increases (higher throughput) but the time to process a whole batch also increases (higher latency).

---
*IMPORTANT* Execute the cell below before starting part 3:


    # Make sure to clean up the GPU memory that Caffe is using before we start this next section.
    try:
        del mean
        del transformer
        del classifier
    except:
        print "Memory already freed"

## Part 3: NVIDIA TensorRT

NVIDIA [TensorRT](https://developer.nvidia.com/tensorrt) (also referenced as GIE) is a high-performance deep learning inference solution for production environments. Power efficiency and speed of response are two key metrics for deployed deep learning applications, because they directly affect the user experience and the cost of the service provided. TensorRT automatically optimizes trained neural networks for run-time performance and delivers GPU-accelerated inference for web/mobile, embedded and automotive applications.

![GIE](GIE_Graphics_FINAL-1.png)

There are two phases in the use of TensorRT: build and deployment.

We are going to build a TensorRT runtime for the GoogLeNet model trained in DIGITS introduced in Part 1. 

### Build Phase

In the build phase, TensorRT performs optimizations on the network configuration and generates an optimized plan for computing the forward pass through the deep neural network. The *plan* is an optimized object code that can be serialized and stored in memory or on disk.

We start by using the model job directory from DIGITS, just as we did in Task 2 using pycaffe, to get two files:

1. a network architecture file (deploy.prototxt)
2. trained weights (net.caffemodel)

**Exercise:** Find the job directory for the GoogLeNet model in DIGITS and use it as the MODEL_JOB_DIR parameter "FIXME" in the code cell below:


    import os
    
    # Set the model job directory from DIGITS here
    MODEL_JOB_DIR='/jobs/20160908-203605-e46b'
    
    # We need to find the iteration number of the final model snapshot saved by DIGITS
    for root, dirs, files in os.walk(MODEL_JOB_DIR):
        for f in files:
            if f.endswith('.solverstate'):
                last_iteration = f.split('_')[2].split('.')[0]
    
    architecture_file = MODEL_JOB_DIR+'/deploy.prototxt'
    weights_file = MODEL_JOB_DIR+'/snapshot_iter_'+last_iteration+'.caffemodel'
    
    print "Architecture file name = "+architecture_file
    print "Trained weights file name = "+weights_file

    Architecture file name = /jobs/20160908-203605-e46b/deploy.prototxt
    Trained weights file name = /jobs/20160908-203605-e46b/snapshot_iter_32.caffemodel


Just as when deploying with pyCaffe above, the two files above represent the trained model. TensorRT takes those two files and some other parameters to optimize the model for a specific deployment. Take a look at the file [InferenceEngine.cpp](../edit/GRE_Web_Demo/GIEEngine/src/InferenceEngine.cpp) to see the types of parameters you can specify. 

The builder (lines 36-42) is responsible for reading the network information when the create_plan application is called.

Note: The builder can also be used to define network information directly if no .prototxt file is available.

Significant lines to inspect in this file are:

- line 46 - the output layer from the network is chosen, we are using the "softmax" layer as we want to classify images.  If we wished to use TensorRT to extract features from an intermediate layer of the network we could specify that layer by name here.
- line 49 - we choose the maximum input batch size that will be used.  In this example, we simply process one image at a time (batch size 1), but if we wished to maximize throughput at the expense of latency we could process larger batches.
- line 50 - TensorRT performs layer optimizations to reduce inference time.  While this is transparent to the user, analyzing the network layers requires memory, so you must specify the maximum workspace size.

Once you have finished inspecting the `InferenceEngine.cpp` code execute the command below to create a TensorRT object called a "plan". 

Plans are smaller than the combination of the models and weights and can be run on the deployment hardware without needing to install and run on top of a deep learning framework.

This "plan" will be from our GoogLeNet model trained in DIGITS.  Execute the cell below to do this - the final line specifies where the TensorRT object (sometimes called a plan) will be written.


    !cd /home/ubuntu/GRE_Web_Demo/GIEEngine/src/ && make && \
    echo "create_plan app created"
    
    !/home/ubuntu/GRE_Web_Demo/GIEEngine/src/create_plan \
    {architecture_file} \
    {weights_file} \
    ~/GRE_Web_Demo/GIEEngine/src/imagenet.plan && \
    echo "Plan created"

    nvcc -std=c++11 -O2 -c -o create_plan.o create_plan.cpp
    nvcc -std=c++11 -O2 -c -o InferenceEngine.o InferenceEngine.cpp
    nvcc -std=c++11 -O2 -o create_plan create_plan.o InferenceEngine.o -lglog -lnvinfer -lnvcaffe_parser
    create_plan app created
    Plan created


If the output of the block above says "Plan created," move on to deployment. 

If the output of the block above says, "Aborted (core dumped)," check to make sure you've put the Model Job Directory for GoogLeNet and not pedestrian_detectNet in the first block of task 3. 

While it'd be interesting to demonstrate *pedestrian_detectNet* through TensorRT, *pedestrian_detectNet* has some custom layers that make doing that beyond the scope of this lab, however TensorRT includes custom layer APIs which handle those. 

### Deploy Phase

With your "plan" created, the next step is to batch and stream data to and from the runtime inference engine. This is beyond the scope of this lab, however, if you have C++ experience, you can examine [classification.cpp](../edit/GRE_Web_Demo/GIE/src/classification.cpp) which contains the key steps required to use the inference engine to process a batch of input data and generate a result. 

We have created a [Docker](https://www.docker.com/) container called gre_gie that contains an application called GIE that uses `classifiation.cpp` to instantiate a TensorRT inference engine from the plan we created and process images that it is passed through a REST service called GRE. The [NVIDIA GPU REST Engine (GRE)](https://developer.nvidia.com/gre) is a critical component for developers building low-latency web services. GRE includes a multi-threaded HTTP server that presents a [RESTful](https://en.wikipedia.org/wiki/Representational_state_transfer) web service and schedules requests efficiently across multiple NVIDIA GPUs. The overall response time depends on how much processing you need to do, but GRE itself adds very little overhead and can process null-requests in as little as 10 microseconds.

Execute the cell below to use [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) to launch the gre_gie container.


    !nvidia-docker run \
        -d --name gre_gie -p 8085:8000 \
        -v /home/ubuntu/GRE_Web_Demo/GIEEngine/src:/inference-engine:ro \
        -v /home/ubuntu/caffe/data/ilsvrc12:/imagenet:ro \
        gre_gie \
        /inference-engine/imagenet.plan \
        /imagenet/imagenet_mean.binaryproto \
        /imagenet/synset_words.txt && \
    echo "GIE container running"

    No manual entry for nvidia-docker
    See 'man 7 undocumented' for help when manual pages are not available.


In order to compare the output and performance between Caffe and TensorRT, we have also created a second container called gre_caffegpu that carries out inference on images passed through GRE but uses GPU accelerated Caffe instead of TensorRT.  The Caffe container uses the same native Caffe model for inference as we used to create the TensorRT plan above.


    !nvidia-docker run \
        -d --name gre_caffegpu -p 8082:8000 \
        -v /home/ubuntu/digits/digits/jobs/20160908-203605-e46b:/model:ro \
        -v /home/ubuntu/caffe/data/ilsvrc12:/imagenet:ro \
        gre_caffegpu \
        /model/deploy.prototxt \
        /model/snapshot_iter_32.caffemodel \
        /imagenet/imagenet_mean.binaryproto \
        /imagenet/synset_words.txt && \
    echo "Caffe GPU container running"

    8a4dc470ce05482e5abb406e38bd12277414963f4a883e6e4cea545456034d3b
    Caffe GPU container running


Finally, we created a Docker container called gre_frontend that generates a simple web interface that you can upload or link an image to and have it classified using either the TensorRT application in the gre_gie container or native GPU Caffe in the gre_caffegpu container.

Execute the cell below to start the web interface.


    !nvidia-docker run --name gre_frontend -d --net=host \
     -v /home/ubuntu/GRE_Web_Demo/FrontEnd/WebFrontEnd:/HTTP_server gre_frontend FrontEnd \
        "0.0.0.0:8080" \
        "0.0.0.0:8081" \
        "0.0.0.0:8082" \
        "0.0.0.0:8083" \
        "0.0.0.0:8084" \
        "0.0.0.0:8085" \
        "0.0.0.0:8086" && \
    echo "GIE web demo container running"

    1d12012a2c166b4c056cf8973077045dad1a1f0c2647914b8a69970cc56b7c0c
    GIE web demo container running


**Once you have received the "GIE container running", "Caffe GPU container running" and "GIE web demo container running" messages above**, click [here](/gre/) to connect to the GIE web interface in a separate tab.

You should see a screen like this:

![](files/GRE.png)

You can now classify an image provided through the web interface through either the running GIE (Using TensorRT) or Caffe containers. Either click [here](files/test_images/GN_test.png) to open our GoogLeNet test image and save it to disk or if you wish you can use your own test image, either by uploading or linking to a .png or .jpeg:

1. Click on "Image Classification" in the top left corner
2. Input URL or click choose file and point to your test image.
3. Click either the "Caffe" or "GIE" buttons to pass the image to the running GIE or Caffe container for classification
4. Click both the "profile" buttons to plot the time taken for making the web request, image preprocessing and inference using GIE and Caffe

You should see an output like this:

![](files/GREoutput.png)

We see that the GIE inference time (dark green bar, using TensorRT) is lower than the Caffe time (light green bar). We also see that we got the same output using both models.

TensorRT optimizes models for inference by performing several important transformations and optimizations to the neural network graph. 

Note that the inference time here includes image preprocessing and the overhead of making a web request, so it is not a true representation of the maximum possible inference speed of TensorRT or Caffe and it is also running an older K520 GPU. To measure the true performance benefits we compared the per-layer timings of the GoogLeNet network using Caffe and TensorRT on NVIDIA Tesla M4 GPUs with a batch size of 1 averaged over 1000 iterations with GPU clocks fixed in the P0 state.

![](files/GIE_GoogLeNet_top10kernels-1.png)

For more information on these transformations see the TensorRT Parallel Forall blog post [here](https://devblogs.nvidia.com/parallelforall/production-deep-learning-nvidia-gpu-inference-engine/). Note that TensorRT currently supports the following layer types:

- Convolution: 2D
- Activation: ReLU, tanh and sigmoid
- Pooling: max and average
- ElementWise: sum, product or max of two tensors
- LRN: cross-channel only
- Fully-connected: with or without bias
- SoftMax: cross-channel only
- Deconvolution


    # stop and remove any existing docker containers
    !sudo docker stop $(docker ps -a -q)
    !sudo docker rm $(docker ps -a -q)

You're now able to take model definition files and trained weights from DIGITS and deploy them using pycaffe and TensorRT. You've seen the effect of batchsize on throughput and latency. You've also seen the difference in workflow and performance when using a high performance inference engine. Combining these skills and this knowhow with previous labs, you should be able to begin solving supervised machine learning problems from dataset to production.

Keep learning! Check out our other offerings with the link below.

 <a href="https://www.nvidia.com/en-us/deep-learning-ai/education/"> <img src="files/DLI Header.png" alt="Drawing" style="width: 400px;"/> </a>
