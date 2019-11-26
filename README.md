# Heres-Waldo-Capstone-3
Utilizing Computer vision (Open CV) and a CNN to find waldo

<p align="center"> 
<img src="/imgs/shining_waldo.jpg">
</p>

# **Table of Contents** <!-- omit in toc -->
- [Heres-Waldo-Capstone-3](#heres-waldo-capstone-3)
- [**Introduction**](#introduction)
- [**Initial results**](#initial-results)
- [**Building a New Model**](#building-a-new-model)
- [**Results of Model**](#results-of-model)
- [**Multicore Processing**](#multicore-processing)
- [**Flask Application**](#flask-application)
- [**How to run and utilize this app**](#how-to-run-and-utilize-this-app)
- [**Contact Me!**](#contact-me)
- [**Readme Images and Data Credits/Sources**](#readme-images-and-data-creditssources)
  - [Readme/Poster Images sources](#readmeposter-images-sources)
  - [Datasets sources](#datasets-sources)

# **Introduction**
This Project is based off of my a preliminary project which can be found [here](https://github.com/ThomasADuffy/Whos-Waldo-Capstone-2). Please read it before continuing to get a full scope of my process and work.

For this project I utilized a CNN to scan through a fed in image utilizing the sliding window technique to find Waldo. I also ended up utilizing Multiprocessing to be able to complete this process much faster than originally. At first it would take 3-5 minutes but now if you use a multi core processor, it will take around 30-40 seconds.  

The whole basis of this model was implementing the sliding window technique this mainly was done by setting a window size and then physically sliding that window across the page and classifying each window to check if it was waldo or not.  

<p align="center">
Sliding window Example <br>
<img src="/imgs/gif/sliding_window_example.gif"><br>
</p>
Then I would save the image with the boxes where the Waldo probability was above the threshold set, colored green. 

# **Initial results**

I first started off by utilizing my original model that I created in my capstone project 2. That proved to be ineffective as the model, though was extremely accurate had so many false positives seen below.  

<p align="center">
Initial model v1 <br>
<img src="/imgs/model_results/test/315_waldos_test6_model_v1.jpg"><br>
<img src="/imgs/model_results/test/390_waldos_test2_model_v1.jpg"><br>
</p>
<p align="center">
Initial model v2 <br>
<img src="/imgs/model_results/test/188_waldos_test5_model_v2.jpg"><br>
<img src="/imgs/model_results/test/390_waldos_test2_model_v1.jpg"><br>
</p>
Though for some reason it seemed to work on one page extremely well  
<p align="center">
Initial model v1<br>
<img src="/imgs/model_results/test/2_waldos_test3_model_v1.jpg"><br>
Initial model v2<br>
<img src="/imgs/model_results/test/15_waldos_test3_model_v2.jpg"><br>
</p>
I assume this is due to the color grading and/or the training data containing a lot of non-waldo pictures from this exact page.  
  
# **Building a New Model**

I implemented a method to save all of the windows where it was classified as Waldo (with the default threshold of 50%). Then I manually sifted through the photos and separated weather a photo contained waldo or not. This allowed me to gain ~6900 more non-Waldo photos to further train a new model. In addition to these new non-waldo photos, I also generated more waldo photos (See Previous project to get more information about Image Generation) to ensure there was a class balance before training my new models.  

At first I utilized the same CNN structure as my model v2, but i had more images i was training it on. This proved ineffective as it still was a pretty terrible model though had a lot less false positives. It didn't even find waldo in the example below:
<p align="center">
Initial model v3<br>
<img src="/imgs/model_results/test/128_waldos_test2_model_v3.jpg"><br>
</p>

I released that accuracy was not the metric I should be looking at but actually recall was the metric I should have been paying attention to the whole time. Therefor when creating my new model I implemented Callbacks (A Tensorflow setting to save models every interval during training looking at a metric to save the best model) focusing on recall. I also believed that my model was not learning enough abstract features about Waldo and decided to add another convolution convolution pooling dropout set of layers to my original models architecture and ended up with this architecture below for my final model:  
<p align="center">
Current Model Structure<br>
5,229,185 total parameters<br>
<img src="/imgs/readme_imgs/model_v4_structure.png"><br>
</p>

# **Results of Model**

My model had relatively high recall, it still contained some false positives but it would always correctly find Waldo with a relatively high probability. The optimal threshold I also found was 65.5%, Though most of the time it will classify waldo at 90%>. Here are the results with the test set and hold out set.(for the holdout set I ended up limiting it to only highlight the 10 top waldos).

<p align="center">
Current Model Results<br>
Test<br>
<img src="/imgs/model_results/test/14_waldos_test6_model_v4.jpg"><br>
<img src="/imgs/model_results/test/30_waldos_test2_model_v4.jpg"><br>
<img src="/imgs/model_results/test/14_waldos_test5_model_v4.jpg"><br>
Holdout<br>
<img src="/imgs/model_results/best_model/23_waldos_holdout1_model_v4.jpg"><br>
<img src="/imgs/model_results/best_model/29_waldos_holdout3_model_v4.jpg"><br>
<img src="/imgs/model_results/best_model/35_waldos_holdout8_model_v4.jpg"><br>
<img src="/imgs/model_results/best_model/3_waldos_holdout7_model_v4.jpg"><br>
</p>

As you can see the model performed much better than the previous models but it still had an optimization problem. This process would take around 3-5 minutes depending on the size of the image and was way too slow to be functional so I ended up using multiprocessing.

# **Multicore Processing**

This process  was extremely slow so I decided to end up utilizing multiple cores to optimize this process. I thought the best way to do this would be to split the image up from one image into multiple slices then scan through the slices simultaneously on different cores and return the coordinates of where the core thought it found waldo. seen below, the original image was chopped into 9 slices for 9 different cores.  

<p align="center">
Original Image<br>
<img src="/imgs/test1.jpg"><br>
Sliced Images<br>
<p align="center"><img src="/imgs/readme_imgs/slice0.jpg"><hr>
<p align="center"><img src="/imgs/readme_imgs/slice1.jpg"> <hr>
<p align="center"><img src="/imgs/readme_imgs/slice2.jpg"> <hr>
<p align="center"><img src="/imgs/readme_imgs/slice3.jpg"> <hr>
<p align="center"><img src="/imgs/readme_imgs/slice4.jpg"> <hr>
<p align="center"><img src="/imgs/readme_imgs/slice5.jpg"> <hr>
<p align="center"><img src="/imgs/readme_imgs/slice6.jpg"> <hr>
<p align="center"><img src="/imgs/readme_imgs/slice7.jpg"> <hr>
<p align="center"><img src="/imgs/readme_imgs/slice8.jpg"> <hr>
</p>

In order to find the optimal number of cores dynamically, I created an algorithm which will find the optimal number of cores so no matter what machine it is running on it should be able to do multi-core processing on it if the machine has multiple cores. I did this by finding the number of cores and total rows and incrementing the number of rows per core until the number of cores required was below the number of available cores. Please see [here](https://github.com/ThomasADuffy/Heres-Waldo-Capstone-3/src/multiprocessing_helpers.py) for more details on the algorithm.  

This speed up the process drastically and allowed me to predict on a large image within 30-40 seconds and even more so on computing power focused instances.

# **Flask Application**
I implemented that into a Web app utilizing Flask. This web app works with smartphones and was tested by having a smartphone take a picture of a wheres waldo page and it would return the top 10 windows of where it thought Waldo was. Below are some pictures of the web app tested on a uploaded scanned image of a page which took around 20 seconds.  
<p align="center">
Web App<br>
Landing Page<br>
<img src="/imgs/readme_imgs/flask_app/Heres Waldo! The Waldo finding app! - Mozilla Firefox_002.jpg"><br>
Uploading A Picture<br>
<img src="/imgs/readme_imgs/flask_app/Find Waldo! - Mozilla Firefox_003.jpg"><br>
Returned picture with found image(Retained Quality)<br>
<img src="/imgs/readme_imgs/flask_app/Find Waldo! - Mozilla Firefox_004.jpg"><br>
<img src="/imgs/readme_imgs/flask_app/Find Waldo! - Mozilla Firefox_005.jpg"><br>
Contact Me Page<br>
<img src="/imgs/readme_imgs/flask_app/Contact Me - Mozilla Firefox_006.jpg"><br>
</p>

Feel free to download and utilize this app also!!!


# **How to run and utilize this app**

First Git clone this repo to any directory and install Docker.  
Next open up command line or terminal and navigate to the root of this repo on your local machine or cloud instance  
to create docker container, run these commands from the root of this github repo:  
**(Please note that this is very large(!1.5GB) due to the tensorflow docker package being installed)**

>docker image build -t waldo_finder .  
   
>docker container run --publish 8080:8080 --name waldo waldo_finder  

(if using a cloud instance add --detach if you would like to detach from the container so you can leave it live without keeping an open SSH connection)
  
Then simply just go to the address listed in the terminal. Please note this defaults to utilizing port 8080 on your machine, make sure it is not utilzed.  
(This will also work on an Cloud based instance like AWS EC2)

# **Contact Me!**
<p class="lead" align="center"><font size='4'>Linked-In<br> <a href="https://www.linkedin.com/in/thomas-a-duffy/">Thomas Duffy</a><br></font><hr />
   <p class="lead" align="center"> <font size='4'>E-Mail</font><br>
    <font size='3'><strong> tommy.duffy@gmail.com</strong><br></p>
    <hr />
    <p class="lead" align="center"> <font size='4'>Github<br>
    <a href="https://github.com/ThomasADuffy">Thomas Duffy</a><br></font></p>

# **Readme Images and Data Credits/Sources**  
## Readme/Poster Images sources

Ennui ~ Elegant Neural Network User Interface ~
Jesse Michel-Zack Holbrook-Stefan Grosser-Hendrik Strobelt-Rikhav Shah - https://math.mit.edu/ennui/

The Shining. (1980). film. Britain: Stanley Kubrick, Warner Bros. 

A Comprehensive Guide To Convolutional Neural Networks - the Eli5 Way
Sumit Saha - https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

## Datasets sources  
Constantinou , Valentino : vc1492a, 2018, Hey Waldo, V1.8, Github, https://github.com/vc1492a/Hey-Waldo  

Sliding Windows For Object Detection with Python and Opencv
Adrian Rosebrock - https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/   
  
Handford, Martin. Where's Waldo? * Somerville, MA :Candlewick Press, 2007.  

I would also like to thank the entire Cohort of g99, the entirety of the Galvanize staff, Frank Burkholder, Kayla Thomas, Nicolas Jacobsohn and Angela Hayes for all there help and support throughout this course. Without all of you this wouldn’t have been possible.  
  
*(The entire collection of Wheres Waldo Books was used, Scanned images of pages for data.)