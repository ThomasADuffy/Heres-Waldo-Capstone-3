# Heres-Waldo-Capstone-3
Utilizing Computer vision (Open CV) and a CNN to find waldo

<p align="center"> 
<img src="/imgs/shining_waldo.jpg">
</p>

# **Table of Contents** <!-- omit in toc -->
- [Heres-Waldo-Capstone-3](#heres-waldo-capstone-3)
- [**Introduction**](#introduction)
- [**How to run and utilize this app**](#how-to-run-and-utilize-this-app)
- [**Contact Me!**](#contact-me)
- [**Readme Images and Data Credits/Sources**](#readme-images-and-data-creditssources)
  - [Readme/Poster Images sources](#readmeposter-images-sources)
  - [Datasets sources](#datasets-sources)

# **Introduction**
This Project is based off of my a preliminary project which can be found [here](https://github.com/ThomasADuffy/Whos-Waldo-Capstone-2). Please read it before continuing to get a full scope of my images.


Readme WIP! Please wait!, Feel free to contact me with more information!



# **How to run and utilize this app**
to create docker container, run these commands from the root of this github repo:  
**(Please note that this is very large(~3.5GB) due to the anaconda docker package being installed)**

>docker image build -t waldo_finder . 
   
>docker container run --publish 8080:8080 --detach --name waldo waldo_finder  
  
Then simply just go to the address listed in the terminal.(This will also work on an Could based instance like AWS EC2)

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