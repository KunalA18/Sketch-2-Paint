# Sketch-2-Paint
Style transfer using GANs

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Tech Stack](#tech-stack)
  * [File Structure](#file-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
* [Usage](#usage)
<!-- * [Results and Demo](#results-and-demo) -->
* [Future Work](#future-work)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [Acknowledgements and Resources](#acknowledgements-and-resources)
* [License](#license)


<!-- ABOUT THE PROJECT -->
## About The Project
<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com)   -->
# Aim

Aim of this project is  to build a Conditional Generative Adversarial Network which accepts a 256x256 px black and white sketch image and predicts the colored version of the image without knowing the ground truth.

# Description

Sketch to Color Image generation is an image-to-image translation model using Conditional Generative Adversarial Networks as described in the original paper by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros 2016, [Image-to-Image Translation with Conditional Adversarial Networks.](https://arxiv.org/abs/1611.07004)
When I first came across this paper, it was amazing to see such great results shown by the authors and the fundamental idea was amazing on its own too. Refer to our [documentation](https://towardsdatascience.com/generative-adversarial-networks-gans-89ef35a60b69)
<!-- Refer this [documentation](https://link/to/report/) -->

### Tech Stack
This section should list the technologies you used for this project. Leave any add-ons/plugins for the prerequisite section. Here are a few examples.
* [Keras](https://keras.io/)
* [TensorFlow](https://www.tensorflow.org/)
* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/doc/#)  
* [Google Colab](https://colab.research.google.com/)
* [Kaggle Dataset(Anime Sketch Colorization Pair)](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair)


### File Structure
    .
    ├── docs                    # Documentation files (alternatively `doc`)
    │   ├── report.pdf          # Project report
    │   └── results             # Folder containing screenshots, gifs, videos of results
    ├── MOODYLYSER2f.ipynb                  # Training program for the Model
    ├── Moodelld1_5de.h5                  # Pretrained Model with set weights
    ├── README.md
    ├── landmarks.py                  # Connects the model to a live videofeed via webcams


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* The installations provided below are subjective to the machine your are using
* We used [pip install(https://pip.pypa.io/en/stable/)] for the installations. If you don't have pip please follow the following command
```sh
 python3 -m pip install -U pip
```
* List of softwares with version tested on:
  * TensorFlow
  
    [Tensorflow 2.2 and above](https://www.tensorflow.org/install/pip)
    
  * Python

    [Python 3.6 and above](https://www.python.org/downloads/release/python-360/)

  
  * Numpy
  ```sh
   python3 -m pip install numpy
  ```
  * dlib
  ```sh
   pip install cmake
   pip install dlib
  ```
  * Download the Shape predictor file from [here(https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)] and insert it in your current project folder
  * Matplotlib
  ```sh
   python3 -m pip install matplotlib
  ```
  * OpenCV
  ```sh
   python3 -m pip install opencv-contrib-python
  ```
  * GPU support requires a CUDA®-enabled card
  
    (https://www.tensorflow.org/install/gpu)



### Installation
1. Clone the repo
```sh
git clone https://github.com/KunalA18/Sketch-2-Paint
```


<!-- USAGE EXAMPLES -->

## Usage

Once the requirements are checked, you can easily download this project and use it on your machine in few simple steps.

* **STEP 1** <br>
    Download this repository as a zip file onto your machine and extract all the files from it.

    ![Download and Extract Zip Folder](./assets/DownloadAndExtractFiles.gif)

    <br>

* **STEP 2** <br>
  Run the [runModel.py](./runModel.py) file using python to see the solution

  ![Run runModel.py Using Python](./assets/RunModelPythonFile.gif)

  <br>

> NOTE:
>
> 1 - You will have to change the path to dataset as per your machine environment on line #12. You can download the dataset from Kaggle at [https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair).
>  
> 2 - GANs are resource intensive models so if you run into OutOfMemory or such erros, try customizing the variables as per your needs available from line #15 to #19

* **STEP 3** <br>
  After the execution is complete, the generator model will be saved in your root direcrtory of the project as `AnimeColorizationModelv1.h5` file. You can use this model to directly generate colored images from any Black and White images in just a few seconds. Please note that the images used for training are digitally drawn sketches. So, use images with perfect white background to see near perfect results.

  <br>

  You can see some of the results from hand drawn sketches shown below: 

  ![Hand Drawn Sketch to Colored Image Output](./assets/HandDrawnSketchtoColoredImageOutput.png)

<!-- RESULTS AND DEMO -->
## Results and Demo
<!-- Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space.   -->
<!-- [**result screenshots**](https://result.png)   -->
<!-- ![**result gif or video**](https://result.gif)   -->
### Detecting Emotions

![Sketch to Color Image Generation using Conditional GANs](./assets/outputs.gif)


<!-- FUTURE WORK -->
## Future Works

We've been working on GANs for a lot of time and planning to continue exploring the field for further applications and research work. Some of the points that I think this project can grow or be a base for are listed below.

1. Trying different databases to get an idea of preprocessing different types of images and building models specific to those input image types.
2. This is a project applied on individual Image to Image translation. Further the model can be used to process black and white sketch video frames to generate colored videos.
3. Converting the model from HDF5 to json and building interesting web apps using [TensorFlow.Js](https://www.tensorflow.org/js).

<!-- TROUBLESHOOTING -->
## Troubleshooting
* Common errors while configuring the project


<!-- CONTRIBUTORS -->
## Contributors
* [Neel Shah](https://github.com/Neel-Shah-29)
* [Kunal Agrawal](https://github.com/KunalA18)



<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements and Resources
* [SRA VJTI](https://www.sravjti.in/) Eklavya 2021  
* Refered [this](https://www.coursera.org/learn/introduction-tensorflow) for understanding how to use tensorflow
* Refered [this](https://www.coursera.org/learn/convolutional-neural-networks) course for understanding Convolutional Neural Networks
* [Refered](https://www.tensorflow.org/tutorials/generative/pix2pix)
* Refered [towardsdatascience](https://towardsdatascience.com/) and [machinelearningmastery](https://machinelearningmastery.com/) for frequent doubts  
...


<!-- LICENSE -->
## License
Describe your [License](LICENSE) for your project.
