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
* After cloning the repo transfer the files to your project folder. Open terminal and go to the project folder and run the following commands
```sh
cd .../projectfolder
python3 landmarks.py
```


<!-- RESULTS AND DEMO -->
## Results and Demo
<!-- Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space.   -->
<!-- [**result screenshots**](https://result.png)   -->
<!-- ![**result gif or video**](https://result.gif)   -->
### Detecting Emotions

[outputs](https://user-images.githubusercontent.com/84740927/137188526-28e39eb4-081c-421b-931b-7fe20ec0b5a4.gif)

### Displaying Statistical Data For Emotions
![Displaying Statistical Data For Emotions](docs/results/graph.png)
<!-- | Use  |  Table  | -->
<!-- |:----:|:-------:| -->
<!-- | For  | Comparison| -->


<!-- FUTURE WORK -->
## Future Work
* See [todo.md](https://todo.md) for seeing developments of this project
- [x] To Make an emotion detector model
- [x] To connect it to a live feed for live detection
- [x] To give statistical data in the form of graphs
- [ ] To increase the accuracy of the model
- [ ] To deploy the model in the form of an emotion detector app or site


<!-- TROUBLESHOOTING -->
## Troubleshooting
* Common errors while configuring the project


<!-- CONTRIBUTORS -->
## Contributors
* [Anushree Sabnis](https://github.com/MOLOCH-dev)
* [Saad Hashmi](https://github.com/hashmis79)
* [Shivam Pawar](https://github.com/theshivv)
* [Vivek Rajput](https://github.com/Vivek-RRajput)


<!-- ACKNOWLEDGEMENTS AND REFERENCES -->
## Acknowledgements and Resources
* [SRA VJTI](http://sra.vjti.info/) Eklavya 2020  
* Refered [this](https://www.coursera.org/learn/introduction-tensorflow) for understanding how to use tensorflow
* Refered [this](https://www.coursera.org/learn/convolutional-neural-networks) course for understanding Convolutional Neural Networks
* Refered [towardsdatascience](https://towardsdatascience.com/) and [machinelearningmastery](https://machinelearningmastery.com/) for frequent doubts  
...


<!-- LICENSE -->
## License
Describe your [License](LICENSE) for your project.
