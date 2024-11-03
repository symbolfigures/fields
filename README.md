# Third Study Animation

This repository is a pipeline that takes scans of drawings and produces animations from them.

1. Drawings are scanned into the computer and processed into a dataset.
2. A generative discriminatory network (a.k.a. generative adversarial network) trains on this dataset and yields a vector space of potential, immitation images.
3. Animations are produced by generating images along continuous paths within the vector space.

The code for step 2 is provided in [Brad Klingensmith's course on Udemy](https://www.udemy.com/course/high-resolution-generative-adversarial-networks). That code is in turn based on the [ProGAN](https://arxiv.org/abs/1710.10196), [StyleGAN](https://arxiv.org/abs/1812.04948), and [StyleGAN2](https://arxiv.org/abs/1912.04958) papers. Links to some of my own drawing scans and a pre-trained generator are provided below, but the code from the course will still be needed to generate images.

The following is a step by step walkthrough on a Linux operating system. It assumes some familiarity with a command line interface.

### i. Requirements

- Python 3.12
- Code from the course above
- Other Python libraries listed in `requirements.txt`
- bash
- ImageMagick
- ffmpeg

### ii. Setup

Clone the repository.  
`$ git clone git@github.com:symbolfigures/thirdstudy.git `

Create a virtual environment, and install the required packages.  
`$ cd thirdstudy`  
`$ python3.12 -m venv .venv`  
`$ source .venv/bin/activate`  
`$ pip install -r requirements.txt`

There are four folders that comprise the pipeline:  
`/data` - Original drawing scans, processed images, and final dataset.  
`/train` - The trained neural network, which is referenced by the image generator.  
`/anim` - Generated images and animations.  
`/color` - Color images.

## 1. Create the dataset

See [`data`](data)

## 2. Train the image generator

See [`train`](train)

## 3. Create the animation

See [`anim`](anim)

## 4. Color images

See [`color`](color)

















