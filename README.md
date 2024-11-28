# Fields

This repository is a set of scripts that prepares scanned drawings as a dataset for deep learning. A succesfully trained model can then produce immitations of those drawings and even animations by transforming within its own latent space of potential images.

1. Drawings are scanned into the computer and processed into a dataset.  
2. A deep learning model with a generative adversarial network (GAN) architecture trains on this dataset and yields a vector space of potential, immitation images.  
3. The model generates images. A set of images may also be related, such as in a sequence that produces an animation.  
4. The images can be colored for a new dataset and model.

The code for step 2 is provided in [Brad Klingensmith's course on Udemy](https://www.udemy.com/course/high-resolution-generative-adversarial-networks). That code is in turn based on the [ProGAN](https://arxiv.org/abs/1710.10196), [StyleGAN](https://arxiv.org/abs/1812.04948), and [StyleGAN2](https://arxiv.org/abs/1912.04958) papers.

The [docs](docs) provide a step by step walkthrough on a Linux operating system. It assumes some familiarity with a command line interface.












