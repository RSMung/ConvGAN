# Introduction
This repository implements the cnn-based gan including DCGAN, WGAN, and WGAN-GP.
Besides, I use the FID score and MMD metric to measure the quality of these generated data.

The EasyLossUtil used in this repository is a tool for managing these loss values. 
It's a simple library implemented by me and it's lighter and more convinent to use than tensorboard and other tools.  

# References
- DCGAN: Radford, A. (2015). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv preprint arXiv:1511.06434.
- WGAN: Arjovsky, M., Chintala, S., & Bottou, L. (2017, July). Wasserstein generative adversarial networks. In International conference on machine learning (pp. 214-223). PMLR.
- WGAN-GP: Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. C. (2017). Improved training of wasserstein gans. Advances in neural information processing systems, 30.
