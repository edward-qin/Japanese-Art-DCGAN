---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: GAN
permalink: /
---

<script>
MathJax = {
  tex: {
    inlineMath: [ ['$', '$'], ['\\(', '\\)'] ],
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>

Authors: Daniel Gao, Edward Qin

# Introduction

Our goal for this project was to generate new visual art in the style of Japansese art from Japan from the Muromachi period (1392-1573) all the way to the Shōwa period (1926-1989). We used a Generative Adversarial Network (GAN), which is composed of a generative and discriminative network. The dataset we used is the Japanese Art directory from the [WikiArt Art Movements/Styles](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles) dataset on Kaggle. 

# GAN Model

A GAN is a generative adversarial network, which is composed of a generator and discriminator model. The discriminator attempts to determine if input images are part of the training dataset (labeled 1) versus if they are from the generated fake images (labeled 0). The generator attempts to generate images to fool the discriminator into thinking they are real. 

In this project, we use a DCGAN, a deep convolutional GAN, which uses convolutional and convolutional-transpose layers in the discriminator and generator respectively. 

If $x$ represents the input image data, then $D(x)$ represents the probability that the discriminator determines that $x$ came from the training dataset. 

Now suppose we have a latent space vector $z$ composed of random values such that when fed to the generator, $G(z)$ maps $z$ to a data space representing an image. The goal of the generator is to estimate the distribution $p_{data}$ that generates the real images from the training data. Formalized, this is when $D(G(z)) = D(x)$, meaning the discriminator cannot discriminate between real and generate images. 

Likewise, the distribution of the generated images $p_g$ is the same as the distribution of the input latent space vector $p_z$. As such, the loss function of the GAN is as follows:

$$\min_G\max_DV(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[logD(x)] + \mathbb{E}_{z \sim p_{z}(z)}[log(1-D(G(z)))]$$

We can see that this loss function is similar to the PyTorch `BCELoss` function which we use in our training. 

# Version 1

For the first version of our model, we based the architecture on the PyTorch DCGAN tutorial model.

The input to the generator is $z$, the latent space vector of length 100. In each of the 5 convolutional-transpose layers, we upsample the input by a factor of $2 \times 2$. After each convolutional-transpose before the final convolutional-transpose, we also apply a batch normalization followed by a ReLU activation. On the last convolutional-transpose layer, we do a final $\tanh$ activation function on the $3\times64\times64$ image.

The discriminator architecture is like the inverse of the generator, with input image of size $3 \times 64 \times 64$. We have 5 convolutional layers that each downsample by a factor of $2 \times 2$. We apply batch normalization on layers other than the first and last layer, and all layers other than the last layer use the Leaky ReLU activation function. In the final layer, we apply the sigmoid function to determine the probability that image is real, $D(x)$.

To train our model, we first resized the 2235 images of our dataset to $64 \times 64$ images. We then used a batch size of 128, learning rate of 0.0005, the Adam optimizer with $\beta_1$ coefficient 0.5, and 300 epochs. 

Our results are then as follows


# Version 2

We noticed that in Version 1, our loss function was diverging. This suggested that our discriminator was becoming too accurate too quickly. Thus, we attempted to slow the rate of change in discriminator loss. We did so by applying label smoothing, which changed the labels from 0 and 1 to ranges $[0, 0.3]$ and $[0.7, 1]$, as well as changing the generator's activation functions from ReLU to Leaky ReLU. These changes were based on tips from this [github page](https://github.com/soumith/ganhacks).

Our results were as follows

# Version 3

Based on setbacks in Version 2, we decided that the 

# Setbacks

Spiking - mode collapse, GAN is hard to train
Unstructured data - harder for model to learn the structure (but got the style)
Converging data - when using both label smoothing and activation function change outputs became similar and only becoming more blurred. 

# Conclusion

What would do if more time
If we successfully train this model, we can expand it to also generate different styles from the same dataset on Kaggle.

Check out our [model](https://colab.research.google.com/drive/16f-V6o3iB7EYTjML0i0ebYNm2gsts9WP?usp=sharing) on Google Colab and our [video demo]()!


