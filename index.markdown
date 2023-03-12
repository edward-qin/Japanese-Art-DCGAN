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
    displayMath: [ ['$$','$$'], ['\[','\]'] ],
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

A GAN is a generative adversarial network, which is composed of a generator and discriminator model. The discriminator attempts to determine if input images are part of the training data set versus if they are from the generated fake images. The generator attempts to generate images to fool the discriminator into thinking they are real. 

In this project, we are using a DCGAN, a deep convolutional GAN, which uses convolutional and convolutional-transpose layers in the discriminator and generator respectively. 

If $x$ represents the input image data, then $$D(x)$$ represents the probability that the discriminator determines that $x$ came from the training dataset. 

Now suppose we have a latent space vector $z$ composed of random values such that when fed to the generator, $G(z)$ maps $z$ to a data space representing an image. The goal of the generator is to estimate the distribution $p_{data}$ that generates the real images from the training data. Formalized, this is when $D(G(z)) = D(x)$, meaning the discriminator cannot discriminate between real and generate images. 

Likewise, the distribution of the generated images $p_g$ is the same as the distribution of the input latent space vector $p_z$. As such, the loss function of the GAN is as follows:

$$\min_G\max_DV(D, G) = \mathbb{E}_{x \sim p_{data}(x)}[logD(x)] + \mathbb{E}_{z \sim p_{z}(z)}[log(1-D(G(z)))]$$

We can see that this loss function is similar to the `BCELoss` function which we use in our training. 

# Version 1

Original, based on pytorch DCGAN

# Modification 1

Label smoothing + activation function change

# Modification 2

Label smoothing only

# Setbacks

Spiking - mode collapse, GAN is hard to train
Unstructured data - harder for model to learn the structure (but got the style)
Converging data - when using both label smoothing and activation function change outputs became similar and only becoming more blurred. 

# Conclusion

What would do if more time
If we successfully train this model, we can expand it to also generate different styles from the same dataset on Kaggle.

Check out our [model](https://colab.research.google.com/drive/16f-V6o3iB7EYTjML0i0ebYNm2gsts9WP?usp=sharing) on Google Colab and our [video demo]()!


