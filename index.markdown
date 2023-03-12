---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
title: GAN
permalink: /
---

Authors: Daniel Gao, Edward Qin

# Introduction

Our goal for this project was to generate new visual art in the style of Japansese art from Japan from the Muromachi period (1392-1573) all the way to the Shōwa period (1926-1989). We used a Generative Adversarial Network (GAN), which is composed of a generative and discriminative network. The dataset we used is the Japanese Art directory from the [WikiArt Art Movements/Styles](https://www.kaggle.com/datasets/sivarazadi/wikiart-art-movementsstyles) dataset on Kaggle. 

# GAN Model

Theory behind the model

# Version 1

Original, based on pytorch DCGAN

# Modification 1

Label smoothing + activation function change

# Modification 2

Label smoothing only

# Setbacks

Spiking - mode collapse, GAN is hard to train
Unstructured data - harder for model to learn the structure (but got the style)

# Conclusion

What would do if more time
If we successfully train this model, we can expand it to also generate different styles from the same dataset on Kaggle.

Check out our [model](https://colab.research.google.com/drive/16f-V6o3iB7EYTjML0i0ebYNm2gsts9WP?usp=sharing) on Google Colab and our [video demo]()!


