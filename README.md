# London Property Image Classification
Capstone Project for General Assembly Data Science Immersion Program in Singapore (Nov 2020)

## Problem Statement
London is one of the world's most interesting property market, due to its long history, its prominent status as a international financial center, as well as the mix of modern and old buildings.
For international investors, the ability to identify different types of buildings available in the market is one of the first steps in the investment process.
However, in the absence of information such as the age of building in the public domain, some degree of judgment is applied, and substantial manual effort is required in the search of properties.
This project aims at tackling this problem by building a classifier to distinguish the modern buildings from period and old ones, based on images only.

## Datasets
1000 images of modern buildings, and 1000 images of period and old buildings

## Model
Neural networks

## Preliminary Model Results

### Convoluted Neural Nets
- Total number of images: 560
- Images of period buildings: 280
- Images of modern buildings: 280

Model details:
- Use standard scaler
- Use early stopper
- Use dropout
- AUC as metrics

Model results:
- AUC > 90%
- Accuracy > 80% (vs baseline 50%)

### VGG16
- Transfer learning using VGG16 as the input model
- Total number of images: 491
- Images of period buildings: 241
- Images of modern buildings: 250
- Run on Google Colab

Model details:
- Dense layer: 128 neurons
- Dropout: 0.5
- Early Stopping: patience=5

Model results:
- Accuracy: 82%