# Image Classification for Property Tech

Capstone Project for General Assembly Data Science Immersive (DSI 19 Singapore), Feb 2021

## Problem Statement
London is one of the world's most interesting property market, due to its long history, its prominent status as a international financial center, as well as the mix of modern and period buildings.
For property buyers or investors, there is one crucial piece of information not easy to obtain in the public domain:
- **Age of Building**, or simply categorised as **Period/Old vs Modern**

As a result, most buyers spend a great deal of time in property search, and receive poor recommendations from auto-generated emails.

This project aims at filling the information gap
by classifying buildings as Period vs Modern, based on 2000+ images from leading portals (Rightmove, Zoopla, Purple Bricks)

A divide-and-conquer approach was taken to drastically cut down the effort in labelling, and increase the output accuracy.

Results: metrics ~90% (AUC, accuracy)

## Datasets
- 3000+ images from public portals eg Rightmove, Zoopla, PurpleBricks
- 2000+ labels done manually

Link to all images in Google Drive [here](https://drive.google.com/drive/folders/1d39ZSjKYY07WGX62vmVFfqE0vXFbVuuM?usp=sharing)

#### Features to extract
- Period vs Modern (primary goal)
- Exterior vs Interior (secondary, which turns out to be crucial)

<img src="graphics/features_extracted.jpg" alt="other_challenges" width="60%">

Figure 1: (from top left to bottom right): Period Exterior, Period Interior, Modern Exterior, Modern Interior


#### Challenges with the images
Several issues were found out in the initial data exploration and analysis stage
- Various exterior styles: period buildings include Georgian (1714-1837), Victorian (1837-1901), Edwardian (1901-1910), council houses, old houses, terraced or detached houses

<img src="graphics/exterior_styles.jpg" alt="other_challenges" width="90%">
Figure 2: Various exterior styles of London period buildings

Other issues:
- Refurnished interior: period buildings are well maintained, making it difficult to distinguish it from the modern ones
- Retro: some modern buildings are designed in a retro styles
- Other challenges include blocked views, watermarks and repeated images

<img src="graphics/other_challenges.jpg" alt="other_challenges" width="90%">
Figure 3: Other challenges include refurnished interior (left), retro buildings (centre), blocked views (right)

## Divide and Conquer Strategy
The ultimate goal is to classify period vs modern buildings. However, during the data exploration and labelling, it became apparent that there is limitation on accuracy based on interior images
Sometimes classification may not be accurate even by human eyes.

<img src="graphics/divide_n_conquer.jpg" alt="divide_n-conquer" width="40%">
Figure 4: Focused on analysing a subset of data (Exterior Period vs Exterior Modern) to boost performance


#### Strategy adopted: Divide and Conquer ####
- Separate “good” images from the pool ie identify “exterior” images as they tend to give more information
- Feed through two layers of classification models
- 1st layer model for classifying exterior vs interior. A customised CNN model is able to learn how to classify this effectively (AUC and accuracy > 95%)
- 2nd layer model for classifying period vs modern, based on exterior images only. Empirically, transfer learning performs much better than customised CNN (AUC and accuracy 90%+ vs 70-80%)


## Data Extraction and Analysis

Web scrapping of the major property portals in UK eg Rightmove, Zoopla, PurpleBricks. A total of 2232 images were obtained, alongside information such as coordinates, description

Summary of features/labels
- 2100 unique images, 132 duplicated images
- Among the unique images, 609 exterior images, 1491 interior images
- Among the unique images, 1085 images of new buildings, 1015 period buildings

Manual labelling
- First round of labelling: basic key word search eg Georgian, Victorian etc to apply first labels
- Second round of labelling: glancing through the images grouped by "first labels", then fine tune the labels manually

Two rounds of labelling drastically reduced the effort and time required

## Data Preprocessing
- Image hashing, using the library [ImageHash](https://github.com/JohannesBuchner/imagehash)
- Resizing
- Standard Scaling


## Modelling
Two Convolutional Neural Network (CNN) models were used:
- Customised CNN
- Transfer learning based on VGG16 (the seminal paper at this [link](https://arxiv.org/abs/1409.1556))


## Overall process
<img src="graphics/overall_process.jpg" alt="overall_process">
Figure 5: Overall process of Data Preprocessing and Modelling



## Models for two layers

### First Layer - Customised Convolutional Neural Networks (CNN)
#### Customised structure
- Two convolutional layers with 6 and 16 filters respectively, each followed by a max pooling layer of 9x9
- One flattening layer
- One layer of 256 densely connected neurons
- One dropout layer, dropout rate 0.5
- Final Sigmoid activation layer
- Optimizer: Adam

Grid Search CV was applied to determine the best parameters
Faster computation in each iteration, but more iterations needed

### Second Layer - Transfer Learning based on VGG16
#### VGG16 as the input model
- No change in the original VGG16 structure and parameters
- One additional layer of 128 densely connected neurons
- One dropout layer, dropout rate 0.5
- Final Sigmoid activation layer

Quicker in convergence, less than 15 epochs based on the tolerance
Longer computation in each iteration due to many more layers


## Model performance
#### Customised CNN
- Classified images into exterior vs interior very effectively
- AUC: 99%
- Accuracy: 95% (238/250 in validation set), vs baseline at 50% (balanced class)
- Data set of 1000 images, 500 each class (Exterior vs Interior)
- [<u><ins>Notebook</ins></u>](code/eda_model7d.ipynb)

<img src="graphics/Model7d_ROC.png" alt="cnn">
Figure 6: ROC of Customised CNN in the first layer of classification (interior vs exterior)

#### Images classified as Interior
<img src="graphics/classified_interior.jpg" alt="interior">

#### Images classified as Exterior
<img src="graphics/classified_exterior.jpg" alt="exterior">

#### Transfer Learning CNN (VGG16)
- Classified exterior images into period vs modern very effectively
- AUC: 96%
- Accuracy: 88% (123/140 in validation set), vs baseline at 50% (balanced class)
- Data set of 560 images, 280 each class (Period vs Modern)
- [<u><ins>Notebook</ins></u>](colab/vgg16c.ipynb)

<img src="graphics/VGG16c_ROC.png" alt="vgg">

#### Images classified as Period/Old
<img src="graphics/classified_period.jpg" alt="period">

#### Images classified as Modern/New
<img src="graphics/classified_modern.jpg" alt="modern">


#### Model Comparison
- Transfer Learning CNN is superior to the customised CNN in dealing with exterior vs interior
- AUC was improved to ~95% from ~80%

#### Why divide and conquer?
- The metric dropped to mediocre levels (70-80%) if all exterior and interior images are used for classification
- That will limit the commercialisation of application
- [<u><ins>Notebook</ins></u>](code/eda_model5c_all_images_grid.ipynb)


## Unseen data plotted on London map
- The final challenge is to feed 1000 images (all unseen data) to go through the preprocessing and two layers of models
- The result is plotted on the London map
- It is encouraging to see period houses are concentrated in South Kensington (orange circle), and modern buildings are more common in Canary Wharf (blue circle)

<img src="graphics/map.jpg" alt="map">
Figure 7: London map - how period houses and modern buildings are distributed


## Conclusion
A feasible and reliable process and modelling has been established to accurately classify period vs modern buildings
With a divide-and-conquer approach, two layers of model were able to learn to classify two set of features
Potential commercial applications in areas of property tech:
- Property Search Engine
- Property Recommender System
- Property Investment Evaluator

There is room for future development to fine tuning the model and aggregating the model outputs with other data currently available



## Datafile dictionary

/data/images_train_val.csv

####
| Field | Description |
|----------|------------------------------------------|
| index | index of the image in the record |
| image_link | full link of image |
| hash | hash value from Average Hash algorithm converted to string |
| repeated | 1 denotes the first occurence of hash value; any values >1 refer to duplicated values |
| ext | 1 denotes "exterior" images, 0 denotes "interior" images |
| new | 1 denotes "modern/new" building, 0 denotes "period/old" buildings  |



## Folder Organization

    |__ code/
    |   |__ eda_model5c_all_images_grid.ipynb   
    |   |__ eda_model7d.ipynb
    |__ colab/
    |   |__ vgg16c.ipynb   
    |__ common/
    |   |__ myfunctions.py
    |__ data
    |   |__ images_train_val.csv
    |__ presentation
    |   |__ Capstone20210225.pdf
    |__ graphics/	
    |__ images/ (Google Drive)
    |   |__ map/
    |   |__ modern_exterior/
    |   |__ modern_more/
    |   |__ modern_samples/
    |   |__ old_interior/
    |   |__ old_more/
    |   |__ old_samples/
    |   |__ predict_samples/
    |__ README.md
	
