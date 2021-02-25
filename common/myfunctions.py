import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from math import ceil
import random

from skimage import io
from skimage import color
from skimage.transform import rescale, resize
from sklearn.preprocessing import StandardScaler

def hello_world(name):
    print("hello world, " + name)

# function to load folder into arrays and  then it returns that same array
def load_files(path):
    # Put files into lists and return them as one list of size 4
    image_files = os.listdir(path)
    image_files = [path + x for x in image_files]
    return image_files

# feeding images into numpy ndarray
def load_array(image_files, min_size):
    X = np.array([])
    for file in image_files:
        try:
            img = io.imread(file)
            img_resized = resize(img, (min_size,min_size), anti_aliasing=True)
            if X.shape[0] == 0:
                X = np.array([img_resized])
            else:
                X = np.append(X, [img_resized], axis = 0)

        except:
            print("image error: ", file)
    return X

def load_train_val(file, criteria, k):

    df_images = pd.read_csv(file, index_col=0)
    df_images.fillna(0, inplace=True)

    df_selected = df_images
    for key, value in criteria.items():
        df_selected = df_selected[df_selected[key]==value]

    if k>df_selected.shape[0]:
        print("Number of images is less than k")

    # use random.sample ie without replacement
    selected_index = random.sample(df_selected.index.to_list(), k)

    df_selected = df_selected.loc[selected_index]

    return df_selected

def image_std_train(X_train):
    X_train_flat = X_train.reshape(X_train.shape[0],-1)
    print(X_train_flat.shape)

    ss = StandardScaler()
    X_train_flat_ss = ss.fit_transform(X_train_flat)

    X_train_ss = X_train_flat_ss.reshape(X_train.shape)
    print(X_train_ss.shape)

    return ss, X_train_ss

def image_std_test(X_test, ss):
    X_test_flat = X_test.reshape(X_test.shape[0],-1)
    X_test_flat_ss = ss.transform(X_test_flat)
    X_test_ss = X_test_flat_ss.reshape(X_test.shape)

    return X_test_ss


def display_images(display_no, df_file_info):
    # displaty_no: must be multiples of 10
    # df: must have "image_link" as a field

    display_height = display_no/10*16

    fig, ax = plt.subplots(ceil(display_no/2), 2, figsize=(8,display_height))
    j=0

    for i in df_file_info.index[:min(display_no,df_file_info.shape[0])]:
        row = j//2
        col = j%2
        image_link = df_file_info.loc[i, "image_link"]
        property_id = re.findall('[0-9]{3}[0-9]+',image_link)[0]
        ax[row][col].imshow(io.imread(image_link))
        ax[row][col].axis('off')
        ax[row][col].set_title(property_id)
        j += 1

def display_predictions(display_no, df_file_info):
    # displaty_no: must be multiples of 10
    # df: must have "image_link" as a field

    display_height = display_no/10*16

    fig, ax = plt.subplots(ceil(display_no/2), 2, figsize=(8,display_height))
    j=0
    for i in df_file_info.index[:min(display_no, df_file_info.shape[0])]:
        row = j//2
        col = j%2
        image_link = df_file_info.loc[i, "image_link"]
        property_id = re.findall('[0-9]{3}[0-9]+',image_link)[0]
        ax[row][col].imshow(io.imread(image_link))
        ax[row][col].axis('off')
        ax[row][col].set_title(property_id + ", label:" + str(df_file_info.loc[i, "label"]) + ", pred:" + str(round(df_file_info.loc[i, "pred"],0)) + ", prob:"  + str(round(df_file_info.loc[i, "prob"],2)))
        j += 1
