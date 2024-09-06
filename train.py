#! /usr/bin/env python3

from typing import List
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict
# visualization
from PIL import Image
# identifying faces
# from mtcnn.mtcnn import MTCNN
# visualizing bounding boxes
import matplotlib.patches as patches
# CNN
import keras
from sklearn.model_selection import train_test_split
# Moving files between directories
import shutil
from shutil import unpack_archive
from subprocess import check_output

dataset_path: str = "./lfw"
# dataset_path = "./lfw/lfw-deepfunneled"

# Define DataFrame type directly
from pandas import DataFrame

# Read CSV files into DataFrames
lfw_allnames: DataFrame = pd.read_csv(dataset_path + "/lfw_allnames.csv")
matchpairsDevTest: DataFrame = pd.read_csv(dataset_path + "/matchpairsDevTest.csv")
matchpairsDevTrain: DataFrame = pd.read_csv(dataset_path + "/matchpairsDevTrain.csv")
mismatchpairsDevTest: DataFrame = pd.read_csv(dataset_path + "/mismatchpairsDevTest.csv")
mismatchpairsDevTrain: DataFrame = pd.read_csv(dataset_path + "/mismatchpairsDevTrain.csv")
pairs: DataFrame = pd.read_csv(dataset_path + "/pairs.csv")

# Tidy pairs data
pairs = pairs.rename(columns={'name': 'name1', 'Unnamed: 3': 'name2'})
matched_pairs: DataFrame = pairs[pairs["name2"].isnull()].drop("name2", axis=1)
mismatched_pairs: DataFrame = pairs[pairs["name2"].notnull()]
people: DataFrame = pd.read_csv(dataset_path + "/people.csv")

# Remove null values
people = people[people.name.notnull()]
peopleDevTest: DataFrame = pd.read_csv(dataset_path + "/peopleDevTest.csv")
peopleDevTrain: DataFrame = pd.read_csv(dataset_path + "/peopleDevTrain.csv")

# Splitting the dataset into train and test

# Shape DataFrame so there is a row per image, matched to relevant jpg file
image_paths: DataFrame = lfw_allnames.loc[lfw_allnames.index.repeat(lfw_allnames['images'])]
image_paths['image_path'] = 1 + image_paths.groupby('name').cumcount()
image_paths['image_path'] = image_paths.image_path.apply(lambda x: '{0:0>4}'.format(x))
image_paths['image_path'] = image_paths.name + "/" + image_paths.name + "_" + image_paths.image_path + ".jpg"
image_paths = image_paths.drop("images", axis=1)

# Ensure image_paths is a DataFrame
if not isinstance(image_paths, pd.DataFrame):
    raise TypeError("image_paths should be a DataFrame")

# Take a random sample: 80% of the data for the test set
lfw_train: DataFrame
lfw_test: DataFrame
lfw_train, lfw_test = train_test_split(image_paths, test_size=0.2)
lfw_train = lfw_train.reset_index(drop=True)  # reset_index with drop=True to avoid adding an extra index column
lfw_test = lfw_test.reset_index(drop=True)    # reset_index with drop=True to avoid adding an extra index column

# Verify that there is a mix of seen and unseen individuals in the test set
print(len(set(lfw_train.name).intersection(set(lfw_test.name))))
print(len(set(lfw_test.name) - set(lfw_train.name)))

# Both comprehensively non-empty - we are ok to proceed.
# N.B. although we don't use this training/test split in the following model, this is the format of the data we
# would use in applying models to the full dataset



# # verify resolution of all images is consistent
# widths = []
# heights = []
# files = image_paths.image_path
# for file in files:
#     path = "../input/lfw-dataset/lfw-deepfunneled/lfw-deepfunneled/" + str(file)
#     im = Image.open(path)
#     widths.append(im.width)
#     heights.append(im.height)
#
# pd.DataFrame({'height':heights,'width':widths}).describe()
#
# # all 250 x 250 resolution



