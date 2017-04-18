"""Detects cars in a single image."""

import pickle

from feature_extraction import generate_features_sliding
import matplotlib.pyplot as plt

import glob
import numpy as np
import cv2
from scipy.ndimage.measurements import label

import os



def find_and_draw(img, feature_def, window_definitions, clf):
    """Identify positive sliding windows according to the classifier.
    
    Args:
        img: a RGB picture
        feature_def: feature extraction window_definitions
        window_definitions: sliding windows definitions
        clf: a classifier accepting features generated with feature_def
        
    Returns:
        the image with bounding boxes added: green means positive sliding
        windows, red boxes indicated at least two overlapping detections
    """
    features, bboxes = generate_features_sliding(img, feature_def, window_definitions)
    pred = clf.predict(features)
    
    summed = np.zeros(img.shape[:2])
    for pt1,pt2 in bboxes[pred]:
        summed[pt1[1]:pt2[1], pt1[0]:pt2[0]] += 1
        cv2.rectangle(img, tuple(pt1), tuple(pt2), (0,255,0), thickness=2)
    summed[summed<=1] = 0
    labels = label(summed)
        
    return draw_labeled_bboxes(img, labels)
    
def draw_labeled_bboxes(img, labels):
    """Draw labels founded with skimage label function."""
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 6)
    # Return the image
    return img

if __name__ == "__main__":
    """Detect cars in image from test_images and save results in output_images."""
    feature_def = pickle.load( open( "feature_def.p", "rb" ) )
    clf = pickle.load( open( "classifier.p", "rb" ) )
    
    window_definitions = [
    {"size": (64, 64), "overlap": (1/8, 1/8), "x_range": (None, None), "y_range": (400,529)},
    {"size": (96, 96), "overlap": (1/8, 1/8), "x_range": (None, None), "y_range": (400,None)},
    ]

    images = glob.glob('test_images/*.jpg')
    for fname in images:
        img = plt.imread(fname)
        img = find_and_draw(img, feature_def, window_definitions, clf)
        plt.imsave('output_images/' + os.path.basename(fname), img)

    
