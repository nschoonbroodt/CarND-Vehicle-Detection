"""Helper functions to extract features.

This module contains functions to extract features from a single 64x64 image,
from a data set collection of 64x64 pictures, or from sliding windows over a
larger image.

Note: for now, only works with one cell per block in hog transform
"""

import cv2
import numpy as np

from skimage.feature import hog
import matplotlib.pyplot as plt

def generate_features_sliding(img, features_definition, windows_definitions):
    """Generate features from sliding windows over an image

    Args:
        img (np.array, (X,Y,3)): The picture, RGB values (0 to 255)
        features_definition (dict): Features definition
        windows_definition (list of dict): definition of sliding windows

    Returns: features, bboxes
        features: np.array (X,Y): the features extracted from the image
            X depends on windows_definitions and image.shape
            Y depends on the features_definition
        bboxes: list of coordinates of all the sliding windows corners
            
    Example:
        Extracting features includes spatial binning and HOG
        >>> img = ...
        >>> features_definition = {
                "spatial_bin": (16,16),
                "hog": {"orientations": 16, "px_per_cell": 8, "cell_per_block": 1, "channel": "ALL", "color_space": "YCrCb",},
                "classifier_size": (64,64),
            }
        >>> windows_definitions = [
                {"size": (64, 64), "overlap": (1/8, 1/8), "x_range": (None, None), "y_range": (400,529)},
                {"size": (96, 96), "overlap": (1/8, 1/8), "x_range": (None, None), "y_range": (400,None)},
            ]
        >>> features,bboxes = generate_features_sliding(img, features_definition, windows_definitions)

    """
    bin_feat = []
    hist_feat = []
    hog_feat = []
    bboxes = []
    for win_def in windows_definitions:
        
        # Crop the image to remove unused parts
        x_lim, y_lim = set_range(win_def, img.shape)
        cropped_img = img[y_lim[0]:y_lim[1],x_lim[0]:x_lim[1]]
        clf_size = features_definition["classifier_size"]
        
        for x in range(x_lim[0], x_lim[1]-win_def["size"][0]+1, int(win_def["size"][0]*win_def["overlap"][0])):
            for y in range(y_lim[0], y_lim[1]-win_def["size"][1]+1, int(win_def["size"][1]*win_def["overlap"][1])):
                bboxes.append(((x,y), (x+win_def["size"][0],y+win_def["size"][1])))

        if "spatial_bin" in features_definition:
            # get the scale of the resize
            bin_size = features_definition["spatial_bin"]
            scale = [target/orig for orig,target in zip(win_def["size"],bin_size)]
            if scale[0] != 1 or scale[1] != 1:
                spatial_bin_img = cv2.resize(cropped_img, (0,0), fx=scale[0], fy=scale[1])
            else:
                spatial_bin_img = cropped_img
            
            # offset from one window to the next
            step = [int(bin_size[0]*win_def["overlap"][0]),
                    int(bin_size[1]*win_def["overlap"][1])]
            
            # slice the array
            for x in range(0, spatial_bin_img.shape[1]-bin_size[0]+1, step[0]):
                for y in range(0, spatial_bin_img.shape[0]-bin_size[1]+1, step[1]):
                    bin_feat.append(spatial_bin_img[y:y+bin_size[1],x:x+bin_size[0],:].ravel())
                    
        # histogram and hog share the same resize
        if "histo" in features_definition or "hog" in features_definition:
            scale = [target/orig for orig,target in zip(win_def["size"],clf_size)]
            if scale[0] != 1 or scale[1] != 1:
                histo_img = cv2.resize(cropped_img, (0,0), fx=scale[0], fy=scale[1])
            else:
                histo_img = cropped_img
                
            if "histo" in features_definition:
                step = [int(clf_size[0]*win_def["overlap"][0]),
                        int(clf_size[1]*win_def["overlap"][1])]
                for x in range(0, histo_img.shape[1]-clf_size[0]+1, step[0]):
                    for y in range(0, histo_img.shape[0]-clf_size[1]+1, step[1]):
                        hist_feat.append(
                            color_hist(histo_img[y:y+clf_size[1],x:x+clf_size[0],:],
                                       nbins=features_definition["histo"]))
                                        
            if "hog" in features_definition:
                hog_params = features_definition["hog"]
                if "color_space" in hog_params and hog_params["color_space"] != 'RGB':
                    transform = eval('cv2.COLOR_RGB2' + hog_params["color_space"])
                    hog_image = cv2.cvtColor(histo_img, transform)
                else:
                    hog_image = histo_img
                
                orient=hog_params["orientations"]
                px_per_c=hog_params["px_per_cell"]
                c_per_bl=hog_params["cell_per_block"]
                assert(c_per_bl==1)
                
                size = [x//px_per_c for x in clf_size]
                step = [int(size[0]*win_def["overlap"][0]),
                        int(size[1]*win_def["overlap"][1])]
                
                if hog_params["channel"] == "ALL":
                    ch0 = hog(hog_image[:,:,0],
                              orientations=orient, 
                              pixels_per_cell=(px_per_c, px_per_c),
                              cells_per_block=(c_per_bl, c_per_bl), 
                              feature_vector=False)
                    ch1 = hog(hog_image[:,:,1],
                              orientations=orient, 
                              pixels_per_cell=(px_per_c, px_per_c),
                              cells_per_block=(c_per_bl, c_per_bl), 
                              feature_vector=False)
                    ch2 = hog(hog_image[:,:,2],
                              orientations=orient, 
                              pixels_per_cell=(px_per_c, px_per_c),
                              cells_per_block=(c_per_bl, c_per_bl), 
                              feature_vector=False)
                    feat = []
                    for x in range(0, ch0.shape[1]-size[0]+1, step[0]):
                        for y in range(0, ch0.shape[0]-size[1]+1, step[1]):
                            dummy=[]
                            dummy.append(ch0[y:y+size[1],x:x+size[0],:].ravel())
                            dummy.append(ch1[y:y+size[1],x:x+size[0],:].ravel())
                            dummy.append(ch2[y:y+size[1],x:x+size[0],:].ravel())
                            hog_feat.append(np.hstack(dummy))
                else:
                    ch0 = hog(hog_image[:,:,hog_params["channel"]],
                              orientations=orient, 
                              pixels_per_cell=(px_per_c, px_per_c),
                              cells_per_block=(c_per_bl, c_per_bl), 
                              feature_vector=False)
                    feat = []
                    for x in range(0, ch0.shape[1]-size[0]+1, step[0]):
                        for y in range(0, ch0.shape[0]-size[1]+1, step[1]):
                            hog_feat.append(ch0[y:y+size[1],x:x+size[0],:].ravel())

    return np.hstack(list(filter(lambda x: len(x)>0, [bin_feat, hist_feat, hog_feat]))), np.array(bboxes)
    
def generate_features_from_dataset(dataset, features_definition):
    """Extract features from the given dataset.

    The given dataset must be a numpy array containing 64x64x3 RGB pictures

    Args:
        dataset (np.array, Nx64x64x3): The dataset, RGB values (0 to 255)
        features_definition (dict): Features definition

    Returns:
        np.array (N,M): the features extracted the dataset pictures
            M depends on the features_definition
            
    Example:
        Extracting features includes spatial binning, histogram and HOG
        >>> dataset = ...
        >>> features_definition = {
                "spatial_bin": (16,16),
                "histo": 32,
                    "hog": {"orientations": 16, "px_per_cell": 8, "cell_per_block": 1, "channel": "ALL", "color_space": "YCrCb",},
                "classifier_size": (64,64),
            }
        >>> features = generate_features_from_dataset(dataset, features_definition)

    """
    
    # get the size to initialise a numpy array
    f = generate_features(dataset[0,:,:,:], features_definition)
    
    features = np.zeros((dataset.shape[0], f.shape[0]))
    
    for i,x in enumerate(dataset):
        features[i,:] = generate_features(x, features_definition=features_definition)
    return features

def generate_features(img, features_definition):
    """Extract features from one instance.

    The given instance must be a 64x64x3 RGB pictures

    Args:
        img (np.array, 64x64x3): The picture, RGB values (0 to 255)
        features_definition (dict): Features definition

    Returns:
        np.array (M): the features extracted the picture
            M depends on the features_definition
            
    Example:
        Extracting features includes spatial binning, histogram and HOG
        >>> features_definition = {
                "spatial_bin": (16,16),
                "histo": 32,
                    "hog": {"orientations": 16, "px_per_cell": 8, "cell_per_block": 1, "channel": "ALL", "color_space": "YCrCb"},
                "classifier_size": (64,64),
            }
        >>> features = generate_features(img, features_definition)

    """
    features = []
    
    if "spatial_bin" in features_definition:
        features.append(bin_spatial(img, size=features_definition["spatial_bin"]))
            
    if "histo" in features_definition:
        features.append(color_hist(img, nbins=features_definition["histo"]))

    if "hog" in features_definition:
        # Convert image desired colorspace
        hog_params = features_definition["hog"]
        if "color_space" in hog_params and hog_params["color_space"] != 'RGB':
            transform = eval('cv2.COLOR_RGB2' + hog_params["color_space"])
            image = cv2.cvtColor(img, transform)
        else:
            image = img
            
        features.append(hog_features(image,
                                     channel=hog_params["channel"],
                                     orient=hog_params["orientations"],
                                     pixels_per_cell=hog_params["px_per_cell"],
                                     cell_per_block=hog_params["cell_per_block"]))
    
    features = np.concatenate(features)
    return features

def hog_features(img, orient=8, pixels_per_cell=8, cell_per_block=1, channel="ALL"):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    See https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients for more
    informations on HOG

    Args:
        img : (M, N, C) array
            Input image
        orientations : int
            Number of orientation bins.
        pixels_per_cell : int
            Size (in pixels) of a cell.
        cells_per_block : int
            Number of cells in each block.
        
    Returns
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.
    """
    if channel == 'ALL':
        features = []
        for channel in range(img.shape[2]):
            features.append(hog(img[:,:,channel], orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                visualise=False, feature_vector=True))
        features = np.ravel(features)        
    else:
        features = hog(img[:,:,channel], orientations=orient, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                      visualise=False, feature_vector=True)
    return features

def bin_spatial(img, size=(16, 16)):        
    """Combining clusters of pixels into single pixels.
    
    Args:
        img (np.array, MxNxC): original image
        size: target dimension
        
    Returns:
        np.array(np.prod(size) * C): the feature vector
    
    """
    return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """Compute a color histogram of the picture
    
    Args:
        img (np.array, MxNxC): original image
        nbins: number of buckets for each color channel
        bins_range: the color min and max values
        
    Returns:
        np.array(nbins * C): the feature vector
    
    """
    feat = []
    for i in range(img.shape[2]):
        hist = np.histogram(img[:,:,i], bins=nbins, range=bins_range)
        feat.append(hist[0])
    
    return np.concatenate(feat).ravel()

def set_range(window_definition, shape):
    """Helper function restricting the range to image shape, handling None"""
    win_size = window_definition["size"]
    
    x_range = list(window_definition["x_range"])
    if x_range[0] is None or x_range[0] < 0:
        x_range[0] = 0
    if x_range[1] is None or x_range[1] > shape[1]:
        x_range[1] = shape[1]
        
    y_range = list(window_definition["y_range"])
    if y_range[0] is None or y_range[0] < 0:
        y_range[0] = 0
    if y_range[1] is None or y_range[1] > shape[0]:
        y_range[1] = shape[0]
    
    return x_range, y_range    
