"""Create and save the classier to identify vehicles from non vehicles."""

from feature_extraction import generate_features_from_dataset

from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle


shape = (64,64,3)

def read_images(vehicle_folder, non_vehicle_folder, valid=.1, test=.1):
    """Read the images manage data formating.
    
    Args:
        vehicle_folder: str
            Path to image to class as vehicle. Pictures must be .png and in
            subfolders
        non_vehicle_folder: str
            Same as above, for non vehicles
        valid: float
            portion of data kept for validation
        valid: float
            portion of data kept for training
    
    Returns: X_train, y_train, X_valid, y_valid, X_test, y_test
        X_ : np.array (N,64,64,3) the image data
        t_ : np.array (N,) the image label (True for vehicle)
    """
    X_train = np.zeros((0, *shape), dtype=np.uint8)
    X_valid = np.zeros((0, *shape), dtype=np.uint8)
    X_test = np.zeros((0, *shape), dtype=np.uint8)
    
    for folder in glob.glob(vehicle_folder + '/*'):
        a,b,c = split_folder(folder)
        X_train = np.concatenate((X_train, a))
        X_valid = np.concatenate((X_valid, b))
        X_test = np.concatenate((X_test, c))
    idx_true = [len(X_train),len(X_valid),len(X_test)]
    
    for folder in glob.glob(non_vehicle_folder + '/*'):
        a,b,c = split_folder(folder)
        X_train = np.concatenate((X_train, a))
        X_valid = np.concatenate((X_valid, b))
        X_test = np.concatenate((X_test, c))
        
    y_train = np.full(len(X_train), True)
    y_valid = np.full(len(X_valid), True)
    y_test = np.full(len(X_test), True)
    
    y_train[idx_true[0]:] = False
    y_valid[idx_true[1]:] = False
    y_test[idx_true[2]:] = False
    
    X_train,y_train = shuffle(X_train,y_train)
    X_valid,y_valid = shuffle(X_valid,y_valid)
    X_test,y_test = shuffle(X_test,y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def split_folder(dirname, valid=.1, test=.1):
    """Split images into train, valid and test sets for one directory.
    
    Images must be in png format.
    
    Args:
        dirname (str): the folder where all files are stored
        valid (float): proportion of the validation set
        test (float): proportion of the test set
    
    Returns: X_train, X_valid, X_test:
        (J,K,L,C) np.array, K,L,C are the picture dimension and channel
        
    TODO: Handle other format with an optional arg
    """
    images = glob.glob(dirname + '/*.png')
    nb = len(images)
    idx_test = int(nb*(1-test))
    idx_valid = int(nb*(1-test-valid))
    
    X_train = np.zeros((idx_valid, *shape,), dtype=np.uint8)
    X_valid = np.zeros((idx_test-idx_valid, *shape,), dtype=np.uint8)
    X_test = np.zeros((nb-idx_test, *shape,), dtype=np.uint8)
    
    for i in range(0, idx_valid):
        X_train[i,:,:,:] = np.uint8(plt.imread(images[i])*255)
    for i in range(idx_valid, idx_test):
        X_valid[i-idx_valid,:,:,:] = np.uint8(plt.imread(images[i])*255)
    for i in range(idx_test, nb):
        X_test[i-idx_test,:,:,:] = np.uint8(plt.imread(images[i])*255)
        
    return X_train, X_valid, X_test
    
if __name__ == "__main__":
    """Create and train a LinearSVC classifier.
    
    Save the classifier and features definitions in 'feature_def.p' and
    'classifier.p'
    The classifier is a pipeline including a feature scaler.
    """
    features_definition = {
    "spatial_bin": (16,16),
    "hog": {"orientations": 24, "px_per_cell": 8, "cell_per_block": 1, "channel": "ALL", "color_space": "HLS"},
    "classifier_size": (64,64),
    }
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = \
        read_images('data/vehicles', 'data/non-vehicles')
    
    feature_train = generate_features_from_dataset(X_train, features_definition)
    feature_valid = generate_features_from_dataset(X_valid, features_definition)
    
    classifier = make_pipeline(StandardScaler(),
                               LinearSVC())
    
    classifier.fit(feature_train, y_train)
    
    print('Training Accuracy of SVC = ', round(classifier.score(feature_train, y_train), 4))
    print('Validation Accuracy of SVC = ', round(classifier.score(feature_valid, y_valid), 4))
    
    pickle.dump(features_definition, open('feature_def.p', 'wb'))
    pickle.dump(classifier, open('classifier.p', 'wb'))
