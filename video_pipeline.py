"""Video pipeline for Vehicle Detection."""
from feature_extraction import generate_features_sliding
from image_pipeline import draw_labeled_bboxes
import pickle
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import cv2

import numpy as np

class VideoPipeline:
    def __init__(self, classifier, features_definition, window_definitions, threshold=2):
        self.heatmap = None
        self.clf = classifier
        self.ft_def = features_definition
        self.win_defs = window_definitions
        self.threshold = threshold
    
    def update(self, img):
        """Handle the next image. Return img with added detected boxes."""
        if self.heatmap is None:
            self.heatmap = np.zeros(img.shape[:2], dtype=np.uint8)
        else:
            self.heatmap[self.heatmap > 0] -= 1
        features, bboxes = generate_features_sliding(img, self.ft_def, self.win_defs)
        pred = self.clf.predict(features)
        for pts1,pts2 in bboxes[pred]:
            self.heatmap[pts1[1]:pts2[1], pts1[0]:pts2[0]] += 1
        
        self.heatmap[self.heatmap <= self.threshold] = 0
        self.heatmap[self.heatmap > self.threshold+2] = self.threshold+2
        
        labels = label(self.heatmap)
        return draw_labeled_bboxes(img, labels)


    
if __name__ == "__main__":
    """Detect cars in test_video.mp4 and project_video.mp4
    
    TODO: Makes this takes argument with input and output."""
    feature_def = pickle.load( open( "feature_def.p", "rb" ) )
    clf = pickle.load( open( "classifier.p", "rb" ) )


    window_definitions = [
        {"size": (64, 64), "overlap": (1/8, 1/8), "x_range": (None, None), "y_range": (400,529)},
        {"size": (96, 96), "overlap": (1/8, 1/8), "x_range": (None, None), "y_range": (400,None)},
    ]
    
    pipeline = VideoPipeline(clf, feature_def, window_definitions)
    output = 'out.mp4'
    clip = VideoFileClip('test_video.mp4', audio=None)
    out_clip = clip.fl_image(pipeline.update)
    out_clip.write_videofile(output, audio=False)


    pipeline = VideoPipeline(clf, feature_def, window_definitions)
    output = 'project_video_out.mp4'
    clip = VideoFileClip('project_video.mp4', audio=None)
    out_clip = clip.fl_image(pipeline.update)
    out_clip.write_videofile(output, audio=False)
