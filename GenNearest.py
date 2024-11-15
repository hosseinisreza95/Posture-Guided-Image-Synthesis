
import numpy as np
import cv2
import os
import pickle
import sys
import math

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton



# class GenNeirest:
#     """ class that Generate a new image from videoSke from a new skeleton posture
#        Fonc generator(Skeleton)->Image
#        Neirest neighbor method: it selects the image in videoSke that has the skeleton closest to the skeleton
#     """
#     def _init_(self, videoSkeTgt):
#         self.videoSkeletonTarget = videoSkeTgt  # VideoSkeleton object containing target frames and skeletons

#     def generate(self, ske):
#         min_distance = float('inf')
#         closest_frame = None

#         for frame_skeleton in self.videoSkeletonTarget.ske:
#             frame = Skeleton(frame_skeleton.ske)  # Creating a Skeleton object from frame_skeleton's skeletal data
#             distance = ske.distance(frame)
            
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_frame = frame_skeleton
        
#         # Get the associated image path if closest_frame is found
#         if closest_frame is not None:
#             closest_image_idx = list(self.videoSkeletonTarget.ske).index(closest_frame)
#             closest_image_path = self.videoSkeletonTarget.imagePath(closest_image_idx)
#             closest_image = cv2.imread(closest_image_path)  # Read image using OpenCV
#         else:
#             closest_image = np.ones((64, 64, 3), dtype=np.uint8)  # Placeholder if no match

#         return closest_image



class GenNeirest:
    """ class that Generate a new image from videoSke from a new skeleton posture
       Fonc generator(Skeleton)->Image
       Neirest neighbor method: it selects the image in videoSke that has the skeleton closest to the skeleton
    """
    def __init__(self, videoSkeTgt):
        self.videoSkeletonTarget = videoSkeTgt  # VideoSkeleton object containing target frames and skeletons

    def generate(self, target_skeleton):
        """ 
        Generates an image from the VideoSkeleton based on a target skeleton posture.
        Finds the image in videoSkeTgt with the skeleton closest to the target_skeleton.
        """
        closest_image = None
        min_distance = float('inf')
        
        for i in range(self.videoSkeletonTarget.skeCount()):
            current_skeleton = self.videoSkeletonTarget.ske[i]
            distance = target_skeleton.distance(current_skeleton)

            if distance < min_distance:
                min_distance = distance
                closest_image = self.videoSkeletonTarget.readImage(i)

        return closest_image if closest_image is not None else np.ones((64, 64, 3), dtype=np.uint8)

