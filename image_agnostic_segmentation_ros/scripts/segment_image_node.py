#! /usr/bin/env python

import os, sys
import numpy as np
import cv2
import argparse
import time
from pathlib import Path 
from PIL import Image
from typing import Tuple

import torch 
import torchvision
from torchvision.transforms import transforms

import clip
from clip.model import CLIP

import rospy
import cv_bridge
import sensor_msgs.msg as sensor_msgs
import image_agnostic_segmentation_ros_msg.msg as segmentation_msgs
# from std_msgs.msg import MultiArrayDimension, UInt8MultiArray

path = Path(__file__).parent.parent
sys.path.append(str(path))

from agnostic_segmentation import agnostic_segmentation

class AgnosticSegmentationNode:
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    
    def __init__(self) -> None:
        
        ## initialize segmentation model ##
        self._model_path = rospy.get_param("~model_path")
        self._model = agnostic_segmentation.build_model(self._model_path)
        
        self._bridge = cv_bridge.CvBridge()
        self._frame_rate = rospy.get_param("~frame_rate")
        self._is_publish_segmented_image = rospy.get_param("~is_publish_segmented_image")
        
        self._frame_time = 1.0 / self._frame_rate
        self._before_time = 0
        
        ## setting for CLIP ##
        self._prompts_path = rospy.get_param("~prompts_path")
        self._target_classes_path = rospy.get_param("~target_classes_path")
        self._clip_model, self._clip_preprocess, self._prompts = AgnosticSegmentationNode._build_clip(self._prompts_path)
        self._class_labels = AgnosticSegmentationNode._generate_class_labels(self._target_classes_path)
        
        ## attribute for debug ##
        self._save_dir = None
        self._is_save_masked_image = rospy.get_param("~is_save_masked_image")
        if self._is_save_masked_image: 
            self._save_dir = str(Path(__file__).parent.parent / "cropped_image")  
            self._instance_count = 0
        
        ## Subscriber ##
        self._is_callback_registered = False

        self._subscriber_args = ("/input/image", sensor_msgs.Image, self._image_callback)
        self._subscriber = None
        
        ## Publisher ##
        self._segmented_image_publisher = rospy.Publisher("/output/image/compressed", sensor_msgs.CompressedImage, queue_size=1)

        rospy.loginfo("Ready...")
        
    def main(self) :
        while not rospy.is_shutdown() :
            connection_exists = (self._segmented_image_publisher.get_num_connections() > 0)
            
            if connection_exists and (not self._is_callback_registered):
                self._subscriber = rospy.Subscriber(*self._subscriber_args)
                self._is_callback_registered = True
                rospy.loginfo(f"Register subscriber: {self._subscriber.resolved_name}")

            elif (not connection_exists) and self._is_callback_registered:
                rospy.loginfo(f"Unregister subscriber: {self._subscriber.resolved_name}")
                self._subscriber.unregister()
                self._subscriber = None
                self._is_callback_registered = False

            rospy.sleep(0.5)
    
    def _image_callback(self, msg: sensor_msgs.CompressedImage):
        
        """segment subscribed image by mask rcnn and publish segmented image

        Args:
            msg (sensor_msgs.CompressedImage): Image msg from HSR
        """        
        
        now_time = time.time() 
        
        if (now_time - self._before_time) < self._frame_time:
            return

        self._before_time = now_time
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        predictions = self._model(image)
        np_batched_image = self._crop(image, predictions)
        
        if len(np_batched_image) >= 1:
            predict_class_labels = self._zero_shot_classify(np_batched_image)
            predictions["instances"].pred_classes = predict_class_labels
            
        if len(predictions["instances"].to("cpu")) > 0:
            segmented_image = agnostic_segmentation.draw_segmented_image(image, predictions, self._class_labels)
            segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
            self._publish_segmented_image(segmented_image)
            
        else: 
            self._publish_segmented_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        elapsed_time = time.time() - now_time
        rospy.loginfo("elapsed_time:{0}".format(elapsed_time) + "[sec]")
        
    def _publish_segmented_image(self, segmented_image: np.ndarray) -> None :
        """publish segmented image

        Args:
            segmented_image (np.ndarray): segmented image
        """        
        
        if not self._is_publish_segmented_image:
            return
        
        segmented_image_msg = self._bridge.cv2_to_compressed_imgmsg(segmented_image)
        segmented_image_msg.header.stamp = rospy.Time.now() 
        self._segmented_image_publisher.publish(segmented_image_msg)
        
    ## crop the region of interest ##
    def _crop(self, image: np.ndarray, predictions: dict) -> np.ndarray :
        instances = predictions["instances"].to("cpu")
        np_batched_images = np.zeros((len(instances), 224, 224, 3)) # B, H, W, C
        for i in range(len(instances)) :
            self._instance_count += 1
            pred_masks = instances.pred_masks[i].cpu().detach().numpy()
            v, u = np.where(pred_masks == True)
            
            # mask rgb image 
            masked_rgb_image = image.copy() 
            masked_rgb_image[pred_masks==False] = np.array([0, 0, 0])
            masked_rgb_image = masked_rgb_image[np.min(v):np.max(v), np.min(u):np.max(u), :]
            masked_rgb_image = cv2.resize(masked_rgb_image, (224, 224))
            np_batched_images[i] = masked_rgb_image
            
            # NOTE: if you want to debug, save cropped iamges
            if self._is_save_masked_image:
                pil_masked_image = Image.fromarray(masked_rgb_image)
                pil_masked_image.save(self._save_dir + f"/{self._instance_count}.jpg")
                
        return np_batched_images
    
    ## zero shot classify by CLIP ##
    def _zero_shot_classify(self, np_batched_image: np.ndarray) -> torch.Tensor:
        
        """method for zero shot image classification by CLIP image encoder and text encoder

        Args:
            np_batched_image (np.ndarray): batch image (B, H, W, C)

        Returns:
            _type_: _description_
        """        
        tensor_batched_image = torch.from_numpy(np_batched_image.astype(np.float32)) # B, H, W, C
        tensor_batched_image = tensor_batched_image.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        tensor_batched_image = self._clip_preprocess(tensor_batched_image).to(AgnosticSegmentationNode.device)
        text = self._prompts.to(AgnosticSegmentationNode.device)
        with torch.no_grad():
            
            image_features = self._clip_model.encode_image(tensor_batched_image)
            text_features = self._clip_model.encode_text(text)
            
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim = True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        value, class_label_index = similarity.topk(1)
        if len(np_batched_image) > 1:
            class_label_index = class_label_index.squeeze()
        else: class_label_index = class_label_index[0]        

        #! debug
        # rospy.loginfo(class_label_index)
        # rospy.loginfo(f"Shape input tensor: {tensor_batched_image.shape}")
        # rospy.loginfo(f"Normalized image features shape: {image_features.shape}")
        # rospy.loginfo(f"similarity value: {value} \n top1 indices: {index}")
        
        return class_label_index
        
        
    @staticmethod
    def _build_clip(prompts_path: str) -> Tuple[CLIP, transforms.Compose, torch.Tensor] :
       
        """static method for build CLIP, image preprocess function and prompts

        Args:
            prompts_path (str): path to prompts.txt
        """        
        
        ## load defined prompts ##
        prompts = []
        with open(prompts_path) as f :
            for line in f :
                prompts.append(line.strip("\n"))
                
        ## setting related to CLIP ##
        device = AgnosticSegmentationNode.device
        model, _ = clip.load("ViT-B/32", device=device)
        preprocess = transforms.Compose([transforms.Resize(model.visual.input_resolution, interpolation=transforms.InterpolationMode.BICUBIC), 
                                         transforms.CenterCrop(224), 
                                         transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        text = clip.tokenize(prompts)
        
        return model, preprocess, text
    
    @staticmethod
    def _generate_class_labels(target_classes_path: str) -> list:
        
        """static method to generate class labels
        
        Args:
            target_classes_path (str): path to classes.txt
        """        
        
        ## load class labels ##
        class_labels = []
        with open(target_classes_path) as f:
            for line in f:
                class_labels.append(line.strip("\n"))
                
        return class_labels
        
        
if __name__ == "__main__" :
    rospy.init_node("segmentation_ros")
    AgnosticSegmentationNode().main()
    