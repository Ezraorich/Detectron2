from detectron2.utils.logger import setup_logger 
setup_logger()

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

config_file_path ='COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
checkpoint_url = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'

output_dir ='D:/XrayCOVIDproject/COVIDGR_2.0/output/instance_segmentation'

num_classes = 1

device  = 'cuda'

train_dataset_name = 'X_train'
train_images_path = 'D:/XrayCOVIDproject/COVIDGR_2.0/train_mask'
train_json_annot_path = 'D:/XrayCOVIDproject/COVIDGR_2.0/train.json'

test_dataset_name = 'X_test'
test_images_path = 'D:/XrayCOVIDproject/COVIDGR_2.0/test_mask'
test_json_annot_path = 'D:/XrayCOVIDproject/COVIDGR_2.0/test.json'

cfg_save_path = 'D:/XrayCOVIDproject/COVIDGR_2.0/IS_cfg.pickle'
 
register_coco_instances(name = train_dataset_name, metadata ={},
                       json_file = train_json_annot_path,
                       image_root = train_images_path)

register_coco_instances(name = test_dataset_name, metadata ={},
                       json_file = test_json_annot_path,
                       image_root = test_images_path)
                       
                       
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode


import random
import cv2
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def plot_samples(dataset_name, n=1):
    
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s['file_name'])
        v = Visualizer(img[:,:,::-1], metadata = dataset_custom_metadata, scale = 0.5)
        v  = v.draw_dataset_dict(s)
        plt.figure(figsize = (15, 20))
        plt.imshow(v.get_image())
        plt.show()
        
plot_samples(dataset_name  =train_dataset_name, n=2)    
def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name,test_dataset_name,
                 num_classes, device, output_dir):
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN =(train_dataset_name,)
    cfg.DATASETS.TEST =(test_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH =2
    cfg.SOLVER.BASE_LR =0.001
    cfg.SOLVER.MAX_ITER =400
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg
    
def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name,
                       test_dataset_name, num_classes, device, output_dir)
    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol = pickle.HIGHEST_PROTOCOL)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume = False)
    trainer.train()    
main()
import torch
torch.cuda.empty_cache()  
from detectron2.engine import DefaultPredictor

import os
import pickle

def on_image(image_path, predictor):
    im  = cv2.imread(image_path)
    outputs = predictor(im)
    v = Visualizer(im[:,:, ::-1], metadata={}, scale =0.5, instance_mode =ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    
    plt.figure(figsize = (14,10))
    plt.imshow(v.get_image())
    plt.show()
    
cfg_save_path  = 'D:/XrayCOVIDproject/COVIDGR_2.0/IS_cfg.pickle'

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)
    
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST  =0.5

predictor =DefaultPredictor(cfg)

image_path = 'D:/XrayCOVIDproject/COVIDGR_2.0/test_mask/P0.jpg'
on_image(image_path, predictor)


def on_video(videoPath, predictor):
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened()== False):
        print('Error opening file...')
        return
    
    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata = {}, instance_mode = ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions['instances'].to('cpu'))
        
        cv2.imshow('Result', output.get_image()[:,:,::-1])
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        (success, image) = cap.read()
video_path = 'path.mp4'
on_video(video_path, predictor)
        
#original code Authors: https://www.youtube.com/watch?v=ffTURA0JM1Q
## from theCodingBug
                       
