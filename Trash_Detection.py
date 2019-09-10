import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib
import skimage
import cv2
import matplotlib.pyplot as plt
class CigButtsConfig(Config):

    NAME = "cig_butts"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 60 + 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 5000

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet101'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = CigButtsConfig()
config.display()

class_names = ['BG','Aluminium foil','Battery','Aluminium blister pack','Carded blister pack','Other plastic bottle','Plastic drink bottle','Glass bottle','Plastic bottle cap','Metal bottle cap','Broken glass','Food Can','Aerosol','Drink can','Toilet tube','Other carton','Egg carton','Drink carton','Corrugated carton','Meal carton','Pizza box','Paper cup','Disposable plastic cup','Foam cup','Glass cup','Other plastic cup','Food waste','Glass jar','Plastic lid','Metal lid','Other plastic','Magazine paper','Tissues','Wrapping paper','Normal paper','Paper bag','Plastified paper bag','Plastic Film','Six pack rings','Garbage bag','Other plastic wrapper','Single-use carrier bag','Polypropylene bag','Crisp packet','Spread tub','Tupperware','Disposable food container','Foam food container','Other plastic container','Plastic glooves','Plastic utensils','Pop tab','Rope & strings','Scrap metal','Shoe','Squeezable tube','Plastic straw','Paper straw','Styrofoam piece','Unlabeled litter','Cigarette']

class InferenceConfig(CigButtsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85
    

inference_config = InferenceConfig()

MODEL_DIR = os.path.join("./", "logs")
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir="./logs")

model_path = ("./logs/mask_rcnn_cig_butts_0008.h5")

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


real_test_dir = './test'
image_paths = []
output_fol = "output_img"
if not os.path.exists(output_fol):
    os.makedirs(output_fol)
for filename in os.listdir(real_test_dir):
    if os.path.splitext(filename)[1].lower() in ['.png', '.jpg', '.jpeg','.JPG']:
        image_paths.append(os.path.join(real_test_dir, filename))

for image_path in image_paths:
    img = skimage.io.imread(image_path)
    img_arr = np.array(img)
    results = model.detect([img_arr], verbose=1)
    r = results[0]
    print("scores",r['scores'])
    img = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
                                class_names, r['scores'],figsize=(5,5))
    img.savefig(f"./{output_fol}/{os.path.basename(image_path)}")
    # print(f"./{output_fol}/{os.path.basename(image_path)}")
    # cv2.imshow("test", pre_img)
    # cv2.waitKey(0)
    # cv2.imwrite(f"./{output_fol}/{os.path.basename(image_path)}",pre_img)