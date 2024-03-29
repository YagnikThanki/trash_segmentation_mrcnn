import cv2
import numpy as np
import os
import sys

# from mrcnn import utils
# from mrcnn import model as modellib

# ROOT_DIR = os.path.abspath("./")
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# sys.path.append(os.path.join(ROOT_DIR,"samples/coco/"))
# import coco
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "model.h5")
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


# class InferenceConfig(coco.CocoConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     NUM_CLASSES = 1 + 2
#     IMAGE_MIN_DIM = 256
#     IMAGE_MAX_DIM = 256
# config = InferenceConfig()
# config.display()

# model = modellib.MaskRCNN(
    # mode="inference", model_dir=MODEL_DIR, config=config
# )
# model.load_weights(COCO_MODEL_PATH, by_name=True)
# class_names = [
#     'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
#     'bus', 'train', 'truck', 'boat', 'traffic light',
#     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
#     'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
#     'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
#     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite', 'baseball bat', 'baseball glove', 'skateboard',
#     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
#     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
#     'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
#     'teddy bear', 'hair drier', 'toothbrush'
# ]
class_names = ['BG','Aluminium foil','Battery','Aluminium blister pack','Carded blister pack','Other plastic bottle','Plastic drink bottle','Glass bottle','Plastic bottle cap','Metal bottle cap','Broken glass','Food Can','Aerosol','Drink can','Toilet tube','Other carton','Egg carton','Drink carton','Corrugated carton','Meal carton','Pizza box','Paper cup','Disposable plastic cup','Foam cup','Glass cup','Other plastic cup','Food waste','Glass jar','Plastic lid','Metal lid','Other plastic','Magazine paper','Tissues','Wrapping paper','Normal paper','Paper bag','Plastified paper bag','Plastic Film','Six pack rings','Garbage bag','Other plastic wrapper','Single-use carrier bag','Polypropylene bag','Crisp packet','Spread tub','Tupperware','Disposable food container','Foam food container','Other plastic container','Plastic glooves','Plastic utensils','Pop tab','Rope & strings','Scrap metal','Shoe','Squeezable tube','Plastic straw','Paper straw','Styrofoam piece','Unlabeled litter','Cigarette']

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    # print("no of potholes in frame :",n_instances)
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]
        mask = mask.astype(np.uint8)*255  #convert mask into 0,255 format
        # cv2.imshow("cont", mask)
        # cv2.waitKey(0)
        contour,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #find contour to calculate pixel area
        # contour_area = cv2.contourArea(contour[0])
        # print("Pothole area {} :".format(i),contour_area)  
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(    
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


# if __name__ == '__main__':
#     """
#         test everything
#     """

#     capture = cv2.VideoCapture(0)

#     # these 2 lines can be removed if you dont have a 1080p camera.
#     capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#     while True:
#         ret, frame = capture.read()
#         results = model.detect([frame], verbose=0)
#         r = results[0]
#         frame = display_instances(
#             frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores']
#         )
#         cv2.imshow('frame', frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     capture.release()
#     cv2.destroyAllWindows()