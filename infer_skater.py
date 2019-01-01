from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from Mask_RCNN.samples.coco import coco
from video_reader import VideoReader
import cv2
import numpy as np
import matplotlib.pyplot as plt


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

COCO_LOGS = 'logs/'
COCO_MODEL = 'models/mask_rcnn_coco.h5'

model = modellib.MaskRCNN(mode="inference", model_dir=COCO_LOGS, config=config)
model.load_weights(COCO_MODEL, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

vid_name = 'vid02'
game_watcher = VideoReader(vid_name)


def draw_masks(image, boxes, masks, class_ids, scores):
    N = boxes.shape[0]
    colors = visualize.random_colors(N)
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        y1, x1, y2, x2 = boxes[i]

        # Label
        class_id = class_ids[i]
        score = scores[i]
        label = class_names[class_id]

        #ax.text(x1, y1 + 8, label,
        #        color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, color)

    return masked_image


while True:
    frame = game_watcher.read_frame(show_frame=False, color_format='bgr')
    #frame = cv2.imread('Mask_RCNN/images/12283150_12d37e6389_z.jpg', 1)
    if frame is None:
        break
    else:
        cv2.imshow('Input', frame)
        in_key = cv2.waitKey()
        if in_key == ord('q'):
            break
        elif in_key == ord('e'):
            skater_inference = model.detect([frame], verbose=1)[0]
            skater_inference_masked = draw_masks(frame,
                                                 skater_inference['rois'],
                                                 skater_inference['masks'],
                                                 skater_inference['class_ids'],
                                                 skater_inference['scores'])
            skater_inference_masked = skater_inference_masked.astype(np.uint8)

            cv2.imshow('Change', skater_inference_masked)
