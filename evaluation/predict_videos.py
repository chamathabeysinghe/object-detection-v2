import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import pandas as pd


plt.rcParams["figure.figsize"] = (60, 20)
track_first_n_frames = 500
CHECKPOINT_DIR = '/home/cabe0006/mb20_scratch/chamath/object-detection-v2/checkpoints'

# BASE_DIR = '/Users/cabe0006/Projects/monash/Datasets'
# DATASET = 'ant_dataset_small'

BASE_DIR = '/home/cabe0006/mb20_scratch/chamath/data'
DATASET = 'ant_dataset'


TAGGED = 'tagged'
VID_DIR = os.path.join(BASE_DIR, DATASET, TAGGED)
DEST_DIR = os.path.join(BASE_DIR, f'{DATASET}_faster_rcnn_predictions', TAGGED)
CSV_DIR = os.path.join(BASE_DIR, f'{DATASET}_faster_rcnn_predictions_csv', TAGGED)

os.makedirs(DEST_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


def convert_frames(vid_file):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    frames = []
    while True:
        success, image = capture.read()
        if not success:
            break
            # raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
        frames.append(image)

        if (read_count % 20 == 0):
            print(read_count)
        read_count += 1
    return frames


def write_file(vid_frames, file_name):
    height, width, layers = vid_frames[0].shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(file_name, fourcc, 6.0, (width, height))

    for i in range(len(vid_frames)):
        out.write((vid_frames[i] * 255).astype(np.uint8))
    out.release()


def show_prediction(sample, boxes, scores, file_name, file_index):
    result_boxes = []
    for index, box in enumerate(boxes):
        if scores[index] > 0.4:
            result_boxes.append([f'{file_name}_{file_index:6d}', box[0], box[1], box[2], box[3]])
            cv2.rectangle(sample,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (0.8, 0, 0), thickness=1)
    return result_boxes
    # plt.imshow(sample)
    # plt.show()


def predict_batch(images, file_name, file_indexes, csv_results):
    images = list(img.to(device) for img in images)
    outputs = model(images)
    outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
    frames = []
    for i in range(len(images)):
        file_index = file_indexes[i]
        sample = images[i].permute(1, 2, 0).cpu().numpy()
        boxes = outputs[i]['boxes'].cpu().detach().numpy().astype(np.int32)
        scores = outputs[i]['scores'].cpu().detach().numpy()
        csv_results += show_prediction(sample, boxes, scores, file_name, file_index)
        frames.append(sample)
    return frames


def process_file(file_name):
    video_path = os.path.join(VID_DIR, file_name+'.mp4')
    frames = convert_frames(video_path)
    resized_frames = [cv2.resize(f, (1024, 542)) for f in frames]
    bbox_frames = []
    csv_results = []
    for i in range(2, len(resized_frames) + 2, 2):
        print(i)
        model_input = (np.asarray(resized_frames[i - 2:i]) / 255.0).astype(np.float32)
        model_input = torch.from_numpy(model_input).to(device).permute(0, 3, 1, 2)
        bbox_frames += predict_batch(model_input, file_name, range(i-2, i), csv_results)
    write_file(bbox_frames, os.path.join(DEST_DIR, file_name + '.mp4'))
    df = pd.DataFrame(csv_results, columns=['image_id', 'x', 'y', 'w', 'h'])
    df.to_csv(os.path.join(CSV_DIR, file_name + '.csv'), index=False)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint-{}.pt'.format(28))
print('Loading from : {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


# files = ['sample2.mp4']
# files = ['masked_sample2_2']
# files = ['sample4', 'sample2', 'sample3', 'sample14', 'sample13', 'sample12',
#          'sample10', 'sample9', 'sample8', 'sample7', 'sample1', 'masked_sample2_2', 'masked_sample2']
# files = [f'sample{x}' for x in range(50, 61)]
files = os.listdir(os.path.join(VID_DIR))
files = list(filter(lambda s: int(s.split('_')[1][:-4]) == 0, files))
for file in files:
    process_file(file.split('.')[0])

