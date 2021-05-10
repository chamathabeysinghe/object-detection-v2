import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

plt.rcParams["figure.figsize"] = (60, 20)
track_first_n_frames = 500
CHECKPOINT_DIR = './checkpoints'
VID_DIR = './raw_videos'
DEST_DIR = './result_videos'
os.makedirs(DEST_DIR, exist_ok=True)


def convert_frames(vid_file):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    frames = []
    while read_count < track_first_n_frames:
        success, image = capture.read()
        if not success:
            raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
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


def show_prediction(sample, boxes, scores):
    for index, box in enumerate(boxes):
        if scores[index] > 0.4:
            cv2.rectangle(sample,
                          (box[0], box[1]),
                          (box[2], box[3]),
                          (220, 0, 0), thickness=1)
    plt.imshow(sample)
    plt.show()


def predict_batch(images):
    images = list(img.to(device) for img in images)
    outputs = model(images)
    outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
    frames = []
    for i in range(len(images)):
        sample = images[i].permute(1, 2, 0).cpu().numpy()
        boxes = outputs[i]['boxes'].cpu().detach().numpy().astype(np.int32)
        scores = outputs[i]['scores'].cpu().detach().numpy()
        show_prediction(sample, boxes, scores)
        frames.append(sample)
    return frames


def process_file(file_name):
    video_path = os.path.join(VID_DIR, file_name)
    frames = convert_frames(video_path)
    resized_frames = [cv2.resize(f, (1024, 542)) for f in frames]
    bbox_frames = []
    for i in range(2, len(resized_frames) + 2, 2):
        print(i)
        model_input = (np.asarray(resized_frames[i - 2:i]) / 255.0).astype(np.float32)
        model_input = torch.from_numpy(model_input).to(device).permute(0, 3, 1, 2)
        bbox_frames += predict_batch(model_input)
    write_file(bbox_frames, os.path.join(DEST_DIR, file_name))


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
num_classes = 2
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint-{}.pt'.format(26))
print('Loading from : {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()


files = ['sample2.mp4']
for file in files:
    process_file(file)

