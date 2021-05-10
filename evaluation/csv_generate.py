import json
import os
import random

import albumentations as A
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# plt.rcParams["figure.figsize"] = (60, 20)

BASE_DIR = '/home/cabe0006/mb20_scratch/chamath/object-detection-v2/'
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints_batch')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
train_csv = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
test_csv = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
TRAIN_ROOT_PATH = os.path.join(DATASET_DIR, 'train/')
TEST_ROOT_PATH = os.path.join(DATASET_DIR, 'test/')
BATCH_SIZE = 4
DEST_DIR = './evaluation_files'
os.makedirs(DEST_DIR, exist_ok=True)

def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=542, width=1024, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        if self.test or random.random() > 0.5:
            image, boxes = self.load_image_and_boxes(index)
        else:
            image, boxes = self.load_image_and_boxes(index)
        #             image, boxes = self.load_cutmix_image_and_boxes(index)

        # there is only one class
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        # print('*************************************************')
        # print(labels.shape)
        # print(boxes.shape)
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        if self.transforms:

            for i in range(10):
                try:
                    sample = self.transforms(**{
                        'image': image,
                        'bboxes': target['boxes'],
                        'labels': labels
                    })
                    if len(sample['bboxes']) > 0:
                        image = sample['image']
                        target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                        target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [0, 1, 2, 3]]  # yxyx: be warning
                        break
                except:
                    print('Augementation error:')
                    print(target['image_id'])
                    print(target['boxes'])

        labels = torch.ones((target['boxes'].shape[0],), dtype=torch.int64)
        target['labels'] = labels

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        # image = cv2.imread(f'{TRAIN_ROOT_PATH}/{image_id}.jpg', cv2.IMREAD_COLOR)
        # print(TRAIN_ROOT_PATH + '{}'.format(image_id) + '.jpg')
        if (self.test):
            image = cv2.imread(TEST_ROOT_PATH + '{}'.format(image_id) + '.jpg', cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(TRAIN_ROOT_PATH + '{}'.format(image_id) + '.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        records = self.marking[self.marking['frame_no'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[
            np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes


def collate_fn(batch):
    return tuple(zip(*batch))


def predict_step(data_loader, is_test=True):
    itr = 1
    json_array = []
    csv_array = [] #image_id, category_id, x, y, w, h, score
    for images, targets, img_ids in data_loader:
        images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for index, prediction in enumerate(outputs):
            boxes = prediction['boxes'].tolist()
            scores = prediction['scores'].tolist()
            img_id = img_ids[index]
            for i, box in enumerate(boxes):
                obj = {
                    "image_id": int(img_id),
                    "category_id": 1,
                    "bbox": [box[0] * 4, box[1] * 4, (box[2] - box[0]) * 4, (box[3] - box[1]) * 4],
                    "score": scores[i]
                }
                json_array.append(obj)
                csv_array.append([
                    int(img_id),
                    1,
                    box[0] * 4, box[1] * 4, (box[2] - box[0]) * 4, (box[3] - box[1]) * 4,
                    scores[i]
                ])
        print(f"Iteration #{itr}")
        itr += 1
    eval_fie_name = 'eval-results-test' if is_test else 'eval-results-train'
    with open(os.path.join(DEST_DIR, eval_fie_name+'.json'), 'w') as outfile:
        json.dump(json_array, outfile)
    df = pd.DataFrame(csv_array, columns=['image_id', 'category_id', 'x', 'y', 'w', 'h', 'score'])
    df.to_csv(os.path.join(DEST_DIR, eval_fie_name+'.csv'))


train_dataset = DatasetRetriever(
    image_ids=train_csv.frame_no.unique(),
    marking=train_csv,
    transforms=get_valid_transforms(),
    test=False,
)
validation_dataset = DatasetRetriever(
    image_ids=test_csv.frame_no.unique(),
    marking=test_csv,
    transforms=get_valid_transforms(),
    test=True,
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=RandomSampler(train_dataset),
    pin_memory=False,
    drop_last=True,
    num_workers=6,
    collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    num_workers=6,
    shuffle=False,
    sampler=SequentialSampler(validation_dataset),
    pin_memory=False,
    collate_fn=collate_fn,
)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Cuda is available: {}'.format(torch.cuda.is_available()))
cpu_device = torch.device("cpu")
num_classes = 2


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)


initial_epoch = 0
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


if len(os.listdir(CHECKPOINT_DIR)) > 0:
    latest_file_index = max([int(f[11:f.index('.')]) for f in os.listdir(CHECKPOINT_DIR)])
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint-{}.pt'.format(latest_file_index))
    print('Loading from : {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])


model.eval()
predict_step(train_loader, is_test=False)
predict_step(val_loader, is_test=True)
