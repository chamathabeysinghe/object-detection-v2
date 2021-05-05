# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# plt.rcParams["figure.figsize"] = (60, 20)

BASE_DIR = '/home/cabe0006/mb20_scratch/chamath/object-detection-v2/'
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
train_csv = pd.read_csv(os.path.join(DATASET_DIR, 'train.csv'))
test_csv = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
TRAIN_ROOT_PATH = os.path.join(DATASET_DIR, 'train/')
TEST_ROOT_PATH = os.path.join(DATASET_DIR, 'test/')
writer = SummaryWriter()
BATCH_SIZE = 4

def get_train_transforms():
    return A.Compose(
        [
            A.RandomSizedCrop(min_max_height=(800, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2,
                                     val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.9),
            ], p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=542, width=1024, p=1.0),
            # A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
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


class Averager:
    def __init__(self, writer, is_train=True, is_coco=False):
        self.current_total = 0.0
        self.loss_box_reg = 0.0
        self.loss_classifier = 0.0
        self.loss_objectness = 0.0
        self.loss_rpn_box_reg = 0.0
        self.iterations = 0.0
        self.writer = writer
        self.is_train = is_train
        self.is_coco = is_coco

    def send(self, value, loss_dict):
        self.current_total += value
        self.loss_box_reg += loss_dict['loss_box_reg'].item()
        self.loss_classifier += loss_dict['loss_classifier'].item()
        self.loss_objectness += loss_dict['loss_objectness'].item()
        self.loss_box_reg += loss_dict['loss_rpn_box_reg'].item()
        self.iterations += 1

    def write_summary(self, epoch):
        print('Saving logs to tensorboard...')
        if self.is_train:
            writer.add_scalar('Train/total_loss', self.current_total, epoch)
            writer.add_scalar('Train/loss_box_reg', self.loss_box_reg, epoch)
            writer.add_scalar('Train/loss_classifier', self.loss_classifier, epoch)
            writer.add_scalar('Train/loss_objectness', self.loss_objectness, epoch)
            writer.add_scalar('Train/loss_rpn_box_reg', self.loss_rpn_box_reg, epoch)
        else:
            writer.add_scalar('Test/total_loss', self.current_total, epoch)
            writer.add_scalar('Test/loss_box_reg', self.loss_box_reg, epoch)
            writer.add_scalar('Test/loss_classifier', self.loss_classifier, epoch)
            writer.add_scalar('Test/loss_objectness', self.loss_objectness, epoch)
            writer.add_scalar('Test/loss_rpn_box_reg', self.loss_rpn_box_reg, epoch)

    def write_coco(self, epoch, m_ap, ap_50, ap_75):
        if self.is_coco:
            print('Saving COCO data to tensorfboard')
            print(m_ap, ap_50, ap_75)
            writer.add_scalar('Test/AP', m_ap, epoch)
            writer.add_scalar('Test/AP50', ap_50, epoch)
            writer.add_scalar('Test/AP75', ap_75, epoch)

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.loss_box_reg = 0.0
        self.loss_classifier = 0.0
        self.loss_objectness = 0.0
        self.loss_rpn_box_reg = 0.0

        self.iterations = 0.0


def collate_fn(batch):
    return tuple(zip(*batch))


def train_step(epoch):
    model.train()
    itr = 1
    loss_hist.reset()
    for images_in, targets_in, img_id in train_loader:
        try:
            images = list(image.to(device) for image in images_in)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets_in]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss_value = loss.item()
            loss_hist.send(loss_value, loss_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Iteration #{itr} loss: {loss_value}")
            itr += 1
        except:
            print("ERROR OCCURRED")
            # print(images_in.shape)
            # print(targets_in.shape)
            print("*************************")
            raise

    loss_hist.write_summary(epoch)
    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss_hist.value


def test_step(epoch):
    model.train()
    itr = 1
    loss_hist_val.reset()
    with torch.no_grad():
        for images, targets, img_id in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss_value = loss.item()
            loss_hist_val.send(loss_value, loss_dict)

            print(f"Iteration #{itr} loss: {loss_value}")
            itr += 1

    loss_hist_val.write_summary(epoch)

    return loss_hist_val.value


def coco_evaluation(epoch):
    # Format and save predictions as a json
    model.eval()
    itr = 1
    json_array = []
    for images, targets, img_ids in val_loader:
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
        print(f"Iteration #{itr}")
        itr += 1
    with open('eval-results.json', 'w') as outfile:
        json.dump(json_array, outfile)

    # load ground truth
    # load results
    cocoDt = cocoGt.loadRes('./eval-results.json')
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    # evaluate coco metrics
    m_ap = cocoEval.stats[0]
    ap_50 = cocoEval.stats[1]
    ap_75 = cocoEval.stats[2]

    # write to tensorboard
    loss_hist_coco.write_coco(epoch, m_ap, ap_50, ap_75)
    return json_array


train_dataset = DatasetRetriever(
    image_ids=train_csv.frame_no.unique(),
    marking=train_csv,
    transforms=get_train_transforms(),
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
annType = 'bbox'
cocoGt = COCO(os.path.join(DATASET_DIR, 'ground_truth-new.json'))
num_classes = 2
EPOCHS = 100
CHECKPOINT_FREQ = 1


model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# if torch.cuda.device_count() > 1:
#   print("Let's use", torch.cuda.device_count(), "GPUs!")
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   model = torch.nn.DataParallel(model)

model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = None


loss_hist = Averager(writer)
loss_hist_val = Averager(writer, is_train=False)
loss_hist_coco = Averager(writer, is_train=False, is_coco=True)

initial_epoch = 0
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if len(os.listdir(CHECKPOINT_DIR)) > 0:
    latest_file_index = max([int(f[11:f.index('.')]) for f in os.listdir(CHECKPOINT_DIR)])
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint-{}.pt'.format(latest_file_index))
    print('********************************************************')
    print('********************************************************')
    print('********************************************************')
    print('********************************************************')
    print('********************************************************')
    print('********************************************************')
    print('********************************************************')
    print('Loading from : {}'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial_epoch = checkpoint['epoch'] + 1


for epoch in range(initial_epoch, EPOCHS):
    print('Training epoch: {}...'.format(epoch))
    train_loss = train_step(epoch)
    print('Evaluating...')
    test_loss = test_step(epoch)
    print(f"Epoch #{epoch} train_loss: {train_loss} test_loss: {test_loss}")
    print('COCO evaluation script...')
    coco_evaluation(epoch)
    if epoch % CHECKPOINT_FREQ == 0:
        print('Saving checkpoint...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(CHECKPOINT_DIR, 'checkpoint-{}.pt'.format(epoch)))
