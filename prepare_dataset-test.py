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

# writer = SummaryWriter()

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

torch.tensor([1,2,4]),shape