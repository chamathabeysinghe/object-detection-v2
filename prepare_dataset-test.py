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
json_path = './raw_data/json/'


files = list(filter(lambda x: len(x.split('.'))>1 and x.split('.')[1] == 'json',os.listdir(json_path)))
files = [x.split('.')[0] for x in files]
test_files = ['sample2', 'sample3']
train_files = list(filter(lambda x: x not in test_files, files))

print(files)


# read_csvsync -av ? progress -e ?ssh -i gpu-instance-oregon.pem? /home/cabe0006/mb20_scratch/chamath/object-detection-v2/dataset/ ubuntu@ec2-34-220-192-22.us-west-2.compute.amazonaws.com:/home/ubuntu/dataset
#
# rsync -av ? progress -e "ssh -i gpu-instance-oregon.pem" /home/cabe0006/mb20_scratch/chamath/object-detection-v2/dataset/ ubuntu@ec2-34-220-192-22.us-west-2.compute.amazonaws.com:/home/ubuntu/dataset
#
# rsync -av -progress -e "ssh -i gpu-instance-oregon.pem" /home/cabe0006/mb20_scratch/chamath/object-detection-v2/dataset/ ubuntu@ec2-34-220-192-22.us-west-2.compute.amazonaws.com:/home/ubuntu/dataset
#
