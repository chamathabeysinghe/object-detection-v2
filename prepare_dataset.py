import os
import cv2
import json
from multiprocessing import Pool
import itertools
import pandas as pd
# import matplotlib.pyplot as plt

video_path = './raw_data/videos/'
json_path = './raw_data/json/'
dest_dir = '/home/cabe0006/mb20_scratch/chamath/object-detection-v2/dataset/'

test_dir = os.path.join(dest_dir, 'test')
os.makedirs(test_dir, exist_ok=True)
train_dir = os.path.join(dest_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

files = list(filter(lambda x: len(x.split('.'))>1 and x.split('.')[1] == 'json',os.listdir(json_path)))
files = [x.split('.')[0] for x in files]
test_files = ['sample2', 'sample3']
train_files = list(filter(lambda x: x not in test_files, files))

track_first_n_frames = 500


def convert_frames(vid_file, video_index, dest_dir):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    while read_count < track_first_n_frames:
        success, image = capture.read()
        if not success:
            raise ValueError("Could not read first frame. Is the video file valid? ({})".format(vid_file))
        path = os.path.join(dest_dir, '{}.jpg'.format(read_count  + video_index * track_first_n_frames))
        cv2.imwrite(path, image)
        if (read_count % 20 == 0):
          print(read_count)
        read_count += 1


def process_test_vid_file(file, dest_dir=test_dir):
    data_row_arr = []
    index = int(file.split('sample')[1])
    video_file = os.path.join(video_path, '{}.mp4'.format(file))
    json_file = os.path.join(json_path, '{}.json'.format(file))
    with open(json_file) as f:
        data = json.load(f)
    convert_frames(video_file, index, dest_dir)
    for i in range(track_first_n_frames):
        bboxes = data[str(i)]
        file_index = index * track_first_n_frames + i
        for key, value in bboxes.items():
            data_row_arr.append([file_index, value[0], value[1], value[2], value[3]])
    return data_row_arr


def process_train_vid_file(file, dest_dir=train_dir):
    data_row_arr = []
    index = int(file.split('sample')[1])
    video_file = os.path.join(video_path, '{}.mp4'.format(file))
    json_file = os.path.join(json_path, '{}.json'.format(file))
    with open(json_file) as f:
        data = json.load(f)
    convert_frames(video_file, index, dest_dir)
    for i in range(track_first_n_frames):
        bboxes = data[str(i)]
        file_index = index * track_first_n_frames + i
        for key, value in bboxes.items():
            data_row_arr.append([file_index, value[0], value[1], value[2], value[3]])
    return data_row_arr


with Pool(20) as p:
    csv_records = p.map(process_test_vid_file, test_files)
df_values = list(itertools.chain.from_iterable(csv_records))
df_test = pd.DataFrame(df_values, columns=['frame_no', 'x', 'y', 'w', 'h'])
df_test.to_csv(os.path.join(dest_dir, 'test.csv'), index=False)


with Pool(20) as p:
    csv_records = p.map(process_train_vid_file, train_files)
df_values = list(itertools.chain.from_iterable(csv_records))
df_train = pd.DataFrame(df_values, columns=['frame_no', 'x', 'y', 'w', 'h'])
df_train.to_csv(os.path.join(dest_dir, 'train.csv'), index=False)