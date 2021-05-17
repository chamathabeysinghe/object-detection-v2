import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import os
plt.rcParams["figure.figsize"] = (60, 20)
SCORE_THRESHOLD = 0.4
ALLOWED_MISSES = 7
IOU_OVERLAP = 0.1
VIDEO_INDEX = 3


class Point:
    def __init__(self, x, y, w, h, score, frame_index):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame_index = frame_index
        self.parent = None
        self.prev = None
        self.next = None
        self.id = None
        self.score = score
        self.missed_count = 0
        self.new_missed_count = 0
        self.terminated = False

    def increment_missed_count(self):
        self.new_missed_count += 1

    def set_missed_count(self):
        self.missed_count = self.new_missed_count

    def pass_threshold(self, threshold):
        return self.score >= threshold

    def get_box(self):
        return {'x1': self.x, 'y1': self.y, 'x2': self.x + self.w, 'y2': self.y + self.h}

    def print_box_str(self):
        print({'x1': self.x, 'y1': self.y, 'x2': self.x + self.w, 'y2': self.y + self.h, 'score': self.score})

    def set_parent_next(self, parent):
        self.parent = parent

    def set_next(self, point):
        self.next = point

    def set_prev(self, point):
        self.prev = point

    def set_id(self, id):
        self.id = id


track_first_n_frames = 500


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
        out.write((vid_frames[i]).astype(np.uint8))
    out.release()

def get_iou(p_bb1, p_bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    bb1 = p_bb1.get_box()
    bb2 = p_bb2.get_box()

    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_frame_detections(vid_index, frame_index):
    return df.loc[df['image_id'] == vid_index*500+frame_index]


def get_frame_points(frame_index):
    detections = list(map(lambda x: Point(x[2], x[3], x[4], x[5], x[6], frame_index),get_frame_detections(VIDEO_INDEX, frame_index).values))
    detections = list(filter(lambda x: x.pass_threshold(SCORE_THRESHOLD), detections))
    return detections

def draw_box(point):
    if point.id not in colors:
        colors[point.id] = (random.randint(0, 256),random.randint(0, 256),random.randint(0, 256))
    color = colors[point.id]

    cv2.rectangle(frames[point.frame_index],
                  (int(point.x), int(point.y)),
                  (int(point.x + point.w), int(point.y + point.h)),
                  color,6)

    cv2.putText(frames[point.frame_index],
            str(point.id),
            (int(point.x), random.randint(int(point.y) - 40, int(point.y) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            3, color,
            2, cv2.LINE_AA)

def draw_tracks():
    parent_nodes = []
    for node in live_tracks+terminated_tracks:
        if node.parent == None:
            parent_nodes.append(node)
        else:
            parent_nodes.append(node.parent)
    parent_nodes = parent_nodes[::-1]
    for index, node in enumerate(parent_nodes):
        node.set_id(index)
        child_node = node.next
        while child_node!= None:
            child_node.set_id(index)
            child_node = child_node.next

    for node in parent_nodes:
        draw_box(node)
        child_node = node.next
        while child_node != None:
            draw_box(child_node)
            child_node = child_node.next


def calculate_cost(tracks, points):
    costs = []
    for p1 in tracks:
        row = []
        for p2 in points:
            c = get_iou(p1, p2)
            row.append(1 - c)
        costs.append(row)
    return costs


def get_matched_point_index(costs, row_ind, col_ind, track_index, cost_threshold=0.9):
    if track_index not in row_ind.tolist():
        return -1
    col_arr_index = row_ind.tolist().index(track_index)
    point_index = col_ind[col_arr_index]
    match_cost = costs[track_index][point_index]
    if (match_cost < cost_threshold):
        return point_index
    return -1


def matching(live_tracks, current_points, missed_count=0, threshold=0.9):
    matching_tracks = list(filter(lambda node: node.missed_count == missed_count, live_tracks))
    matching_points = list(filter(lambda node: node.prev == None, current_points))
    if len(matching_tracks) == 0:
        return
    if len(matching_points) == 0:
        for track in matching_tracks:
            track.increment_missed_count()
        return

    costs = calculate_cost(matching_tracks, matching_points)
    row_ind, col_ind = linear_sum_assignment(costs)

    for track_index, track in enumerate(matching_tracks):
        point_index = get_matched_point_index(costs, row_ind, col_ind, track_index, threshold)
        if point_index == -1:
            matching_tracks[track_index].increment_missed_count()
            continue
        point = matching_points[point_index]

        point.set_prev(track)
        if (track.parent == None):
            point.parent = track
        else:
            point.parent = track.parent
        track.next = point
        track.new_missed_count = 0


def link_detections(frame_index):
    global live_tracks
    current_points = get_frame_points(frame_index)
    for miss_count in range(0, ALLOWED_MISSES):
        matching(live_tracks, current_points, miss_count, 1 - IOU_OVERLAP)
    new_live_tracks = []
    for index, track in enumerate(live_tracks):
        track.set_missed_count()
        if track.next == None and track.missed_count >= ALLOWED_MISSES:
            terminated_tracks.append(track)
        if track.next == None and track.missed_count < ALLOWED_MISSES:
            new_live_tracks.append(track)
    live_tracks = new_live_tracks + current_points


files = ['sample3.mp4']
os.makedirs('./results', exist_ok=True)

for f in files:
    frames = convert_frames(os.path.join('./videos', f))
    df = pd.read_csv('./detection_data/eval-results-test.csv')
    del df['Unnamed: 0']

    terminated_tracks = []
    live_tracks = get_frame_points(0)
    colors = {}

    for i in range(1, track_first_n_frames):
        link_detections(i)
    draw_tracks()
    print('Writing results to video file...')
    write_file(frames, os.path.join('./results', f))

