import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import os
import math
from pykalman import KalmanFilter
import draw_utils
from multiprocessing.pool import ThreadPool
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
        self.bbox_center_chain = None
        self.bbox_size_chain = None
        self.frames_chain = None
        self.measured_bbox_center_chain = None

    def get_parent(self):
        if (self.parent == None):
            return self
        return self.parent

    def get_observations_chain(self):
        # TODO Complete the array of positions
        current_node = self
        chain = []
        while True:
            chain.append([current_node.x, current_node.y])
            prev_node = current_node.prev
            if prev_node == None:
                break
            skip_count = 1
            while current_node.frame_index - skip_count > prev_node.frame_index:
                chain.append([-1, -1])
                skip_count += 1
            current_node = prev_node
        chain = chain[::-1]
        return chain

    def increment_missed_count(self):
        self.new_missed_count += 1

    def set_missed_count(self):
        self.missed_count = self.new_missed_count

    def pass_threshold(self, threshold):
        return self.score >= threshold

    def get_box(self):
        return {'x1': self.x, 'y1': self.y, 'x2': self.x + self.w, 'y2': self.y + self.h}

    def __build_kf_chain(self, bbox_center_chain):
        position_index = -1
        for i in range(len(bbox_center_chain) - 1):
            if bbox_center_chain[i][0] > -1 and bbox_center_chain[i+1][0] > -1:
                position_index = i
                break

        if position_index == -1:
            return bbox_center_chain

        NonMeasured = bbox_center_chain[:position_index]
        Measured = bbox_center_chain[position_index:]
        Measured = np.asarray(Measured)
        # while True:
        #     if Measured[0, 0] == -1.:
        #         Measured = np.delete(Measured, 0, 0)
        #     else:
        #         break
        # while True:
        #     if Measured[1, 0] == -1.:
        #         Measured = np.delete(Measured, 1, 0)
        #     else:
        #         break

        MarkedMeasure = np.ma.masked_less(Measured, 0)
        Transition_Matrix = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        Observation_Matrix = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]])
        xinit = MarkedMeasure[0, 0]
        yinit = MarkedMeasure[0, 1]
        vxinit = MarkedMeasure[1, 0] - MarkedMeasure[0, 0]
        vyinit = MarkedMeasure[1, 1] - MarkedMeasure[0, 1]
        initstate = [xinit, yinit, vxinit, vyinit]
        initcovariance = 1.0e-3 * np.eye(4)
        x = 1
        transistionCov = np.asarray(
            [[x / 3, x / 2, 0, 0], [x / 2, x, 0, 0], [0, 0, x / 3, x / 2], [0, 0, x / 2, x]])  # 1.0e-1*np.eye(4)
        observationCov = 1.0e-1 * np.eye(2)
        kf = KalmanFilter(transition_matrices=Transition_Matrix,
                          observation_matrices=Observation_Matrix,
                          initial_state_mean=initstate,
                          initial_state_covariance=initcovariance,
                          transition_covariance=transistionCov,
                          observation_covariance=observationCov)
        (filtered_state_means, filtered_state_covariances) = kf.filter(MarkedMeasure)
        filtered_state_means = filtered_state_means.tolist()
        return NonMeasured + filtered_state_means

    def get_bbox_chain(self):
        # if self.parent is None:
        bbox_center_chain = []
        bbox_size_chain = []
        frames_chain = []
        current_node = self.get_parent()
        while current_node is not None:
            bbox_center_chain.append([current_node.x, current_node.y])
            bbox_size_chain.append([current_node.w, current_node.h])
            frames_chain.append(current_node.frame_index)
            next_node = current_node.next
            if next_node is None:
                skip_count = current_node.missed_count
            else:
                skip_count = next_node.frame_index - current_node.frame_index
            for skip_index in range(1, skip_count):
                bbox_center_chain.append([-1, -1])
                bbox_size_chain.append([180, 180])
                frames_chain.append(current_node.frame_index + skip_index)
            current_node = next_node

        self.bbox_center_chain = bbox_center_chain
        self.bbox_size_chain = bbox_size_chain
        self.frames_chain = frames_chain
        self.measured_bbox_center_chain = self.__build_kf_chain(bbox_center_chain)
        return self.bbox_center_chain, self.bbox_size_chain,  self.frames_chain, self.measured_bbox_center_chain

    def get_predict_box(self):
        Measured = self.get_observations_chain()
        if (len(Measured) < 2):
            return self.get_box()
        Measured += [[-1, -1] for _ in range(self.missed_count)]
        Measured.append([-1, -1])
        Measured = np.asarray(Measured)
        while True:
            if Measured[0, 0] == -1.:
                Measured = np.delete(Measured, 0, 0)
            else:
                break
        while True:
            if Measured[1, 0] == -1.:
                Measured = np.delete(Measured, 1, 0)
            else:
                break
        MarkedMeasure = np.ma.masked_less(Measured, 0)
        Transition_Matrix = np.asarray([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
        Observation_Matrix = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]])
        xinit = MarkedMeasure[0, 0]
        yinit = MarkedMeasure[0, 1]
        vxinit = MarkedMeasure[1, 0] - MarkedMeasure[0, 0]
        vyinit = MarkedMeasure[1, 1] - MarkedMeasure[0, 1]
        initstate = [xinit, yinit, vxinit, vyinit]
        initcovariance = 1.0e-3 * np.eye(4)
        x = 1
        transistionCov = np.asarray(
            [[x / 3, x / 2, 0, 0], [x / 2, x, 0, 0], [0, 0, x / 3, x / 2], [0, 0, x / 2, x]])  # 1.0e-1*np.eye(4)
        observationCov = 1.0e-1 * np.eye(2)
        kf = KalmanFilter(transition_matrices=Transition_Matrix,
                          observation_matrices=Observation_Matrix,
                          initial_state_mean=initstate,
                          initial_state_covariance=initcovariance,
                          transition_covariance=transistionCov,
                          observation_covariance=observationCov)
        (filtered_state_means, filtered_state_covariances) = kf.filter(MarkedMeasure)
        filtered_state_means = filtered_state_means.tolist()
        new_x = filtered_state_means[-1][0]
        new_y = filtered_state_means[-1][1]
        if (math.isnan(new_x) or math.isnan(new_y)):
            return self.get_box()

        return {'x1': new_x, 'y1': new_y, 'x2': new_x + self.w, 'y2': new_y + self.h}

    def print_box_str(self):
        print({'x1': self.x, 'y1': self.y, 'x2': self.x + self.w, 'y2': self.y + self.h, 'score': self.score})

    #     def set_parent_next(self, parent):
    #         self.parent = parent

    def set_next(self, point):
        self.next = point

    def set_prev(self, point):
        self.prev = point

    def set_id(self, id):
        self.id = id

capture_len = 500
def convert_frames(vid_file):
    import cv2
    capture = cv2.VideoCapture(vid_file)
    # capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
    read_count = 0
    print("Converting video file: {}".format(vid_file))
    frames = []
    while read_count < capture_len:
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
    bb1 = p_bb1.get_predict_box()
    bb2 = p_bb2.get_predict_box()

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
    return df.loc[df['image_id'] == vid_index * 500 + frame_index]


def draw_box(point):
    if point.id not in colors:
        colors[point.id] = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    color = colors[point.id]

    cv2.rectangle(frames[point.frame_index],
                  (int(point.x), int(point.y)),
                  (int(point.x + point.w), int(point.y + point.h)),
                  color, 6)

    cv2.putText(frames[point.frame_index],
                str(point.id),
                (int(point.x), random.randint(int(point.y) - 40, int(point.y) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                3, color,
                2, cv2.LINE_AA)


def draw_box_1(index, bbox_center_chain, bbox_size_chain, frames_chain, measured_bbox_center_chain, color, id):
    frame_index = frames_chain[index]
    frame = frames[frame_index]
    bbox = bbox_center_chain[index]
    predicted_bbox = measured_bbox_center_chain[index]
    bbox_size = bbox_size_chain[index]
    dotted = False
    if bbox[0] == -1 or bbox[1] == -1:
        dotted = True
        bbox = predicted_bbox

    if bbox[0] == -1 or bbox[1] == -1:
        return

    if dotted:
        draw_utils.drawrect(frame,
                            (int(bbox[0]), int(bbox[1])),
                            (int(bbox[0] + bbox_size[0]), int(bbox[1] + bbox_size[1])),
                            color, 6)
    else:
        cv2.rectangle(frame,
                      (int(bbox[0]), int(bbox[1])),
                      (int(bbox[0] + bbox_size[0]), int(bbox[1] + bbox_size[1])),
                      color, 6)
    cv2.putText(frame,
                str(id),
                (int(bbox[0]), random.randint(int(bbox[1]) - 40, int(bbox[1]) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                3, color,
                2, cv2.LINE_AA)


def draw_tail(index, bbox_center_chain, bbox_size_chain, frames_chain, measured_bbox_center_chain, color):

    frame_index = frames_chain[index]
    frame = frames[frame_index]
    # bbox = bbox_center_chain[index]
    # predicted_bbox = measured_bbox_center_chain[index]
    # bbox_size = bbox_size_chain[index]

    tail_length = min(10, index)
    tail_positions = []

    for t in range(index - tail_length, index):
        bbox = bbox_center_chain[t]
        predicted_bbox = measured_bbox_center_chain[t]
        if bbox[0] == -1 or bbox[1] == -1:
            bbox = predicted_bbox
        if bbox[0] == -1 or bbox[1] == -1:
            continue
        box_size = bbox_size_chain[t]
        tail_positions.append([int(bbox[0] + box_size[0]/2), int(bbox[1] + box_size[1]/2)])
    points = np.array(tail_positions)
    cv2.polylines(frame, np.int32([points]), isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

def draw_parent_node_track(node):
    if node.id not in colors:
        colors[node.id] = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    color = colors[node.id]
    bbox_center_chain, bbox_size_chain, frames_chain, measured_bbox_center_chain = node.get_bbox_chain()

    for i in range(len(frames_chain)):
        draw_box_1(i,  bbox_center_chain, bbox_size_chain, frames_chain, measured_bbox_center_chain, color, node.id)
        draw_tail(i,  bbox_center_chain, bbox_size_chain, frames_chain, measured_bbox_center_chain, color)


def draw_tracks():
    parent_nodes = []
    for node in live_tracks + terminated_tracks:
        if node.parent == None:
            parent_nodes.append(node)
        else:
            parent_nodes.append(node.parent)
    parent_nodes = parent_nodes[::-1]
    for index, node in enumerate(parent_nodes):
        node.set_id(index)
        child_node = node.next
        while child_node != None:
            child_node.set_id(index)
            child_node = child_node.next

    for node in parent_nodes:
        draw_parent_node_track(node)
    # for node in parent_nodes:
    #     draw_box(node)
    #     child_node = node.next
    #     while child_node != None:
    #         draw_box(child_node)
    #         child_node = child_node.next


def get_frame_points(frame_index):
    detections = list(map(lambda x: Point(x[2], x[3], x[4], x[5], x[6], frame_index),
                          get_frame_detections(VIDEO_INDEX, frame_index).values))
    detections = list(filter(lambda x: x.pass_threshold(SCORE_THRESHOLD), detections))
    return detections


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


track_first_n_frames = 500
start_index = 0
files = ['sample2.mp4', 'sample3.mp4']
# files = ['sample1.mp4', 'sample4.mp4','sample7.mp4','sample8.mp4','sample9.mp4','sample10.mp4','sample12.mp4','sample13.mp4','sample14.mp4',]

os.makedirs('/home/cabe0006/mb20_scratch/chamath/object-detection-v2/tracker/results_kalman2', exist_ok=True)


for f in files:
    print('File: {}'.format(f))
    VIDEO_INDEX = int(f[6: -4])
    frames = convert_frames(os.path.join('/home/cabe0006/mb20_scratch/chamath/object-detection-v2/tracker/videos', f))
    df = pd.read_csv('./detection_data/eval-results-test.csv')
    del df['Unnamed: 0']

    terminated_tracks = []
    live_tracks = get_frame_points(start_index)
    colors = {}

    for i in range(start_index + 1, start_index + track_first_n_frames):
        print(i)
        link_detections(i)
    print('Drawing tracks...')
    draw_tracks()
    print('Writing results to video file...')
    write_file(frames, os.path.join('/home/cabe0006/mb20_scratch/chamath/object-detection-v2/tracker/results_kalman2', f))
    print('Done writing file: {}'.format(f))

# p = ThreadPool(2)
# p.map(hello, range(3))

