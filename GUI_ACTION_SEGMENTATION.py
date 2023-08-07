from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import cv2
import numpy as np
import os.path as osp
import time
from Camera_SDK.get_frame_camera import *
# from PoseEstimation.Hrnet import Hrnet
# from PoseEstimation.Yolov7 import Yolov7
from dtw import dtw
from PoseEstimation_v2.pose_detect import Hrnet, Yolov7
from Action_Segmentation import ST_GCN
# from SCL import ST_GCN as Temporal_ST_GCN
from SCL_version2 import ST_GCN as Temporal_ST_GCN
import torch

import torch.nn.functional as F
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os
from Node_classification import Node_classification
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


class Transform:
    def __init__(self):
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.theta = 0.0
        self.translate_x = 0.0
        self.translate_y = 0.0

    def rotate(self, theta):
        self.theta = theta

    def scale(self, scale_x, scale_y):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def translate(self, translate_x, translate_y):
        self.translate_x = translate_x
        self.translate_y = translate_y

    def apply(self, points):
        theta_rad = np.radians(self.theta)
        rotation_matrix = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                                    [np.sin(theta_rad), np.cos(theta_rad)]])
        scaled_points = np.dot(points, np.diag([self.scale_x, self.scale_y]))
        translated_points = scaled_points + np.array([self.translate_x, self.translate_y])
        rotated_points = np.dot(translated_points, rotation_matrix)
        return rotated_points

    def error_function(self, params, points_A, points_B):
        scale_x, scale_y, theta, translate_x, translate_y = params
        self.scale(scale_x, scale_y)
        self.rotate(theta)
        self.translate(translate_x, translate_y)
        transformed_points = self.apply(points_A)
        # error = 0
        # for i in range(6,17):
        #     error += np.sum((transformed_points[i] - points_B[i]) ** 2)
        error = np.sum((transformed_points - points_B) ** 2)
        return error

    def find_parameters(self, points_A, points_B):
        initial_params = [self.scale_x, self.scale_y, self.theta, self.translate_x, self.translate_y]
        result = minimize(self.error_function, initial_params, args=(points_A, points_B))
        optimal_params = result.x
        self.scale_x, self.scale_y, self.theta, self.translate_x, self.translate_y = optimal_params


class PoseComparison:
    def __init__(self, skeleton1, skeleton2):
        self.skeleton1 = skeleton1
        self.skeleton2 = skeleton2
        self.connections = [[6,8,10], [5, 7, 9], [8,6,12], [11,5,7], [6,12,14], [5,11,13],[12, 14,16], [11,13,15]]

    def compute_angles(self, skeleton):
        angles = []
        for connection in self.connections:
            point1 = skeleton[connection[0]-5]
            point2 = skeleton[connection[1]-5]
            point3 = skeleton[connection[2]-5]

            vector1 = point1 - point2
            vector2 = point3 - point2

            dot_product = np.dot(vector1, vector2)
            norm_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

            angle = np.arccos(dot_product / norm_product)*180/np.pi
            angles.append(angle)

        return angles

    def compare_poses(self, threshold=0.1):
        angles1 = self.compute_angles(self.skeleton1)
        angles2 = self.compute_angles(self.skeleton2)

        different_angles = []
        for i, angle1 in enumerate(angles1):
            angle2 = angles2[i]
            diff = np.abs(angle1 - angle2)
            if diff > threshold:
                different_angles.append((self.connections[i], diff))

        return different_angles


def align(query_feats, candidate_feats, use_dtw=False):
    """Align videos based on nearest neighbor or dynamic time warping."""
    if use_dtw:
        _, _, _, path = dtw(query_feats, candidate_feats, dist='cosine')
        _, uix = np.unique(path[0], return_index=True)
        nns = path[1][uix]
        return nns
    else:
        dists = cdist(query_feats, candidate_feats, 'sqeuclidean')
        nns = np.argmin(dists, axis=1)
        return nns


def resize_image(im, new_shape=(640, 480), color=(0, 0, 0), auto=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im



class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    fps_display_signal = pyqtSignal(float)
    def __init__(self, path, record=True):
        super().__init__()
        self._run_flag = True
        self.camera = VideoStream(path)
        self.record = record
        self.close = False

    def run(self):
        self.camera.start()
        if self.record:
            self.camera.record()
        while self._run_flag:
            frame = self.camera.read()
            self.change_pixmap_signal.emit(frame)
            self.fps_display_signal.emit(self.camera.fps)
            cv2.waitKey(1)
            if self.close:
                break
        self.camera.stop()

    def stop(self):
        self.threadactive = False
        self.wait()

class Pose_detect_thread(QThread):
    progressing = pyqtSignal(int)
    pose_results = pyqtSignal(list)
    finished = pyqtSignal(float)

    def __init__(self, Yolov7, Hrnet, video):
        super(Pose_detect_thread, self).__init__()
        self.Yolov7 = Yolov7
        self.Hrnet = Hrnet
        self.video = video
        self.time_process = 0

    def run(self):
        start_time = time.time()
        cnt = 0
        pose_results = []
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flag, image = self.video.read()
        while flag:

            image = resize_image(image)

            detections = self.Yolov7.inference(image)
            pose_res = self.Hrnet.inference_from_bbox(image, detections)

            cnt += 1
            self.progressing.emit(cnt)
            pose_results.append(pose_res)
            flag, image = self.video.read()
        time_process = time.time() - start_time
        self.pose_results.emit(pose_results)
        self.finished.emit(time_process)


def uniformly_sample(num_frames, clip_len):
    """Uniformly sample indices for training clips.

    Args:
        num_frames (int): The number of frames.
        clip_len (int): The length of the clip.
    """
    if num_frames < clip_len:
        start = np.random.randint(0, num_frames)
        inds = np.arange(start, start + clip_len)
    elif clip_len <= num_frames < 2 * clip_len:
        basic = np.arange(clip_len)
        inds = np.random.choice(
            clip_len + 1, num_frames - clip_len, replace=False)
        offset = np.zeros(clip_len + 1, dtype=np.int64)
        offset[inds] = 1
        offset = np.cumsum(offset)
        inds = basic + offset[:-1]
    else:
        bids = np.array(
            [i * num_frames // clip_len for i in range(clip_len + 1)])
        bsize = np.diff(bids)
        bst = bids[:clip_len]
        offset = np.random.randint(bsize)
        inds = bst + offset
    return inds


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

def get_segments(
    frame_wise_label,
    id2class_map,
    bg_class: str = "background"):
    """
    Args:
        frame-wise label: frame-wise prediction or ground truth. 1D numpy array
    Return:
        segment-label array: list (excluding background class)
        start index list
        end index list
    """

    labels = []
    starts = []
    ends = []

    frame_wise_label = [
        id2class_map[frame_wise_label[i]] for i in range(len(frame_wise_label))
    ]

    # get class, start index and end index of segments
    # background class is excluded
    last_label = frame_wise_label[0]
    if frame_wise_label[0] != bg_class:
        labels.append(frame_wise_label[0])

        starts.append(0)

    for i in range(len(frame_wise_label)):
        # if action labels change
        if frame_wise_label[i] != last_label:
            # if label change from one class to another class
            # it's an action starting point
            if frame_wise_label[i] != bg_class:
                labels.append(frame_wise_label[i])
                starts.append(i)

            # if label change from background to a class
            # it's not an action end point.
            if last_label != bg_class:
                ends.append(i)

            # update last label
            last_label = frame_wise_label[i]

    if last_label != bg_class:
        ends.append(i)
    return labels, starts, ends

class My_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('GUI.ui', self)

        self.camera = None
        self.video = None
        self.pose_results = None
        self.action_pred = None
        self.save_dir = 'C:/Users/Vas/Desktop/Video Record/'
        # self.model_yolov7 = Yolov7(engine_path='PoseEstimation/yolov7-tiny-nms.trt')
        # self.model_hrnet = Hrnet(engine_path='PoseEstimation/HR_net48.trt')
        self.model_yolov7 = Yolov7(engine_path='PoseEstimation_v2/rtmdet-nano')
        self.model_hrnet = Hrnet(engine_path='PoseEstimation_v2/rtmpose-m')
        self.name = 0
        # self.action_model = TransformerModel(n_classes=4)
        self.action_model = ST_GCN(in_channels=2, n_classes=4)
        self.action_model.load_state_dict(torch.load('checkpoints/model_best_51.pth'))
        self.action_model.to('cuda')
        self.action_model.eval()
        self.node_classification_model = Node_classification()
        self.node_classification_model.load_state_dict(torch.load('checkpoints/model_error_detection_best5.pth'))
        self.node_classification_model.to('cuda')
        self.node_classification_model.eval()

        self.tem_st_gcn = Temporal_ST_GCN(in_channels=2, embedding_size=128)
        self.tem_st_gcn.load_state_dict(torch.load('checkpoints/model_70.pth'))
        self.tem_st_gcn.eval()
        self.tem_st_gcn.to('cuda')
        self.tem_st_gcn.protection = False

        self.start_frame = 0
        self.top_frame = 0
        self.impact_frame = 0
        self.end_frame = 0
        self.node_label = None
        self.pose_trainner_results = None
        self.nns_play_trainner = None
        self.nns_trainner_play = None
        self.quit = QAction("Quit", self)
        self.quit.triggered.connect(self.closeEvent)

        # Thread for pose estimation

        self.thread = Pose_detect_thread(Yolov7=self.model_yolov7, Hrnet=self.model_hrnet, video=None)

        self.thread.progressing.connect(self.progressing_bar)
        self.thread.pose_results.connect(self.receive_pose_result)
        self.thread.finished.connect(self.print_time_pose_detect)

        self.thread_trainner = Pose_detect_thread(Yolov7=self.model_yolov7, Hrnet=self.model_hrnet, video=None)
        self.thread_trainner.progressing.connect(self.progressing_bar2)
        self.thread_trainner.pose_results.connect(self.receive_pose_result2)

        self.skeleton_transform = Transform()


        self.btn_open_camera.clicked.connect(self.connect_camera)
        self.btn_close_camera.clicked.connect(self.disconnect_camera)
        self.btn_start_record.clicked.connect(self.start_record)
        self.btn_stop_record.clicked.connect(self.stop_record)
        self.btn_load_video1.clicked.connect(self.load_video_from_camera)
        self.btn_load_video2.clicked.connect(self.load_video_from_file)
        self.slider_frame_no.valueChanged.connect(self.set_frame_no_slider)
        self.slider_frame_no_2.valueChanged.connect(self.set_frame_no_slider2)
        self.btn_sample.clicked.connect(self.sample_process_video)
        self.btn_detect_pose.clicked.connect(self.pose_detect)
        self.btn_detect_pose_2.clicked.connect(self.pose_detect2)
        self.btn_run_action_model.clicked.connect(self.action_detect)

        self.btn_start_frame.clicked.connect(self.show_start_frame)
        self.btn_top_frame.clicked.connect(self.show_top_frame)
        self.btn_impact_frame.clicked.connect(self.show_impact_frame)
        self.btn_finish_frame.clicked.connect(self.show_end_frame)
        self.btn_back_swing.clicked.connect(self.show_back_swing)
        self.btn_down_swing.clicked.connect(self.show_down_swing)
        self.btn_fthrough.clicked.connect(self.show_fthrough_swing)
        self.btn_error_detection.clicked.connect(self.show_error)
        self.btn_load_trainner.clicked.connect(self.load_video_trainner)
        self.btn_temporal_alignment.clicked.connect(self.temporal_alignment)

        self.canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.gridLayout.addWidget(self.canvas)
        self.canvas.axes.set_yticks(range(4), ['Other Action', 'BackSwing', 'DownSwing', 'F-Through'])
        self.canvas.axes.set_xlabel('Frame #')
        self.canvas.draw()

    def temporal_alignment(self):
        if (self.pose_results is not None) and (self.pose_trainner_results is not None):
            kp1 = []
            for pose in self.pose_results:
                kp1.append(pose['keypoints'])
            kp1 = np.array(kp1)

            h1, w1 = (640, 480)
            x1 = kp1
            x1[:, :, 0] = (x1[:, :, 0] - w1 / 2) / (w1 / 2)
            x1[:, :, 1] = (x1[:, :, 1] - h1 / 2) / (h1 / 2)
            # joint1, velocity1, bone1 = self.multi_input(x1[:, :, :2])
            # joint1 = torch.from_numpy(joint1).float()
            # velocity1 = torch.from_numpy(velocity1).float()
            # bone1 = torch.from_numpy(bone1).float()
            ft1 = torch.from_numpy(x1[:,:,:2]).float()
            kp2 = []
            for pose in self.pose_trainner_results:
                kp2.append(pose['keypoints'])
            kp2 = np.array(kp2)

            h2, w2 = (640, 480)
            x2 = kp2
            x2[:, :, 0] = (x2[:, :, 0] - w2 / 2) / (w2 / 2)
            x2[:, :, 1] = (x2[:, :, 1] - h2 / 2) / (h2 / 2)
            # joint2, velocity2, bone2 = self.multi_input(x2[:, :, :2])
            # joint2 = torch.from_numpy(joint2).float()
            # velocity2 = torch.from_numpy(velocity2).float()
            # bone2 = torch.from_numpy(bone2).float()
            ft2 = torch.from_numpy(x2[:,:,:2]).float()

            # self.embs1 = self.tem_st_gcn(joint1[None].to('cuda'), bone1[None].to('cuda'), velocity1[None].to('cuda'))
            # self.embs2 = self.tem_st_gcn(joint2[None].to('cuda'), bone2[None].to('cuda'), velocity2[None].to('cuda'))
            self.embs1 = self.tem_st_gcn(ft1[None].to('cuda'))
            self.embs2 = self.tem_st_gcn(ft2[None].to('cuda'))
            self.embs1 = self.embs1[0].detach().cpu().numpy()
            self.embs2 = self.embs2[0].detach().cpu().numpy()
            if self.rbtn_pl_train.isChecked():
                nns = align(self.embs1, self.embs2, use_dtw=True)
                self.nns_play_trainner = nns
                self.nns_trainner_play = None
            else:
                nns = align(self.embs2, self.embs1, use_dtw=True)
                self.nns_trainner_play = nns
                self.nns_play_trainner = None
            self.btn_temporal_alignment.setText('Aligned')



    def multi_input(self, data):
            conn = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (8, 6), (9, 7), (10, 8),
                    (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
            data = np.transpose(data, (2, 0, 1))  # T V C -> C T V
            C, T, V = data.shape
            joint = np.zeros((C * 2, T, V))
            velocity = np.zeros((C * 2, T, V))
            bone = np.zeros((2 * C, T, V))
            joint[:C, :, :] = data
            for i in range(V):
                joint[C:, :, i] = data[:, :, i] - data[:, :, 0]

            for i in range(T - 2):
                velocity[:C, i, :] = data[:, i + 1, :] - data[:, i, :]
                velocity[C:, i, :] = data[:, i + 2, :] - data[:, i, :]

            for v1, v2 in conn:
                bone[:C, :, v1] = data[:, :, v1] - data[:, :, v2]
            bone_length = 0

            for i in range(C):
                bone_length += bone[i, :, :] ** 2
            bone_length = np.sqrt(bone_length) + 0.0001

            for i in range(C):
                bone[C + i, :, :] = np.arccos(bone[i, :, :] / bone_length)
            joint = np.transpose(joint, (1, 2, 0))
            velocity = np.transpose(velocity, (1, 2, 0))
            bone = np.transpose(bone, (1, 2, 0))
            return joint, velocity, bone

    def multi_input_v2(self, data):
            conn = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (8, 6), (9, 7), (10, 8),
                    (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
            data = np.transpose(data, (2, 0, 1))  # T V C -> C T V
            C, T, V = data.shape
            joint = np.zeros((C * 2, T, V))
            velocity = np.zeros((C * 2, T, V))
            bone = np.zeros((2 * C, T, V))
            joint[:C, :, :] = data
            for i in range(V):
                joint[C:, :, i] = data[:, :, i] - data[:, :, 0]

            for i in range(T - 2):
                velocity[:C, i, :] = data[:, i + 1, :] - data[:, i, :]
                velocity[C:, i, :] = data[:, i + 2, :] - data[:, i, :]

            for v1, v2 in conn:
                bone[:C, :, v1] = data[:, :, v1] - data[:, :, v2]
            bone_length = 0

            for i in range(C):
                bone_length += bone[i, :, :] ** 2
            bone_length = np.sqrt(bone_length) + 0.0001

            for i in range(C):
                bone[C + i, :, :] = np.arccos(bone[i, :, :] / bone_length)
            # joint = np.transpose(joint, (1, 2, 0))
            # velocity = np.transpose(velocity, (1, 2, 0))
            # bone = np.transpose(bone, (1, 2, 0))
            return joint, velocity, bone


    def connect_camera(self):
        if self.camera is None:
            self.camera = VideoThread('test.avi')
            self.camera.change_pixmap_signal.connect(self.image_set)
            self.camera.fps_display_signal.connect(self.fps_display)
            self.camera.start()
            self.btn_open_camera.setText('OPENNING')

    def fps_display(self, value):
        self.text_fps.setText(str(round(value, 4)))

    def disconnect_camera(self):
        if self.camera is not None:
            self.camera.close = True
            self.camera.stop()
            self.camera = None
            self.btn_open_camera.setText('OPEN CAMERA')



    def start_record(self):
        if self.camera is not None:
            self.disconnect_camera()
            time.sleep(1)
            self.camera = VideoThread(osp.join(f'last_record.avi'), record=True)
            self.camera.change_pixmap_signal.connect(self.image_set)
            self.camera.start()
            self.btn_open_camera.setText('OPENNING')
            self.btn_start_record.setText('RECORDING')


    def stop_record(self):
        if self.camera is not None:
            if self.camera.record:
                self.disconnect_camera()
                self.btn_start_record.setText('START RECORD')
                time.sleep(1)
                self.connect_camera()
                # self.load_video_from_camera()



    def load_video_from_camera(self):
        self.video_path = 'last_record.avi'

        # frame_paths = self.extract_frame(self.video_path)
        if len(self.video_path) == 0:
            return
        self.video = cv2.VideoCapture(self.video_path)
        self.total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.text_total_frame.setText(str(self.total_frame))
        flag, self.curr_frame = self.video.read()


        self.curr_frame = resize_image(self.curr_frame)
        # self.curr_frame = cv2.rotate(self.curr_frame, cv2.ROTATE_90_CLOCKWISE)
        self.image_size = self.curr_frame.shape[:2]

        self.slider_frame_no.setRange(0, int(self.total_frame) - 1)
        self.slider_frame_no.setValue(0)
        self.image_set(self.curr_frame)

    def load_video_trainner(self):
        self.video_trainner_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open video file', filter='Video files (*.mp4 *.mkv *.avi *.ts)')[0]

        if len(self.video_trainner_path) == 0:
            return


        self.video_trainer = cv2.VideoCapture(self.video_trainner_path)
        self.total_frame = int(self.video_trainer.get(cv2.CAP_PROP_FRAME_COUNT))
        # self.text_total_frame.setText(str(self.total_frame))
        _, self.trainner_frame = self.video_trainer.read()
        self.trainner_frame = resize_image(self.trainner_frame)

        self.image_size = (640, 480)

        self.slider_frame_no_2.setRange(0, int(self.total_frame) - 1)
        self.slider_frame_no_2.setValue(0)
        self.image_tranner_set(self.trainner_frame)


    def load_video_from_file(self):
        self.video_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open video file', filter='Video files (*.mp4 *.mkv *.avi)')[0]

        if len(self.video_path) == 0:
            return


        self.video = cv2.VideoCapture(self.video_path)
        self.total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.text_total_frame.setText(str(self.total_frame))
        _, self.curr_frame = self.video.read()
        self.curr_frame = resize_image(self.curr_frame)

        self.image_size = (640, 480)

        self.slider_frame_no.setRange(0, int(self.total_frame) - 1)
        self.slider_frame_no.setValue(0)
        self.image_set(self.curr_frame)


    def extract_frame(self, video_path):
        dname = 'temp/images'
        os.makedirs(dname, exist_ok=True)
        frame_tmpl = os.path.join(dname, 'img_{:05d}.jpg')
        vid = cv2.VideoCapture(video_path)
        frame_paths = []
        flag, frame = vid.read()
        cnt = 0
        while flag:
            frame_path = frame_tmpl.format(cnt + 1)
            frame_paths.append(frame_path)

            cv2.imwrite(frame_path, frame)
            cnt += 1
            flag, frame = vid.read()
        return frame_paths


    def set_frame_no_slider(self, value):
        self.video.set(cv2.CAP_PROP_POS_FRAMES, value)
        _, frame = self.video.read()
        frame = resize_image(frame)

        self.curr_frame = frame
        if self.pose_results is not None:
            if self.rbtn_Skeleton_show.isChecked():
                frame = self.vis_pose(frame, self.pose_results[value])
            if self.rbtn_show_all.isChecked():
                skeleton_frame = np.ones((640, 480, 3), dtype=np.uint8) * 0
                frame = self.vis_pose(skeleton_frame, self.pose_results[value])
        if self.action_pred is not None:
        #     frame_no = self.frame_no_slider.value()
            self.canvas.axes.cla()
            self.canvas.axes.plot(range(len(self.action_pred)), self.action_pred)
            self.canvas.axes.plot(value, self.action_pred[int(value)], marker="o", markersize=10,
                                  markeredgecolor="red", markerfacecolor="red")
            self.canvas.axes.set_yticks(range(4), ['Other Action', 'Back Swing', 'Down Swing', 'F-Through'])
            self.canvas.axes.set_xlabel('Frame #')
            self.canvas.draw()
        if self.nns_play_trainner is not None:
            kp1 = []
            for pose in self.pose_results:
                kp1.append(pose['keypoints'])
            kp1 = np.array(kp1)
            kp_error = kp1[value,5:,:2]

            kp2 = []
            for pose in self.pose_trainner_results:
                kp2.append(pose['keypoints'])
            kp2 = np.array(kp2)

            kp_true = kp2[self.nns_play_trainner[value],5:,:2]
            self.skeleton_transform.find_parameters(kp_error, kp_true)
            points_transformed = self.skeleton_transform.apply(kp_error[:, :2])
            compare = PoseComparison(kp_true, points_transformed)
            different_angles = compare.compare_poses(threshold=20)
            for angle in different_angles:
                p1, p2, p3 = angle[0]
                p1, p2, p3 = p1-5, p2-5, p3-5
                cv2.line(frame, (int(kp_error[p1][0]),int(kp_error[p1][1])), (int(kp_error[p2][0]),int(kp_error[p2][1])),(0, 0, 255), 2)
                cv2.line(frame, (int(kp_error[p2][0]),int(kp_error[p2][1])), (int(kp_error[p3][0]),int(kp_error[p3][1])),(0, 0, 255), 2)
                frame = cv2.circle(frame, (int(kp_error[p1][0]), int(kp_error[p1][1])), 7, (0,0,255), -1)
                frame = cv2.circle(frame, (int(kp_error[p2][0]), int(kp_error[p2][1])), 7, (0, 0, 255), -1)
                frame = cv2.circle(frame, (int(kp_error[p3][0]), int(kp_error[p3][1])), 7, (0, 0, 255), -1)
            self.slider_frame_no_2.setValue(self.nns_play_trainner[value])
            self.set_frame_no_slider2(self.nns_play_trainner[value])

        #
        # if self.pose_results is not None:
        #     skeleton_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        #     skeleton_frame = self.vis_pose(skeleton_frame, self.pose_results[value])
        #     self.res_set(skeleton_frame)

        self.text_frame_no.setText(str(value))
        self.image_set(frame)

    def set_frame_no_slider2(self, value):
        self.video_trainer.set(cv2.CAP_PROP_POS_FRAMES, value)
        _, frame = self.video_trainer.read()
        frame = resize_image(frame)

        self.trainner_frame = frame
        if self.pose_trainner_results is not None:
            if self.rbtn_Skeleton_show.isChecked():
                frame = self.vis_pose(frame, self.pose_trainner_results[value])
            if self.rbtn_show_all.isChecked():
                skeleton_frame = np.ones((640, 480, 3), dtype=np.uint8) * 0
                frame = self.vis_pose(skeleton_frame, self.pose_trainner_results[value])

        #
        # if self.pose_results is not None:
        #     skeleton_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        #     skeleton_frame = self.vis_pose(skeleton_frame, self.pose_results[value])
        #     self.res_set(skeleton_frame)

        self.text_frame_no_2.setText(str(value))
        self.image_tranner_set(frame)
        if self.nns_trainner_play is not None:
            self.slider_frame_no.setValue(self.nns_trainner_play[value])
            self.set_frame_no_slider(self.nns_trainner_play[value])


    def sample_process_video(self):

        if self.video is not None:
            self.btn_sample.setText('Processing')
            inds = uniformly_sample(self.total_frame, int(self.text_num_frame.text()))
            w = int(self.video.get(3))
            h = int(self.video.get(4))

            save_video = cv2.VideoWriter('processed_video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))
            for index in inds:
                self.video.set(cv2.CAP_PROP_POS_FRAMES, index)
                _, frame = self.video.read()
                save_video.write(frame)
                # cv2.waitKey(1)
            time.sleep(1)
            self.video_path = 'processed_video.avi'

            if len(self.video_path) == 0:
                return
            self.video = cv2.VideoCapture(self.video_path)
            self.total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            _, self.curr_frame = self.video.read()
            self.curr_frame = resize_image(self.curr_frame)

            self.image_size = self.curr_frame.shape[:2]

            self.slider_frame_no.setRange(0, int(self.total_frame) - 1)
            self.slider_frame_no.setValue(0)
            self.image_set(self.curr_frame)
            self.btn_sample.setText('Process')


    def pose_detect(self):
        self.thread.video = self.video
        self.thread.start()

    def pose_detect2(self):
        self.thread_trainner.video = self.video_trainer
        self.thread_trainner.start()


    def progressing_bar(self, value):
        value = int(value * 100 / self.total_frame)
        self.progressBar.setValue(value)

    def progressing_bar2(self, value):
        value = int(value * 100 / self.total_frame)
        self.progressBar_2.setValue(value)


    def receive_pose_result(self, pose_results):
        self.pose_results = pose_results

    def receive_pose_result2(self, pose_results):
        self.pose_trainner_results = pose_results

    def print_time_pose_detect(self, value):

        self.text_time_detec_pose.setText(str(round(value, 4)))

    def action_detect(self):
        kp = []
        for pose in self.pose_results:
            kp.append(pose['keypoints'])
        kp = np.array(kp)
        h, w = self.image_size
        kp[:, :, 0] = (kp[:, :, 0] -w/2)/ (w/2)
        kp[:, :, 1] = (kp[:, :, 1]-h/2) / (h/2)

        # joint1, velocity1, bone1 = self.multi_input_v2(kp[:, :, :2])
        # joint1 = torch.from_numpy(joint1).float()
        # velocity1 = torch.from_numpy(velocity1).float()
        # bone1 = torch.from_numpy(bone1).float()

        ft = torch.from_numpy(kp[:,:,:2]).float()
        # ft = ft.permute(2, 0, 1).contiguous()
        # out = self.action_model(joint1[None].to('cuda'), bone1[None].to('cuda'), velocity1[None].to('cuda'))
        out = self.action_model(ft[None].to('cuda'))
        # pred = self.clasifier.predict(embedding[0].detach().cpu().numpy())
        prob = F.softmax(out[0], dim=0)
        pred = prob.argmax(dim=0)
        pred = pred.detach().cpu().numpy()
        self.start_frame = 0
        self.top_frame = 0
        self.impact_frame = 0
        self.end_frame = 0
        self.action_pred = pred
        p_label, p_start, p_end = get_segments(pred, {0: 'Other Action', 1: 'back swing', 2: 'down swing', 3: 'follow through'})

        for i in range(len(p_label)):
            if p_label[i] == 'back swing':
                self.start_frame = p_start[i]
            if p_label[i] == 'down swing':
                self.top_frame = p_start[i]
            if p_label[i] == 'follow through':
                if self.impact_frame != 0:
                    continue
                self.impact_frame = p_start[i]
                self.end_frame = p_end[i]

            # if p_label[i] == 'Other Action':
            #     if self.end_frame != 0:
            #         continue
            #     self.end_frame = p_start[i]
        print(self.start_frame, self.top_frame, self.impact_frame, self.end_frame)

        frame_no = self.slider_frame_no.value()
        self.canvas.axes.cla()
        self.canvas.axes.plot(range(len(pred)), self.action_pred)
        self.canvas.axes.plot(frame_no, self.action_pred[int(frame_no)], marker="o", markersize=5,
                              markeredgecolor="red", markerfacecolor="green")
        self.canvas.axes.set_yticks(range(4), ['Other-Action', 'Back-Swing', 'Down-Swing', 'F-Through'])
        self.canvas.axes.set_xlabel('Frame #')
        self.canvas.draw()
        ################################################################
        # bbox = self.pose_results[self.impact_frame]['bbox'].copy()
        # x1, y1, x2, y2 = bbox
        # box_width, box_height = x2 - x1, y2 - y1
        self.node_label_list = []
        start = int(self.impact_frame)
        end = int(start + 0.5*(self.end_frame - self.impact_frame))
        for value in range(start, end):
            x = self.pose_results[value]['keypoints'].copy()
            x = x[:,:2]

            x[:,0] = (x[:,0] - w/2)/(w/2)
            x[:,1] = (x[:,1] - h/2)/(h/2)
            x = torch.from_numpy(x).float()
            out = self.node_classification_model(x[None].to('cuda'))
            out = out.view(17, 3)
            self.node_label = out.argmax(dim=1)
            self.node_label_list.append(self.node_label)
            point_error = []
            if 1 in self.node_label:
                # self.btn_error_detection.setStyleSheet('QPushButton {background-color: #ff0004; color: white;}')
                point_error.append(1)
            elif 2 in self.node_label:
                # self.btn_error_detection.setStyleSheet('QPushButton {background-color: #0000ff; color: white;}')
                point_error.append(2)

        ################################################################
        # fig = Figure(figsize=(640, 480), dpi=200)
        # axes = fig.add_subplot(111)
        # axes.plot(range(len(pred)), self.action_pred)
        # axes.set_yticks(range(4), ['Other-Action', 'Back-Swing', 'Down-Swing', 'Follow-Through'])
        # axes.set_xlabel('Frame #')
        # fig.show()
    def show_start_frame(self):
        self.slider_frame_no.setValue(self.start_frame)
        self.text_frame_no.setText(str(self.start_frame))
        self.set_frame_no_slider(self.start_frame)

    def show_top_frame(self):
        self.slider_frame_no.setValue(self.top_frame)
        self.text_frame_no.setText(str(self.top_frame))
        self.set_frame_no_slider(self.top_frame)


    def show_impact_frame(self):
        self.slider_frame_no.setValue(self.impact_frame)
        self.text_frame_no.setText(str(self.impact_frame))
        self.set_frame_no_slider(self.impact_frame)

    def show_end_frame(self):
        self.slider_frame_no.setValue(self.end_frame)
        self.text_frame_no.setText(str(self.end_frame))
        self.set_frame_no_slider(self.end_frame)

    def show_back_swing(self):
        for i in range(self.start_frame, self.top_frame):
            self.slider_frame_no.setValue(i)
            self.text_frame_no.setText(str(i))
            self.set_frame_no_slider(i)
            if self.x1_slow.isChecked():
                cv2.waitKey(1)
            if self.x2_slow.isChecked():
                cv2.waitKey(10)
            if self.x4_slow.isChecked():
                cv2.waitKey(100)

    def show_down_swing(self):
        for i in range(self.top_frame, self.impact_frame+1):
            self.slider_frame_no.setValue(i)
            self.text_frame_no.setText(str(i))
            self.set_frame_no_slider(i)
            if self.x1_slow.isChecked():
                cv2.waitKey(1)
            if self.x2_slow.isChecked():
                cv2.waitKey(10)
            if self.x4_slow.isChecked():
                cv2.waitKey(100)

    def show_fthrough_swing(self):
        for i in range(self.impact_frame, self.end_frame):
            self.slider_frame_no.setValue(i)
            self.text_frame_no.setText(str(i))
            self.set_frame_no_slider(i)
            if self.x1_slow.isChecked():
                cv2.waitKey(1)
            if self.x2_slow.isChecked():
                cv2.waitKey(10)
            if self.x4_slow.isChecked():
                cv2.waitKey(100)

    def show_error(self):

        start = int(self.impact_frame)
        end = int(start + 0.5*(self.end_frame-self.impact_frame))
        value = self.slider_frame_no.value()
        if value in range(start, end):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, value)
            _, frame = self.video.read()
            frame = resize_image(frame)
            h, w = self.image_size
            x = self.pose_results[value]['keypoints'].copy()
            x = x[:, :2]

            x[:, 0] = (x[:, 0] - w / 2) / (w / 2)
            x[:, 1] = (x[:, 1] - h / 2) / (h / 2)
            x = torch.from_numpy(x).float()
            out = self.node_classification_model(x[None].to('cuda'))
            out = out.view(17, 3)
            self.node_label = out.argmax(dim=1)
            # self.node_label = self.node_label_list[value-self.impact_frame]
            self.curr_frame = frame
            if self.pose_results is not None:
                if self.rbtn_Skeleton_show.isChecked():
                    if self.node_label is not None:
                        frame = self.vis_pose(frame, self.pose_results[value], node_label=self.node_label)
                if self.rbtn_show_all.isChecked():
                    skeleton_frame = np.ones((640, 480, 3), dtype=np.uint8) * 0
                    frame = self.vis_pose(skeleton_frame, self.pose_results[value])
            if self.action_pred is not None:
                #     frame_no = self.frame_no_slider.value()
                self.canvas.axes.cla()
                self.canvas.axes.plot(range(len(self.action_pred)), self.action_pred)
                self.canvas.axes.plot(value, self.action_pred[int(value)], marker="o", markersize=10,
                                      markeredgecolor="red", markerfacecolor="red")
                self.canvas.axes.set_yticks(range(4), ['Other Action', 'Back Swing', 'Down Swing', 'F-Through'])
                self.canvas.axes.set_xlabel('Frame #')
                self.canvas.draw()

            #
            # if self.pose_results is not None:
            #     skeleton_frame = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            #     skeleton_frame = self.vis_pose(skeleton_frame, self.pose_results[value])
            #     self.res_set(skeleton_frame)

            self.text_frame_no.setText(str(value))
            self.image_set(frame)


    def image_set(self, image):
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # contrast = 5.  # Contrast control ( 0 to 127)
        # brightness = 2.  # Brightness control (0-100)
        # image = cv2.addWeighted(image, contrast, image, 0, brightness)

        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.image_vis_label.setPixmap(QPixmap.fromImage(image_Qt))

    def image_tranner_set(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.image_vis_label_2.setPixmap(QPixmap.fromImage(image_Qt))


    def vis_pose(self, image, pose_result, threshold=0.3, node_label=None):
        bbox = pose_result['bbox']
        keypoints = pose_result['keypoints'][:,:2]
        keypoints_score = pose_result['keypoints'][:,2]

        skeleton_edge = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                         [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                         [8, 10], [1, 2], [0, 1], [0, 2], [1,   3], [2, 4],
                         [3, 5], [4, 6]]
        for edge in skeleton_edge:
            start = keypoints[edge[0]]
            end = keypoints[edge[1]]
            if keypoints_score[edge[0]] < threshold or keypoints_score[edge[1]] < threshold:
                continue
            image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255, 255, 0), 2)

        for i in range(17):
            if keypoints_score[i] < threshold:
                continue
            (x, y) = keypoints[i]
            if node_label is None:
                color = (255, 255, 255)
            else:
                if node_label[i] == 0:
                    color = (255, 255, 255)
                elif node_label[i] == 1:
                    color = (0, 0, 255)
                elif node_label[i] == 2:
                    color = (255, 0, 0)

            image = cv2.circle(image, (int(x), int(y)), 7, color, -1)

        image_vis = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        return image_vis

    def closeEvent(self, event):
        # self.model_yolov7.destory()
        # self.model_hrnet.destory()
        event.accept()

def main():
    app = QApplication([])
    window = My_GUI()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()
