from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
import cv2
import os
import os.path as osp
import imutils
import numpy as np
import torch
import time
import pandas as pd
# from PoseEstimation.Hrnet import Hrnet
# from PoseEstimation.Yolov7 import Yolov7
from PoseEstimation_v2.pose_detect import Hrnet, Yolov7

from torch.nn.functional import softmax
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from SCL import ST_GCN
import pickle
from sklearn.svm import SVC
from scipy.spatial.distance import cdist
from dtw import dtw
from sklearn.manifold import TSNE
np.random.seed(42)

def dist_fn(x, y):
  dist = 1 - (x @ y.T)/(np.linalg.norm(x)*np.linalg.norm(y))
  return dist


def align(query_feats, candidate_feats, use_dtw=False):
    """Align videos based on nearest neighbor or dynamic time warping."""
    if use_dtw:
        _, _, _, path = dtw(query_feats, candidate_feats, dist='cosine')
        _, uix = np.unique(path[0], return_index=True)
        nns = path[1][uix]
        return nns
    else:
        dists = cdist(query_feats, candidate_feats, 'cosine')
        nns = np.argmin(dists, axis=1)
        return nns


def resize_img(im, new_shape=(640, 480), color=(0, 0, 0), auto=False, scaleup=True, stride=32):
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

class Pose_detect_thread(QThread):
    pose_results = pyqtSignal(list)
    progressing = pyqtSignal(int)


    def __init__(self, Yolov7, Hrnet, video):
        super(Pose_detect_thread, self).__init__()
        self.Yolov7 = Yolov7
        self.Hrnet = Hrnet
        self.video = video
        self.time_process = 0


    def run(self):
        cnt = 0
        total_frame = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_results = []
        self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        flag, image = self.video.read()
        while flag:
            image = resize_img(image, (640, 480))
            detections = self.Yolov7.inference(image)
            pose_res = self.Hrnet.inference_from_bbox(image, detections)
            cnt += 1
            self.progressing.emit(cnt)
            pose_results.append(pose_res)
            flag, image = self.video.read()
        self.pose_results.emit(pose_results)


class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)



class My_GUI(QMainWindow):

    def __init__(self):
        super(My_GUI, self).__init__()
        uic.loadUi('form1.ui', self)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.canvas1 = MplCanvas(self, width=5, height=4, dpi=100)

        self.gridLayout.addWidget(self.canvas)
        self.gridLayout_2.addWidget(self.canvas1)
        self.tsne = TSNE(random_state=1, metric="cosine")

        self.pose_results = None
        self.nns = None
        self.model_yolov7 = Yolov7(engine_path='PoseEstimation_v2/rtmdet-nano')
        self.model_hrnet = Hrnet(engine_path='PoseEstimation_v2/rtmpose-m')

        self.quit = QAction("Quit", self)
        self.quit.triggered.connect(self.closeEvent)

        self.thread = Pose_detect_thread(Yolov7=self.model_yolov7, Hrnet=self.model_hrnet, video=None)
        # self.thread.pose_results.connect(self.receive_pose_result)

        self.model = ST_GCN(in_channels=4, embedding_size=128)
        self.model.load_state_dict(torch.load('model_110.pth'))
        self.model.eval()
        self.model.to('cuda')

        self.video_path1 = ''
        self.video_path2 = ''
        self.load_video_btn1.clicked.connect(self.load_video1)
        self.load_video_btn2.clicked.connect(self.load_video2)
        self.detect_pose_btn1.clicked.connect(self.pose_detect1)
        self.detect_pose_btn2.clicked.connect(self.pose_detect2)
        self.compare_btn.clicked.connect(self.extract_embeddings)
        self.frame_no_slider.valueChanged.connect(self.set_frame_no_slider)

    def pose_detect1(self):
        self.thread.video = self.video1
        self.thread.progressing.connect(self.progressing_bar1)
        self.thread.pose_results.connect(self.receive_pose_result1)
        self.thread.start()

    def pose_detect2(self):
        self.thread.video = self.video2

        self.thread.progressing.disconnect(self.progressing_bar1)
        self.thread.progressing.connect(self.progressing_bar2)

        self.thread.pose_results.disconnect(self.receive_pose_result1)
        self.thread.pose_results.connect(self.receive_pose_result2)
        self.thread.start()

    def progressing_bar1(self, value):
        value = int(value * 100 / self.total_frame1)
        self.progressBar1.setValue(value)

    def progressing_bar2(self, value):
        value = int(value * 100 / self.total_frame2)
        self.progressBar2.setValue(value)

    def extract_embeddings(self):
        kp1 = []
        for pose in self.pose_results1:
            kp1.append(pose['keypoints'])
        kp1 = np.array(kp1)

        h1, w1 = self.image_size1
        x1 = kp1
        x1[:, :, 0] = (x1[:, :, 0] - w1/2)/(w1/2)
        x1[:, :, 1] = (x1[:, :, 1] - h1/2)/(h1/2)
        joint1, velocity1, bone1 = self.multi_input(x1[:,:,:2])
        joint1 = torch.from_numpy(joint1).float()
        velocity1 = torch.from_numpy(velocity1).float()
        bone1 = torch.from_numpy(bone1).float()


        kp2 = []
        for pose in self.pose_results2:
            kp2.append(pose['keypoints'])
        kp2 = np.array(kp2)

        h2, w2 = self.image_size2
        x2 = kp2
        x2[:, :, 0] = (x2[:, :, 0]-w2/2)/(w2/2)
        x2[:, :, 1] = (x2[:, :, 1]-h2/2)/(h2/2)
        joint2, velocity2, bone2 = self.multi_input(x2[:,:,:2])
        joint2 = torch.from_numpy(joint2).float()
        velocity2 = torch.from_numpy(velocity2).float()
        bone2 = torch.from_numpy(bone2).float()

        self.embs1 = self.model(joint1[None].to('cuda'), bone1[None].to('cuda'), velocity1[None].to('cuda'))
        self.embs2 = self.model(joint2[None].to('cuda'), bone2[None].to('cuda'), velocity2[None].to('cuda'))
        self.embs1 = self.embs1[0].detach().cpu().numpy()
        self.embs2 = self.embs2[0].detach().cpu().numpy()
        # nns, dist = align(self.embs1, self.embs2)
        # nns,_ = align(self.embs1, self.embs2, use_dtw=False)
        nns = align(self.embs1, self.embs2, use_dtw=False)
        self.nns = nns
        distance = []
        for i in range(len(nns)):
            d = dist_fn(self.embs1[i], self.embs2[nns[i]])
            distance.append(d)
        self.distance = distance
        self.canvas.axes.cla()
        self.canvas.axes.plot(range(len(distance)), distance)
        self.canvas.axes.set_xlabel('Frame #')
        self.canvas.axes.plot(self.frame_no_slider.value(), distance[self.frame_no_slider.value()], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")

        self.canvas.axes.set_ylabel('Distance')
        self.canvas.draw()
        self.canvas1.axes.cla()
        self.emb1_vis = self.tsne.fit_transform(self.embs1)
        self.emb2_vis = self.tsne.fit_transform(self.embs2)
        self.canvas1.axes.scatter(self.emb1_vis[:,0], self.emb1_vis[:,1], c='r', label='video1')
        self.canvas1.axes.scatter(self.emb2_vis[:, 0], self.emb2_vis[:, 1], c='b', label='video2')
        self.canvas1.axes.legend()
        self.canvas1.draw()



    def set_frame_no_slider(self, value):
        self.video1.set(cv2.CAP_PROP_POS_FRAMES, value)
        _, frame = self.video1.read()
        frame = resize_img(frame, (640, 480))
        self.curr_frame1 = frame
        if self.checkBox1.isChecked():
            self.curr_frame1 = self.vis_pose(frame, self.pose_results1[value])

        if self.checkBox1_2.isChecked():
            skeleton_frame = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)*255
            self.curr_frame1 = self.vis_pose(skeleton_frame, self.pose_results1[value])

        self.video2.set(cv2.CAP_PROP_POS_FRAMES, self.nns[value])
        _, frame = self.video2.read()
        frame = resize_img(frame, (640, 480))
        self.curr_frame2 = frame
        if self.checkBox2.isChecked():
            self.curr_frame2 = self.vis_pose(frame, self.pose_results2[self.nns[value]])

        if self.checkBox2_2.isChecked():
            skeleton_frame = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)*255
            self.curr_frame2 = self.vis_pose(skeleton_frame, self.pose_results2[self.nns[value]])

        if self.nns is not None:
            self.canvas.axes.cla()
            self.canvas.axes.plot(range(len(self.distance)), self.distance)
            self.canvas.axes.set_xlabel('Frame #')
            self.canvas.axes.plot(value, self.distance[value], marker="o", markersize=5,
                                  markeredgecolor="red", markerfacecolor="green")

            self.canvas.axes.set_ylabel('Distance')
            self.canvas.draw()
            self.canvas1.axes.cla()
            self.canvas1.axes.scatter(self.emb1_vis[:, 0], self.emb1_vis[:, 1], c='r', label='video1')
            self.canvas1.axes.scatter(self.emb2_vis[:, 0], self.emb2_vis[:, 1], c='b', label='video2')
            self.canvas1.axes.plot(self.emb1_vis[value, 0], self.emb1_vis[value, 1],marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green" )
            self.canvas1.axes.plot(self.emb2_vis[self.nns[value], 0], self.emb2_vis[self.nns[value], 1],marker="o", markersize=5, markeredgecolor="black", markerfacecolor="black" )

            self.canvas1.axes.legend()
            self.canvas1.draw()

        self.image1_set(self.curr_frame1)
        self.image2_set(self.curr_frame2)

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

    def receive_pose_result1(self, pose_results):
        self.pose_results1 = pose_results

    def receive_pose_result2(self, pose_results):
        self.pose_results2 = pose_results

    def load_video1(self):
        self.video_path1 = QtWidgets.QFileDialog.getOpenFileName(self,'Open video file', filter='Video files (*.mp4 *.mkv, *.avi)')[0]
        if len(self.video_path1) == 0:
            return
        self.video1 = cv2.VideoCapture(self.video_path1)
        self.total_frame1 = int(self.video1.get(cv2.CAP_PROP_FRAME_COUNT))
        _, self.curr_frame1 = self.video1.read()
        self.curr_frame1 = resize_img(self.curr_frame1, (640, 480))
        self.image_size1 = (640, 480)
        self.frame_no_slider.setRange(0, int(self.total_frame1) - 1)
        self.frame_no_slider.setValue(0)
        self.image1_set(self.curr_frame1)
        self.pose_results1 = None


    def load_video2(self):
        self.video_path2 = QtWidgets.QFileDialog.getOpenFileName(self,'Open video file', filter='Video files (*.mp4 *.mkv, *.avi)')[0]
        if len(self.video_path2) == 0:
            return
        self.video2 = cv2.VideoCapture(self.video_path2)
        self.total_frame2 = int(self.video2.get(cv2.CAP_PROP_FRAME_COUNT))
        _, self.curr_frame2 = self.video2.read()
        self.curr_frame2 = resize_img(self.curr_frame2, (640, 480))
        self.image_size2 = (640, 480)

        self.image2_set(self.curr_frame2)
        self.pose_results = None


    def image1_set(self, image):
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_img(image, (640, 480))
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.image_label_1.setPixmap(QPixmap.fromImage(image_Qt))

    def image2_set(self, image):
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_img(image, (640, 480))
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.image_label_2.setPixmap(QPixmap.fromImage(image_Qt))


    def vis_pose(self, image, pose_result, threshold=0.3):
        bbox = pose_result['bbox']
        keypoints = pose_result['keypoints'][:,:2]
        keypoints_score = pose_result['keypoints'][:,2]

        skeleton_edge = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
                         [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                         [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
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
            color = (255, 255, 255)

            image = cv2.circle(image, (int(x), int(y)), 5, color, -1)

        image_vis = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        return image_vis

    def closeEvent(self, event):
        self.model_yolov7.destory()
        self.model_hrnet.destory()
        event.accept()

def main():
    app = QApplication([])
    window = My_GUI()
    window.show()
    app.exec_()

if __name__ == '__main__':
    main()