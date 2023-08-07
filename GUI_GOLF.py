from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import cv2
import imutils
import numpy as np
import copy
import sys
import pandas as pd
import time
import tensorrt
# from mmdeploy_runtime import PoseDetector, Detector
from PoseEstimation.Hrnet1 import Hrnet
from PoseEstimation.Yolov71 import Yolov7
# from inference_topdown_pose import inference_img

# DET_MODEL = Detector('Pose-Estimate/Yolov7-tiny/', 'cuda')
# POSE_MODEL = PoseDetector('Pose-Estimate/HRNet/', 'cuda')

def inference_image(img,detect:Yolov7,pose:Hrnet):
    det_results = detect.inference(img)
    pose_results = pose.inference_from_bbox(img,det_results)
    return pose_results

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

class My_GUI(QMainWindow):
    def __init__(self):
        super(My_GUI, self).__init__()
        uic.loadUi('form.ui',self)
        self.show()

        self.det_config = 'Pose/yolox_s_8x8_300e_coco.py'
        self.det_checkpoint = 'Pose/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'
        self.pose_config = 'Pose/hrnet_w48_coco_256x192.py'
        self.pose_checkpoint = 'Pose/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'

        self.skeleton_edge = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
                                (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
                                (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),
                                (3, 5), (4, 6)]

        self.msg = QMessageBox()
        self.keypoints = []
        self.msg.setWindowTitle('Error')
        self.label = np.zeros(17)
        self.ano = []
        self.Hrnet = Hrnet(engine_path='PoseEstimation/HR_net48.trt')
        self.Hrnet.get_fps()
        # self.Hrnet.destory()
        self.Yolov7 = Yolov7(engine_path='PoseEstimation/yolov7-tiny-nms.trt')
        self.Yolov7.get_fps()
        # self.Hrnet.destory()
        
        self.quit = QAction("Quit", self)
        self.quit.triggered.connect(self.closeEvent)
        
        self.slider_frame_no.valueChanged.connect(self.frame_change)

        self.checkBox_1.stateChanged.connect(self.r_shoulder)
        self.checkBox_2.stateChanged.connect(self.r_elbow)
        self.checkBox_3.stateChanged.connect(self.r_wrist)
        self.checkBox_4.stateChanged.connect(self.r_hip)
        self.checkBox_5.stateChanged.connect(self.r_knee)
        self.checkBox_6.stateChanged.connect(self.r_ankle)
        self.checkBox_15.stateChanged.connect(self.l_shoulder)
        self.checkBox_18.stateChanged.connect(self.l_elbow)
        self.checkBox_17.stateChanged.connect(self.l_wrist)
        self.checkBox_7.stateChanged.connect(self.l_hip)
        self.checkBox_19.stateChanged.connect(self.l_knee)
        self.checkBox_16.stateChanged.connect(self.l_ankle)
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_load_image.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.save)
        self.btn_export.clicked.connect(self.export)
        # self.btn_remove_last.clicked.connect(self.remove)
        # self.btn_detect_vid.clicked.connect(self.detect)
        #
    def detect(self):
        # mask = np.zeros((self.size_image[0]+2,self.size_image[1]+2), np.uint8)
        # result = self.frame_original.copy()
        # cv2.floodFill(result,mask,(0,0),(255,255,255),flags=8)
        # image_Qt = QImage(frame_show, frame_show.shape[1], frame_show.shape[0], frame_show.strides[0], QImage.Format_RGB888)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(image_Qt))
        # img = np.full(self.frame_original.shape, 120, dtype = np.uint8)
        # frame_show = self.vis_pose(img,self.pose_result)
        cv2.imwrite('BBOX.png',self.frame_original)
        
    def closeEvent(self, event):
        self.Yolov7.destory()
        self.Hrnet.destory()
        event.accept()    
    
    def remove(self):
        if len(self.ano) == 0:
            return
        self.ano.pop()
        self.Text_display.setText(f'Length anno: {len(self.ano)} \nsave:{self.ano}')
    
    def load_image(self):
        self.image_path = QtWidgets.QFileDialog.getOpenFileName(self,'Open image file',filter='Image file (*.jpg *png)')[0]
        if len(self.image_path)==0:
            return
        self.usingimage = True
        self.frame_original= cv2.imread(self.image_path)
        self.frame_original = resize_img(self.frame_original)
        self.size_image = self.frame_original.shape[:2]
        self.frame_original.flags.writeable = False

        start = time.time()
        # _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
        #                                                   self.pose_checkpoint, self.frame_original)
        self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet)
        print(time.time() - start)
        ###########
        # bbox = self.pose_result[0]['bbox'][:4]
        # frame_show = cv2.rectangle(self.frame_original, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        #############
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        frame_show = self.vis_pose(self.frame_original, self.pose_result)
        # img = np.full((500,500,3), 120, dtype = np.uint8)
        # frame_show = self.vis_pose(img,self.pose_result)
        self.image_set(frame_show)
        # cv2.imwrite('BBOX.png',frame_show)

    def load_video(self):
        # self.video_path = QtWidgets.QFileDialog.getOpenFileName()[0]
        # self.Video = cv2.VideoCapture(self.video_path)
        # _, frame_show = self.Video.read()
        # self.size_image = frame_show.shape[:2]
        # self.image_set(frame_show)
        self.Video_path = QtWidgets.QFileDialog.getOpenFileName(self,'Open video file', filter='Video files (*.mp4 *.mkv *.avi)')[0]
        if len(self.Video_path) == 0:
            return
        self.usingimage=False
        self.Video = cv2.VideoCapture(self.Video_path)
        _, self.frame_original = self.Video.read()
        self.size_image = (self.frame_original.shape[0],self.frame_original.shape[1])
        self.frame_original.flags.writeable = False
        # self.image_set(self.frame_original)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(frame_show))
        self.total_frame = int(self.Video.get(cv2.CAP_PROP_FRAME_COUNT))
        # _, self.curr_frame = self.Video.read()
        self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet)
        frame_show = self.vis_pose(self.frame_original, self.pose_result)
        self.image_set(frame_show)
        self.slider_frame_no.setRange(0, int(self.total_frame) - 1)
        self.slider_frame_no.setValue(0)
        
    def frame_change(self, value):
        self.Video.set(cv2.CAP_PROP_POS_FRAMES, value)
        _, self.frame_original = self.Video.read()
        # self.size_image = self.frame_original.shape[:2]
        # _, self.pose_result = inference_img(self.det_config, self.det_checkpoint, self.pose_config,
        #                                                   self.pose_checkpoint, self.frame_original)
        # # print(time.time() - start)
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        self.frame_original.flags.writeable = False
        # self.frame_no = value
        self.Text_frame_no.setText(str(value))
        # self.pose_result = inference_image(self.frame_original,self.Yolov7,self.Hrnet)
        ##############
        bbox = self.pose_result[0]['bbox'][:4]
        cv2.rectangle(self.frame_original, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        ######################
        # frame_show = self.vis_pose(self.frame_original, self.pose_result)
        self.image_set(self.frame_original)
        # self.Label_Img_Show.setPixmap(QPixmap.fromImage(self.frame_original))
        # self.image_set(frame)
        # self.frame_no_txt.setText(str(value))


    def vis_pose(self, image, pose_result):
        bbox = []
        bbox_score = []
        keypoints = []
        keypoints_score = []
        if pose_result is None:
            return image
        for pos in pose_result:
            bbox.append(pos['bbox'][:4])
            bbox_score.append(pos['bbox'][4])
            keypoints.append(pos['keypoints'][:,:2])
            keypoints_score.append(pos['keypoints'][:,2])
        max_score_indx = np.argmax(bbox_score)
        bbox = bbox[max_score_indx]
        keypoints = keypoints[max_score_indx]
        self.skeleton_features = pose_result[max_score_indx]['keypoints']
        self.keypoints = keypoints
        for edge in self.skeleton_edge:
            start = keypoints[edge[0]]
            end = keypoints[edge[1]]
            image = cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (255,255,0), 2)

        for i in range(17):
            (x, y) = keypoints[i]
            if self.label[i] == 0:
                color = (255, 255, 255)
            elif self.label[i] == 1:
                color = (0, 0, 255)
            elif self.label[i] == 2:
                color = (255, 0, 0)

            image = cv2.circle(image, (int(x), int(y)), 7, color, -1)

        image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])) , (0,255,0), 1)
        return image


    # def image_set(self, image):
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     image = imutils.resize(image, height=640)
    #     image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
    #     self.Label_Img_Show.setPixmap(QPixmap.fromImage(image_Qt))
    
    def image_set(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] >640:
            image = imutils.resize(image, height=640)
            # image = resize_img(image,(640,480))
        # image = resize_img(image,(640,480))
        # self.size_image = image.shape[:2]
        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.Label_Img_Show.setPixmap(QPixmap.fromImage(image_Qt))

    
    def r_shoulder(self, state):
        index = 6
        class_name = self.Cbox_RShoulder.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)


    def r_elbow(self, state):
        index = 8
        class_name = self.Cbox_RElbow.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def r_wrist(self, state):
        index = 10
        class_name = self.Cbox_RWrist.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def r_hip(self, state):
        index = 12
        class_name = self.Cbox_RHip.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def r_knee(self, state):
        index = 14
        class_name = self.Cbox_RKnee.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)


    def r_ankle(self, state):
        index = 16
        class_name = self.Cbox_RAnkle.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def l_shoulder(self, state):
        index = 5
        class_name = self.Cbox_LShoulder.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def l_elbow(self, state):
        index = 7
        class_name = self.Cbox_LElbow.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def l_wrist(self, state):
        index = 9
        class_name = self.Cbox_LWrist.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def l_hip(self, state):
        index = 11
        class_name = self.Cbox_LHip.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def l_knee(self, state):
        index = 13
        class_name = self.Cbox_LKnee.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def l_ankle(self, state):
        index = 15
        class_name = self.Cbox_LAnkle.currentText()

        if state == QtCore.Qt.Checked:
            if class_name == 'Class1':
                self.label[index] = 1

            elif class_name == 'Class2':
                self.label[index] = 2
        else:
            self.label[index] = 0

        frame_show = self.vis_pose(self.frame_original, self.pose_result)

        self.Text_display.setText(f'label: {self.label}')
        self.image_set(frame_show)

    def save(self):
        if len(self.skeleton_features) == 0:
            self.msg.setText(f"Frame don't have the skeleton" )
            self.msg.exec_()
        self.ano.append({'keypoints': self.skeleton_features, 'label':self.label, 'image size': self.size_image})
        if self.usingimage:
            print(f'Save image from: {self.image_path}')
        else:
            self.frame_no = self.slider_frame_no.value() + 1
            self.slider_frame_no.setValue(self.frame_no)
            self.frame_change(self.frame_no)
        self.Text_display.setText(f'len ano: {len(self.ano)} \nsave:{self.ano}')

    def export(self):
        file_name = self.Edit_file_name.text()

        pd.to_pickle(self.ano, file_name)

def main():
    app = QApplication([])
    window = My_GUI()
    app.exec_()


if __name__ == "__main__":
    main()