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
from PoseEstimation.Hrnet import Hrnet
from PoseEstimation.Yolov7 import Yolov7
from Action_Segmentation import TransformerModel, ST_GCN
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import os



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



class My_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('GUI.ui', self)

        self.camera = None
        self.video = None
        self.pose_results = None
        self.action_pred = None
        self.save_dir = 'C:/Users/Vas/Desktop/Video Record/'
        self.model_yolov7 = Yolov7(engine_path='PoseEstimation/yolov7-tiny-nms.trt')
        self.model_hrnet = Hrnet(engine_path='PoseEstimation/HR_net48.trt')
        # self.action_model = TransformerModel(n_classes=4)
        self.action_model = ST_GCN(in_channels=3, n_classes=4)
        self.action_model.load_state_dict(torch.load('checkpoints/model_best.pth'))
        self.action_model.to('cuda')
        self.action_model.eval()

        self.quit = QAction("Quit", self)
        self.quit.triggered.connect(self.closeEvent)

        # Thread for pose estimation

        self.thread = Pose_detect_thread(Yolov7=self.model_yolov7, Hrnet=self.model_hrnet, video=None)
        self.thread.progressing.connect(self.progressing_bar)
        self.thread.pose_results.connect(self.receive_pose_result)
        self.thread.finished.connect(self.print_time_pose_detect)

        self.btn_open_camera.clicked.connect(self.connect_camera)
        self.btn_close_camera.clicked.connect(self.disconnect_camera)
        self.btn_start_record.clicked.connect(self.start_record)
        self.btn_stop_record.clicked.connect(self.stop_record)
        self.btn_load_video1.clicked.connect(self.load_video_from_camera)
        self.btn_load_video2.clicked.connect(self.load_video_from_file)
        self.slider_frame_no.valueChanged.connect(self.set_frame_no_slider)
        self.btn_sample.clicked.connect(self.sample_process_video)
        self.btn_detect_pose.clicked.connect(self.pose_detect)
        self.btn_run_action_model.clicked.connect(self.action_detect)

        self.canvas = MplCanvas(self, width=6, height=4, dpi=100)
        self.gridLayout.addWidget(self.canvas)
        self.canvas.axes.set_yticks(range(4), ['Other Action', 'BackSwing', 'DownSwing', 'F-Through'])
        self.canvas.axes.set_xlabel('Frame #')
        self.canvas.draw()


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
            self.camera = VideoThread(osp.join('last_record.avi'), record=True)
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


    def progressing_bar(self, value):
        value = int(value * 100 / self.total_frame)
        self.progressBar.setValue(value)


    def receive_pose_result(self, pose_results):
        self.pose_results = pose_results

    def print_time_pose_detect(self, value):
        self.text_time_detec_pose.setText(str(round(value, 4)))

    def action_detect(self):
        kp = []
        for pose in self.pose_results:
            kp.append(pose['keypoints'])
        kp = np.array(kp)
        h, w = self.image_size
        kp[:, :, 0] = kp[:, :, 0] / w
        kp[:, :, 1] = kp[:, :, 1] / h
        ft = torch.from_numpy(kp).float()
        # ft = ft.permute(2, 0, 1).contiguous()
        out = self.action_model(ft[None].to('cuda'))
        # pred = self.clasifier.predict(embedding[0].detach().cpu().numpy())
        prob = F.softmax(out[0], dim=0)
        pred = prob.argmax(dim=0)
        pred = pred.detach().cpu().numpy()

        self.action_pred = pred
        frame_no = self.slider_frame_no.value()
        self.canvas.axes.cla()
        self.canvas.axes.plot(range(len(pred)), self.action_pred)
        self.canvas.axes.plot(frame_no, self.action_pred[int(frame_no)], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
        self.canvas.axes.set_yticks(range(4), ['No Action', 'Back Swing', 'Down Swing', 'F-Through'])
        self.canvas.axes.set_xlabel('Frame #')
        self.canvas.draw()

    def image_set(self, image):
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # contrast = 5.  # Contrast control ( 0 to 127)
        # brightness = 2.  # Brightness control (0-100)
        # image = cv2.addWeighted(image, contrast, image, 0, brightness)

        image_Qt = QImage(image, image.shape[1], image.shape[0], image.strides[0], QImage.Format_RGB888)
        self.image_vis_label.setPixmap(QPixmap.fromImage(image_Qt))


    def vis_pose(self, image, pose_result, threshold=0.3):
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
