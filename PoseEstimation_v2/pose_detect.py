import cv2
import numpy as np
from mmdeploy_runtime import PoseDetector, Detector
import time
import threading


class myThread(threading.Thread):
   def __init__(self, func, args):
      threading.Thread.__init__(self)
      self.func = func
      self.args = args
   def run(self):
      print ("Starting thread" )
      print(self.func(self.args))
      print ("Exiting thread")


class Yolov7(object):
    def __init__(self, engine_path):
        self.detector = Detector(model_path=engine_path, device_name='cuda')

    def inference(self, image):
        bboxes, labels, _ = self.detector(image)
        return bboxes, labels

class Hrnet(object):
    def __init__(self, engine_path):
        self.pose_detector = PoseDetector(model_path=engine_path, device_name='cuda')

    def inference_from_bbox(self, image, detections):
        bboxes, labels = detections

        keep = np.logical_and(labels == 0, bboxes[..., 4] >= 0.4)
        bboxes = bboxes[keep, :4]
        if len(bboxes) == 0:
            return  {'bbox':np.zeros(4), 'keypoints': np.zeros((17, 3))}
        area = []
        for (x1, y1, x2, y2) in bboxes:
            area.append((x2 - x1) * (y2 - y1))
        idx_best = np.array(area).argmax()
        bboxes = bboxes[idx_best]
        pose = self.pose_detector(image, bboxes)
        return {'bbox': bboxes, 'keypoints': pose[0]}


def vis_pose(image, pose_result, threshold=0.3, node_label=None):
    bbox = pose_result['bbox']
    keypoints = pose_result['keypoints'][:, :2]
    keypoints_score = pose_result['keypoints'][:, 2]

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
        if node_label is None:
            color = (255, 255, 255)
        else:
            if node_label[i] == 0:
                color = (255, 255, 255)
            elif node_label[i] == 1:
                color = (0, 0, 255)
            elif node_label[i] == 2:
                color = (255, 0, 0)

        image = cv2.circle(image, (int(x), int(y)), 5, color, -1)

    image_vis = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
    return image_vis



if __name__ == '__main__':

    # detector = Detector(
    #         model_path='mmdeploy/rtmpose-trt/rtmdet-nano', device_name='cuda')
    #
    # pose_detector = PoseDetector(
    #         model_path='mmdeploy/rtmpose-trt/rtmpose-m', device_name='cuda')

    image = cv2.imread('output3.png')
    Hrnet_model = Hrnet(engine_path='mmdeploy/rtmpose-trt/rtmpose-m')
    # Hrnet.get_fps()
    Yolov7_model = Yolov7(engine_path='mmdeploy/yolov7')
    detections = Yolov7_model.inference(image)
    pose_results = Hrnet_model.inference_from_bbox(image, detections)
    print(pose_results)
    img_show = vis_pose(image, pose_results)
    cv2.imshow('vis',img_show)
    cv2.waitKey(-1)
    # latency_list = []
    # for i in range(20):
    #     start_time = time.perf_counter()
    #     bboxes, labels, _ = detector(image)
    #
    #     # filter detections
    #     keep = np.logical_and(labels == 0, bboxes[..., 4] > 0.6)
    #     bboxes = bboxes[keep, :4]
    #     print(bboxes)
    #     pose = pose_detector(image, bboxes)
    #     print(pose.shape)
    #     # filter detections
    #     latency_list.append(time.perf_counter() - start_time)
    # print(latency_list)
