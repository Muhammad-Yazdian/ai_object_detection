# ==============================================================================
# object_detection_video.py
#
# A simple OpenCV example for detecting common objects in a real-time video 
# stream from USB camere
#
# Author(s):
#   Seied Muhammad Yazdian
#
# Last update:
#   April 6, 2022
#
# Sources:
#   Instruction: https://www.youtube.com/watch?v=HXDD7-EnGBY
#   ssd coco: https://github.com/zafarRehan/object_detection_COCO
#
# Note:
#   Use Ctrl+C to terminate the program
# ==============================================================================

import cv2
import numpy as np

THRESHOLD = 0.60

classNames = []
classFile = 'coco.names'

with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean(127.5)
net.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()
    classIds, confidences, bboxes = net.detect(img, THRESHOLD)
    # print(classIds, bboxes)
    if len(classIds) != 0:
        for classId, confidence, bbox in zip(classIds.flatten(), 
                                             confidences.flatten(), 
                                             bboxes):
            cv2.rectangle(img, bbox[:2], bbox[2:],
                          color=(0, 255, 0), thickness=2)
            cv2.putText(img, 
                        classNames[classId].upper(), 
                        bbox[:2]+np.array([4, 22]),
                        cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)

    cv2.imshow('Output', img)
    cv2.waitKey(1)
