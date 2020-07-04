#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import cv2
import pathlib

def draw_rect(image, box):
    y_min = int(max(1, (box[0] * 300)))
    x_min = int(max(1, (box[1] * 300)))
    y_max = int(min(300, (box[2] * 300)))
    x_max = int(min(300, (box[3] * 300)))
    
    # draw a rectangle on the image
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)


# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model/detect.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

# set threshold
threshold = 0.75

# start webcam capture
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()
    
    # resize
    img = cv2.resize(img, (300, 300))
    
    # detection
    interpreter.set_tensor(input_details[0]['index'], [img])
    interpreter.invoke()
    
    # output details
    rects = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    count = int(interpreter.get_tensor(output_details[3]['index'])[0])
    
    # store results
    # @TODO - can be used to display labels later
    results = [{'bounding_box': rects[i],
                'class_id': classes[i],
                'score': scores[i]} \
               for i in range(count) \
               if scores[i] >= threshold]
    
    # if score is high enough, draw rectangle
    for index, score in enumerate(scores):
        if score > threshold:
            draw_rect(img, rects[index])

    # Display the resulting frame
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()