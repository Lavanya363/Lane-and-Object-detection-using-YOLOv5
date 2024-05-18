# -*- coding: utf-8 -*-
"""
Created on Wed May 15 23:24:45 2024

@author: SRI LAVANYA
"""


#"C:\Users\SRI LAVANYA\Desktop\files\deploying ma\detection4.py"

#case - 2
'''import cv2
import numpy as np
from matplotlib import pyplot as plt


def region_of_intrest(img,vertices):
	black = np.zeros_like(img)
	# channel_count = img.shape[2]
	match_mask_color = 255 
	cv2.fillPoly(black,vertices,match_mask_color)
	mask_image = cv2.bitwise_and(img,black)
	return mask_image

def draw_the_lines(img,lines):
	blank = np.zeros_like(img)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(blank,(x1,y1),(x2,y2),(0,230,0),thickness=10)	
	img = cv2.addWeighted(img,0.8,blank,1,1)
	return img		

def process(img):
	# print(img.shape)
	height = img.shape[0]
	width = img.shape[1]

	region_of_intrest_vertices = [
		(0,height),

		(width/2,447),
		(width,height)
	]

	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	guass = cv2.GaussianBlur(gray,(5,5),0)
	canny_image = cv2.Canny(guass,100,120)
	new_img = region_of_intrest(canny_image,np.array([region_of_intrest_vertices],np.int32))
	lines = cv2.HoughLinesP(new_img,rho=2,theta=np.pi/180,threshold=85,lines=np.array([]),minLineLength=40,maxLineGap=250)
	final_img = draw_the_lines(img,lines)
	return final_img

cap = cv2.VideoCapture("C:/Users/SRI LAVANYA/Downloads/video (3).mp4")
while(cap.isOpened()):
	ret,frame = cap.read()
	img = process(frame)
	cv2.imshow('lane_lines_detection',img)
	if cv2.waitKey(1)==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()'''



'''import cv2
import numpy as np
from matplotlib import pyplot as plt

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    blank_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 1)
    return img

def process(img):
    height = img.shape[0]
    width = img.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2 + 50),
        (width, height)
    ]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gauss = cv2.GaussianBlur(gray, (5, 5), 0)
    canny_image = cv2.Canny(gauss, 100, 120)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]), minLineLength=40, maxLineGap=100)
    final_image = draw_the_lines(img, lines)
    return final_image

# Update the video file path if necessary
video_path = "C:/Users/SRI LAVANYA/Downloads/video (3).mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    # If frame is read correctly ret is True, otherwise False
    if not ret:
        print("Error: Could not read frame.")
        break

    img = process(frame)
    cv2.imshow('Lane Lines Detection', img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''


#case -3
#github code
import numpy as np
import cv2

def process(image):
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold_low = 50
    threshold_high = 200
    image_canny = cv2.Canny(image_g, threshold_low, threshold_high)
    
    # Adjust vertices according to your video resolution if needed
    vertices = np.array([[(205, 1005), (400, 570), (800, 570), (1016, 707)]], dtype=np.int32)
    cropped_image = region_of_interest(image_canny, vertices)
    
    rho = 2           
    theta = np.pi / 180
    threshold = 10   
    min_line_len = 20
    max_line_gap = 20

    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_image = draw_the_lines(image, lines)
    return line_image

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    if lines is not None:
        line_image = np.zeros_like(img)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 20)
        img = cv2.addWeighted(img, 1, line_image, 1, 0)
    return img

# Ensure the video file path is correct
video_path = 'C:/Users/SRI LAVANYA/Downloads/video (3).mp4'

#video_path = "C:/Users/SRI LAVANYA/Downloads/lanes_clip.mp4"

output_path = './lines.avi'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# Initialize VideoWriter with appropriate codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
result = cv2.VideoWriter(output_path, fourcc, 20, size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    processed_frame = process(frame)
    result.write(processed_frame)

    # Optionally, display the frame
    cv2.imshow('Lane Lines Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()




'''import numpy as np
import cv2
import torch
import sys
import os
import random

# Add YOLOv5 repository to sys.path
yolov5_path = 'C:/Users/SRI LAVANYA/Desktop/files/deploying ma/yolov5'
sys.path.append(yolov5_path)

# Import YOLOv5
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device

# Initialize YOLOv5 model
device = select_device('')
model = DetectMultiBackend(yolov5_path + '/yolov5s.pt', device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt
img_size = 640

def process(image):
    # YOLOv5 Object Detection
    img = cv2.resize(image, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, image, label=label, color=[255, 0, 0], line_thickness=2)

    # Lane detection
    image_g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold_low = 50
    threshold_high = 200
    image_canny = cv2.Canny(image_g, threshold_low, threshold_high)
    
    # Adjust vertices according to your video resolution if needed
    vertices = np.array([[(205, 1005), (400, 570), (800, 570), (1016, 707)]], dtype=np.int32)
    cropped_image = region_of_interest(image_canny, vertices)
    
    rho = 2           
    theta = np.pi / 180
    threshold = 10   
    min_line_len = 20
    max_line_gap = 20

    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_image = draw_the_lines(image, lines)
    
    combined_image = cv2.addWeighted(image, 1, line_image, 1, 0)
    return combined_image

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    if lines is not None:
        line_image = np.zeros_like(img)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 20)
    return line_image

# Function to draw bounding box
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Ensure the video file path is correct
video_path = 'C:/Users/SRI LAVANYA/Downloads/video (3).mp4'
output_path = './lines.avi'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# Initialize VideoWriter with appropriate codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
result = cv2.VideoWriter(output_path, fourcc, 20, size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    processed_frame = process(frame)
    result.write(processed_frame)

    # Optionally, display the frame
    cv2.imshow('Lane Lines and Object Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()'''















#sucessful running code


'''import numpy as np
import cv2
import torch
import sys
import random
from pathlib import Path

# Add YOLOv5 repository to sys.path
yolov5_path = 'C:/Users/SRI LAVANYA/Desktop/files/deploying ma/yolov5'
sys.path.append(yolov5_path)

# Import YOLOv5
from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_coords)
from utils.torch_utils import select_device

# Initialize YOLOv5 model
device = select_device('')
model = DetectMultiBackend(yolov5_path + '/yolov5s.pt', device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt
img_size = 640

# Define colors for each class
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def process(image):
    # YOLOv5 Object Detection
    img = cv2.resize(image, (img_size, img_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.45, 0.5, classes=None, agnostic=False)  # Adjusted thresholds

    # Prepare the output image
    output_image = image.copy()

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], output_image.shape).round()

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, output_image, label=label, color=colors[int(cls)], line_thickness=2)

    # Lane detection
    image_g = cv2.cvtColor(output_image, cv2.COLOR_RGB2GRAY)
    threshold_low = 50
    threshold_high = 200
    image_canny = cv2.Canny(image_g, threshold_low, threshold_high)
    
    # Adjust vertices according to your video resolution if needed
    vertices = np.array([[(205, 1005), (400, 570), (800, 570), (1016, 707)]], dtype=np.int32)
    cropped_image = region_of_interest(image_canny, vertices)
    
    rho = 2           
    theta = np.pi / 180
    threshold = 10   
    min_line_len = 20
    max_line_gap = 20

    lines = cv2.HoughLinesP(cropped_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_image = draw_the_lines(image, lines)
    
    combined_image = cv2.addWeighted(output_image, 1, line_image, 1, 0)
    return combined_image

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # thinner lines
    return line_image

# Function to draw bounding box with different colors
def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = line_thickness or round(0.002 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

# Ensure the video file path is correct
video_path = 'C:/Users/SRI LAVANYA/Downloads/video (3).mp4'
output_path = './lines.avi'

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (frame_width, frame_height)

# Initialize VideoWriter with appropriate codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
result = cv2.VideoWriter(output_path, fourcc, 20, size)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    processed_frame = process(frame)
    result.write(processed_frame)

    # Optionally, display the frame
    cv2.imshow('Lane Lines and Object Detection', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
result.release()
cv2.destroyAllWindows()'''























