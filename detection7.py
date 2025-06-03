#C:\Users\SRI LAVANYA\Desktop\files\deploying ma
#

import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import cv2
import torch
import sys
import random
from pathlib import Path

# Set the YOLOv5 path
yolov5_path = Path('C:/Users/SRI LAVANYA/Desktop/files/deploying ma/yolov5')
sys.path.append(str(yolov5_path))

# Import YOLOv5
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

# Initialize YOLOv5 model
device = select_device('')
model = DetectMultiBackend(yolov5_path / 'yolov5s.pt', device=device, dnn=False)
stride, names, pt = model.stride, model.names, model.pt
img_size = 640

# Define colors for each class
colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

def process_frame(image, detect_objects=False):
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

    if detect_objects:
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
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  # thicker lines
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

def main():
    st.title("Lane and Object Detection")
    menu = ["Home", "Lane Detection", "Lane and Object Detection", "About"]
    
    # Create the menu in the sidebar
    with st.sidebar:
        choice = option_menu("Menu", menu, icons=['house', 'road', 'binoculars', 'info-circle'], menu_icon="cast", default_index=0, orientation="vertical")

    if choice == "Home":
        st.markdown("## Welcome to the predictive modelling...!")
        st.write("Our system helps in predicting lane conditions using advanced machine learning models. Explore the app to know more!")
        st.markdown("""
        <div style='text-align: center;'>
            <h3>Explore Our Features:</h3>
            <ul style='list-style-type: none; padding: 0;'>
                <li style='margin: 10px 0;'><button id='traffic-button' style='padding: 10px 20px; font-size: 16px; background-color: #4CAF50; color: white; border: none; border-radius: 5px;'>Lane detection</button></li>
                <li style='margin: 10px 0;'><button id='data-button' style='padding: 10px 20px; font-size: 16px; background-color: #008CBA; color: white; border: none; border-radius: 5px;'>Object detection</button></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <script>
        document.getElementById('traffic-button').onclick = function() {
            // Your logic to handle Traffic Prediction button click
            // For example, you can redirect to a different page or display a message
            alert('Redirecting to Traffic Prediction...');
        };
        document.getElementById('data-button').onclick = function() {
            // Your logic to handle Data Visualiser button click
            // For example, you can redirect to a different page or display a message
            alert('Redirecting to Data Visualiser...');
        };
        </script>
        """, unsafe_allow_html=True)
        st.subheader("📫 Connect with the Developer")
        
        if st.button("Say Hi 👋"):
            st.snow()
            st.success("Thanks for checking out the project! Feel free to reach out 😄")


    elif choice == "Lane Detection":
        st.subheader("Lane Detection")
        video_path = st.text_input("Enter video file path", "C:/Users/SRI LAVANYA/Downloads/video (3).mp4")
        if st.button("Start Lane Detection"):
            process_video(video_path, detect_objects=False)

    elif choice == "Lane and Object Detection":
        st.subheader("Lane and Object Detection")
        video_path = st.text_input("Enter video file path", "C:/Users/SRI LAVANYA/Desktop/files/deploying ma/yolov5-object-tracking/2.mp4")
        if st.button("Start Lane and Object Detection"):
            process_video(video_path, detect_objects=True)
    elif choice == "About":
        st.title("👩‍💻 About This Project")

        st.write("Welcome to the interactive About section! Let’s explore what this app offers 👇")

        tab1, tab2, tab3 = st.tabs(["🔍 Overview", "📦 Features", "🧠 Tech Stack"])

        with tab1:
            st.image("https://i.imgur.com/U4H0m2Z.png", use_container_width=True, caption="AI meets Lane & Object Detection")

            st.markdown("""
            This application performs **real-time lane and object detection** using computer vision techniques.
            
            It combines **YOLOv5** for object detection with **Hough Transform** for lane detection, allowing us to:
            - Identify road objects like vehicles or pedestrians 🚗
            - Highlight lane boundaries in traffic scenarios 🛣️
            - Process and display video frames in real-time 🎥
            """)

        with tab2:
            st.success("✅ Real-time video frame processing")
            st.info("🚗 Object detection with confidence scores")
            st.warning("🛣️ Lane detection with edge detection + Hough Lines")
            st.code("""
            if detect_objects:
                processed_frame = process_frame(frame, detect_objects=True)
            """, language="python")

        with tab3:
            st.metric(label="Framework", value="YOLOv5 + OpenCV")
            st.metric(label="Interface", value="Streamlit")
            st.metric(label="Language", value="Python 3.10")

        st.markdown("---")
        st.subheader("📫 Connect with the Developer")
        st.markdown("Made with ❤️ by **SriLavanya Pallampati**")
        
def process_video(video_path, detect_objects):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return

    stop_button = st.button("Stop Video Processing")

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, detect_objects)
        stframe.image(processed_frame, channels="BGR")

        if stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()
    st.write("Processing Stopped")

if __name__ == "__main__":
    main()
