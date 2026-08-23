#A Gender and Age Detection program 

import cv2
import argparse
import os
import time
import numpy as np

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes

def process_with_rate_limit(net, blob, operation_name=""):
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            net.setInput(blob)
            predictions = net.forward()
            return predictions
        except Exception as e:
            if "resource_exhausted" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"Rate limit hit for {operation_name}. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Max retries reached for {operation_name}. Please try again later.")
                    raise
            else:
                raise

def load_models():
    # Load face detection model
    face_net = cv2.dnn.readNet(
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000.caffemodel"
    )
    
    # Load age detection model
    age_net = cv2.dnn.readNet(
        "models/age_deploy.prototxt",
        "models/age_net.caffemodel"
    )
    
    # Load gender detection model
    gender_net = cv2.dnn.readNet(
        "models/gender_deploy.prototxt",
        "models/gender_net.caffemodel"
    )
    
    return face_net, age_net, gender_net

def detect_age_gender(frame, face_net, age_net, gender_net):
    # Define the lists of age and gender labels
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']
    
    # Get frame dimensions
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    # Create a blob from the frame
    face_blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        [104, 117, 123], True, False
    )
    
    # Detect faces
    face_net.setInput(face_blob)
    face_detections = face_net.forward()
    
    # Process each detected face
    for i in range(face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        
        if confidence > 0.5:
            # Get face box coordinates
            x1 = int(face_detections[0, 0, i, 3] * frame_width)
            y1 = int(face_detections[0, 0, i, 4] * frame_height)
            x2 = int(face_detections[0, 0, i, 5] * frame_width)
            y2 = int(face_detections[0, 0, i, 6] * frame_height)
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_width, x2)
            y2 = min(frame_height, y2)
            
            # Extract face ROI
            face = frame[y1:y2, x1:x2]
            
            if face.shape[0] == 0 or face.shape[1] == 0:
                continue
                
            # Prepare face for gender and age detection
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                [78.4263377603, 87.7689143744, 114.895847746],
                swapRB=False
            )
            
            # Gender detection
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            
            # Age detection
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            
            # Draw results on frame
            color = (0, 255, 0) if gender == "Male" else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Display gender and age
            label = f"{gender}, {age}"
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(frame, label, (x1, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Please download the required model files and place them in the 'models' directory:")
        print("1. deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel (face detection)")
        print("2. age_deploy.prototxt and age_net.caffemodel (age detection)")
        print("3. gender_deploy.prototxt and gender_net.caffemodel (gender detection)")
        return
    
    # Load models
    try:
        face_net, age_net, gender_net = load_models()
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure all model files are present in the 'models' directory")
        return
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture device")
        return
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process frame
        output_frame = detect_age_gender(frame, face_net, age_net, gender_net)
        
        # Display result
        cv2.imshow("Age and Gender Detection", output_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
