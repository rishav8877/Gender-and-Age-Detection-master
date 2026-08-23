# Gender and Age Detection for images

import cv2
import os
import numpy as np

# Image to process
image_path = "girl1.jpg"  # You can change this to any other image file

# Load models
print("Loading models...")

# Face detection model
face_proto = "opencv_face_detector.pbtxt"
face_model = "opencv_face_detector_uint8.pb"
face_net = cv2.dnn.readNet(face_proto, face_model)

# Age detection model
age_proto = "models/age_deploy.prototxt"
age_model = "models/age_net.caffemodel"
age_net = cv2.dnn.readNet(age_proto, age_model)

# Gender detection model
gender_proto = "models/gender_deploy.prototxt"
gender_model = "models/gender_net.caffemodel"
gender_net = cv2.dnn.readNet(gender_proto, gender_model)

# Define age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

print(f"Processing image: {image_path}")

# Read image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not read image {image_path}")
    exit()

# Get image dimensions
height, width = img.shape[:2]
print(f"Image dimensions: {width}x{height}")

# Create a blob from the image for face detection
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)

# Detect faces
face_net.setInput(blob)
detections = face_net.forward()

# Process each detected face
faces_found = 0
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    
    if confidence > 0.5:
        faces_found += 1
        print(f"Face #{faces_found} detected with confidence: {confidence:.2f}")
        
        # Get face box coordinates
        x1 = int(detections[0, 0, i, 3] * width)
        y1 = int(detections[0, 0, i, 4] * height)
        x2 = int(detections[0, 0, i, 5] * width)
        y2 = int(detections[0, 0, i, 6] * height)
        
        # Ensure coordinates are within image boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        
        # Extract face ROI
        face = img[y1:y2, x1:x2]
        
        if face.shape[0] == 0 or face.shape[1] == 0:
            print("Invalid face dimensions, skipping")
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
        
        # Draw results on image
        color = (0, 255, 0) if gender == "Male" else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Display gender and age
        label = f"{gender}, {age}"
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.putText(img, label, (x1, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        print(f"Result: {label}")

if faces_found == 0:
    print("No faces detected in the image")
else:
    print(f"Total faces detected: {faces_found}")

# Save the result
output_path = f"result_{os.path.basename(image_path)}"
cv2.imwrite(output_path, img)
print(f"Result saved to: {output_path}")

print("Processing completed")
