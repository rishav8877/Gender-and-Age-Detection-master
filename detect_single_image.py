# Gender and Age Detection for a single image

import cv2
import os
import sys

# Print diagnostic information
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")

# Define the image to process
image_file = "girl1.jpg"  # Change this to any image file you want to process
print(f"\nProcessing image: {image_file}")

# Check if the image exists
if not os.path.exists(image_file):
    print(f"Error: Image file not found: {image_file}")
    available_images = [f for f in os.listdir('.') if f.endswith('.jpg') or f.endswith('.png')]
    if available_images:
        print(f"Available images: {available_images}")
        image_file = available_images[0]
        print(f"Using {image_file} instead")
    else:
        print("No image files found")
        sys.exit(1)

# Define model paths and check if they exist
print("\nChecking model files:")
model_files = {
    'face': ['opencv_face_detector.pbtxt', 'opencv_face_detector_uint8.pb'],
    'age': ['models/age_deploy.prototxt', 'models/age_net.caffemodel'],
    'gender': ['models/gender_deploy.prototxt', 'models/gender_net.caffemodel']
}

for model_type, files in model_files.items():
    print(f"\n{model_type.capitalize()} detection model:")
    for file in files:
        if os.path.exists(file):
            print(f"  {file}: Found")
        else:
            print(f"  {file}: Not found")

# Load models
print("\nLoading models...")
try:
    face_net = cv2.dnn.readNet(model_files['face'][0], model_files['face'][1])
    print("Face detection model loaded successfully")
    
    age_net = cv2.dnn.readNet(model_files['age'][0], model_files['age'][1])
    print("Age detection model loaded successfully")
    
    gender_net = cv2.dnn.readNet(model_files['gender'][0], model_files['gender'][1])
    print("Gender detection model loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

# Define age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Read the image
print(f"\nReading image: {image_file}")
try:
    frame = cv2.imread(image_file)
    if frame is None:
        print(f"Error: Could not read image {image_file}")
        sys.exit(1)
    print(f"Image loaded successfully: {frame.shape}")
except Exception as e:
    print(f"Error reading image: {e}")
    sys.exit(1)

# Get frame dimensions
frame_height, frame_width = frame.shape[:2]

# Create a blob from the frame for face detection
print("\nPreparing image for face detection...")
try:
    face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
    print("Image prepared successfully")
except Exception as e:
    print(f"Error preparing image: {e}")
    sys.exit(1)

# Detect faces
print("\nDetecting faces...")
try:
    face_net.setInput(face_blob)
    face_detections = face_net.forward()
    print("Face detection completed")
except Exception as e:
    print(f"Error detecting faces: {e}")
    sys.exit(1)

# Process each detected face
faces_found = 0
print("\nProcessing detected faces:")

for i in range(face_detections.shape[2]):
    confidence = face_detections[0, 0, i, 2]
    
    if confidence > 0.5:
        faces_found += 1
        print(f"\nFace #{faces_found} detected with confidence: {confidence:.2f}")
        
        # Get face box coordinates
        x1 = int(face_detections[0, 0, i, 3] * frame_width)
        y1 = int(face_detections[0, 0, i, 4] * frame_height)
        x2 = int(face_detections[0, 0, i, 5] * frame_width)
        y2 = int(face_detections[0, 0, i, 6] * frame_height)
        
        # Ensure coordinates are within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_width, x2), min(frame_height, y2)
        print(f"Face coordinates: ({x1}, {y1}) to ({x2}, {y2})")
        
        # Extract face ROI
        face = frame[y1:y2, x1:x2]
        
        if face.shape[0] == 0 or face.shape[1] == 0:
            print("Invalid face dimensions, skipping")
            continue
        
        print(f"Face dimensions: {face.shape}")
        
        # Prepare face for gender and age detection
        try:
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                [78.4263377603, 87.7689143744, 114.895847746],
                swapRB=False
            )
            print("Face prepared for gender and age detection")
        except Exception as e:
            print(f"Error preparing face for gender and age detection: {e}")
            continue
        
        # Gender detection
        try:
            print("\nDetecting gender...")
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print(f"Gender detected: {gender}")
        except Exception as e:
            print(f"Error detecting gender: {e}")
            continue
        
        # Age detection
        try:
            print("\nDetecting age...")
            age_net.setInput(face_blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print(f"Age detected: {age}")
        except Exception as e:
            print(f"Error detecting age: {e}")
            continue
        
        # Draw results on frame
        color = (0, 255, 0) if gender == "Male" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Display gender and age
        label = f"{gender}, {age}"
        y = y1 - 10 if y1 - 10 > 10 else y1 + 10
        cv2.putText(frame, label, (x1, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        print(f"Result: {label}")

if faces_found == 0:
    print("\nNo faces detected in the image")
else:
    print(f"\nTotal faces detected: {faces_found}")

# Save the result
output_path = f"result_{os.path.basename(image_file)}"
try:
    cv2.imwrite(output_path, frame)
    print(f"\nResult saved to: {output_path}")
except Exception as e:
    print(f"Error saving result: {e}")

print("\nProcessing completed")
