#A Gender and Age Detection program for images

import cv2
import os
import numpy as np

def check_file_exists(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

def load_models():
    # Check if models exist
    model_files = {
        'face': ['models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel'],
        'age': ['models/age_deploy.prototxt', 'models/age_net.caffemodel'],
        'gender': ['models/gender_deploy.prototxt', 'models/gender_net.caffemodel']
    }
    
    # Check for alternative face detection model
    if not os.path.exists(model_files['face'][0]) or not os.path.exists(model_files['face'][1]):
        if os.path.exists('opencv_face_detector.pbtxt') and os.path.exists('opencv_face_detector_uint8.pb'):
            print("Using alternative face detection model")
            model_files['face'] = ['opencv_face_detector.pbtxt', 'opencv_face_detector_uint8.pb']
    
    # Load face detection model
    try:
        face_net = cv2.dnn.readNet(model_files['face'][0], model_files['face'][1])
        print(f"Loaded face detection model: {model_files['face'][0]}, {model_files['face'][1]}")
    except Exception as e:
        print(f"Error loading face detection model: {e}")
        return None, None, None
    
    # Load age detection model
    try:
        age_net = cv2.dnn.readNet(model_files['age'][0], model_files['age'][1])
        print(f"Loaded age detection model: {model_files['age'][0]}, {model_files['age'][1]}")
    except Exception as e:
        print(f"Error loading age detection model: {e}")
        return None, None, None
    
    # Load gender detection model
    try:
        gender_net = cv2.dnn.readNet(model_files['gender'][0], model_files['gender'][1])
        print(f"Loaded gender detection model: {model_files['gender'][0]}, {model_files['gender'][1]}")
    except Exception as e:
        print(f"Error loading gender detection model: {e}")
        return None, None, None
    
    return face_net, age_net, gender_net

def detect_age_gender(image_path, face_net, age_net, gender_net):
    # Define the lists of age and gender labels
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    gender_list = ['Male', 'Female']
    
    # Read the image
    print(f"Reading image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return None
    
    # Get frame dimensions
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    print(f"Image dimensions: {frame_width}x{frame_height}")
    
    # Create a blob from the frame
    face_blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        [104, 117, 123], True, False
    )
    
    # Detect faces
    print("Detecting faces...")
    face_net.setInput(face_blob)
    face_detections = face_net.forward()
    
    faces_found = 0
    
    # Process each detected face
    for i in range(face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]
        
        if confidence > 0.5:
            faces_found += 1
            print(f"Face #{faces_found} detected with confidence: {confidence:.2f}")
            
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
                print("Invalid face dimensions, skipping")
                continue
                
            # Prepare face for gender and age detection
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                [78.4263377603, 87.7689143744, 114.895847746],
                swapRB=False
            )
            
            # Gender detection
            print("Detecting gender...")
            gender_net.setInput(face_blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            
            # Age detection
            print("Detecting age...")
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
            
            print(f"Result: {label}")
    
    if faces_found == 0:
        print("No faces detected in the image")
    else:
        print(f"Total faces detected: {faces_found}")
    
    # Save the result
    output_path = f"result_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, frame)
    print(f"Result saved to: {output_path}")
    
    # Display the result
    cv2.imshow("Age and Gender Detection", frame)
    print("Press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return frame

def main():
    # Check if image file is provided
    image_path = "girl1.jpg"  # Default image
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        print("Available images:")
        for file in os.listdir("."):
            if file.endswith(".jpg") or file.endswith(".png"):
                print(f"  {file}")
        return
    
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
        print("Models directory created")
    
    # Load models
    print("Loading models...")
    face_net, age_net, gender_net = load_models()
    if face_net is None or age_net is None or gender_net is None:
        print("Failed to load one or more models")
        return
    
    # Process image
    detect_age_gender(image_path, face_net, age_net, gender_net)

if __name__ == "__main__":
    main()
