import cv2
import os
import sys
import numpy as np

# Print Python and OpenCV versions for debugging
print(f"Python version: {sys.version}")
print(f"OpenCV version: {cv2.__version__}")

# Define model paths
face_model_path = 'opencv_face_detector_uint8.pb'
face_config_path = 'opencv_face_detector.pbtxt'
age_model_path = 'models/age_net.caffemodel'
age_config_path = 'models/age_deploy.prototxt'
gender_model_path = 'models/gender_net.caffemodel'
gender_config_path = 'models/gender_deploy.prototxt'

# Check if model files exist
print("\nChecking model files:")
for path in [face_model_path, face_config_path, age_model_path, age_config_path, gender_model_path, gender_config_path]:
    if os.path.exists(path):
        print(f"  {path}: Found")
    else:
        print(f"  {path}: Not found")

# Age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

try:
    # Load models
    print("\nLoading models...")
    face_net = cv2.dnn.readNet(face_model_path, face_config_path)
    age_net = cv2.dnn.readNet(age_config_path, age_model_path)
    gender_net = cv2.dnn.readNet(gender_config_path, gender_model_path)
    print("Models loaded successfully")
    
    # Use webcam or image
    use_webcam = False
    
    if use_webcam:
        # Initialize webcam
        print("\nInitializing webcam...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            sys.exit(1)
        print("Webcam initialized successfully")
        
        print("\nPress 'q' to quit")
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame here
            # ...
            
            # Display result
            cv2.imshow("Age and Gender Detection", frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
    else:
        # Use image file
        image_path = "girl1.jpg"  # Change this to your image file
        print(f"\nProcessing image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            sys.exit(1)
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            sys.exit(1)
        
        print(f"Image loaded: {frame.shape}")
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Create a blob from the frame for face detection
        face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        
        # Detect faces
        print("Detecting faces...")
        face_net.setInput(face_blob)
        face_detections = face_net.forward()
        
        # Process each detected face
        faces_found = 0
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
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_width, x2), min(frame_height, y2)
                
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

except Exception as e:
    print(f"\nError: {e}")
