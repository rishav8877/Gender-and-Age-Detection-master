import cv2
import os

def test_opencv():
    print("OpenCV version:", cv2.__version__)
    
    # Test image loading
    image_files = [f for f in os.listdir('.') if f.endswith('.jpg') or f.endswith('.png')]
    if image_files:
        test_image = image_files[0]
        print(f"Testing image loading with: {test_image}")
        img = cv2.imread(test_image)
        if img is not None:
            print(f"Successfully loaded image: {img.shape}")
            
            # Save a test output
            output_path = "test_output.jpg"
            cv2.imwrite(output_path, img)
            print(f"Saved test image to: {output_path}")
        else:
            print(f"Failed to load image: {test_image}")
    else:
        print("No image files found for testing")
    
    # Test webcam
    print("\nTesting webcam access...")
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Webcam opened successfully")
            ret, frame = cap.read()
            if ret:
                print(f"Successfully captured frame: {frame.shape}")
                # Save a test frame
                cv2.imwrite("webcam_test.jpg", frame)
                print("Saved webcam test frame to: webcam_test.jpg")
            else:
                print("Failed to capture frame from webcam")
            cap.release()
        else:
            print("Failed to open webcam")
    except Exception as e:
        print(f"Error accessing webcam: {e}")
    
    # Test model loading
    print("\nTesting model loading...")
    model_files = {
        'face': ['models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel'],
        'alt_face': ['opencv_face_detector.pbtxt', 'opencv_face_detector_uint8.pb'],
        'age': ['models/age_deploy.prototxt', 'models/age_net.caffemodel'],
        'gender': ['models/gender_deploy.prototxt', 'models/gender_net.caffemodel']
    }
    
    for model_type, files in model_files.items():
        print(f"\nChecking {model_type} model:")
        for file in files:
            if os.path.exists(file):
                print(f"  {file}: Found")
                file_size = os.path.getsize(file) / (1024 * 1024)  # Convert to MB
                print(f"  Size: {file_size:.2f} MB")
            else:
                print(f"  {file}: Not found")
        
        # Try to load the model if both files exist
        if all(os.path.exists(file) for file in files):
            try:
                net = cv2.dnn.readNet(files[0], files[1])
                print(f"  Successfully loaded {model_type} model")
            except Exception as e:
                print(f"  Error loading {model_type} model: {e}")

if __name__ == "__main__":
    test_opencv()
