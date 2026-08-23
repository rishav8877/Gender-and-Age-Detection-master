import cv2
import os

# Simple script to display an image using OpenCV

try:
    # List available image files
    image_files = [f for f in os.listdir('.') if f.endswith('.jpg') or f.endswith('.png')]
    print(f"Available images: {image_files}")
    
    if not image_files:
        print("No image files found")
        exit()
    
    # Use the first image file found
    image_path = image_files[0]
    print(f"Opening image: {image_path}")
    
    # Read and display the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        exit()
    
    print(f"Image loaded successfully: {img.shape}")
    
    # Display the image
    cv2.imshow("Image Display Test", img)
    print("Press any key to exit")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Image displayed successfully")
    
except Exception as e:
    print(f"Error: {e}")
