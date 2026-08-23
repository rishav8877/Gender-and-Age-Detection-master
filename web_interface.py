import cv2
import os
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
import base64

app = Flask(__name__)

# Define model paths
face_model_path = 'opencv_face_detector_uint8.pb'
face_config_path = 'opencv_face_detector.pbtxt'
age_model_path = 'models/age_net.caffemodel'
age_config_path = 'models/age_deploy.prototxt'
gender_model_path = 'models/gender_net.caffemodel'
gender_config_path = 'models/gender_deploy.prototxt'

# Age and gender labels
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

# Load models
face_net = cv2.dnn.readNet(face_model_path, face_config_path)
age_net = cv2.dnn.readNet(age_config_path, age_model_path)
gender_net = cv2.dnn.readNet(gender_config_path, gender_model_path)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Create HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Gender and Age Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            text-align: center;
        }
        h1 {
            color: #333;
        }
        .image-container {
            margin: 20px 0;
        }
        img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
        }
        select, button {
            padding: 10px;
            margin: 10px;
            font-size: 16px;
        }
        .result {
            margin-top: 20px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h1>Gender and Age Detection</h1>
    
    <div>
        <select id="imageSelect">
            <option value="">Select an image</option>
            {% for image in images %}
            <option value="{{ image }}">{{ image }}</option>
            {% endfor %}
        </select>
        <button onclick="processImage()">Process Image</button>
    </div>
    
    <div class="image-container">
        <img id="selectedImage" src="" alt="Select an image" style="display: none;">
    </div>
    
    <div class="image-container">
        <img id="resultImage" src="" alt="Result will appear here" style="display: none;">
    </div>
    
    <div id="result" class="result" style="display: none;"></div>
    
    <script>
        function processImage() {
            const imageSelect = document.getElementById('imageSelect');
            const selectedImage = document.getElementById('selectedImage');
            const resultImage = document.getElementById('resultImage');
            const resultDiv = document.getElementById('result');
            
            if (imageSelect.value) {
                // Show selected image
                selectedImage.src = `/image/${imageSelect.value}`;
                selectedImage.style.display = 'block';
                
                // Process image
                resultDiv.innerHTML = 'Processing...';
                resultDiv.style.display = 'block';
                resultImage.style.display = 'none';
                
                fetch(`/process/${imageSelect.value}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            resultImage.src = `data:image/jpeg;base64,${data.image}`;
                            resultImage.style.display = 'block';
                            resultDiv.innerHTML = data.message;
                        } else {
                            resultDiv.innerHTML = `Error: ${data.message}`;
                        }
                    })
                    .catch(error => {
                        resultDiv.innerHTML = `Error: ${error.message}`;
                    });
            }
        }
        
        // Update image when selection changes
        document.getElementById('imageSelect').addEventListener('change', function() {
            const selectedImage = document.getElementById('selectedImage');
            const resultImage = document.getElementById('resultImage');
            const resultDiv = document.getElementById('result');
            
            if (this.value) {
                selectedImage.src = `/image/${this.value}`;
                selectedImage.style.display = 'block';
                resultImage.style.display = 'none';
                resultDiv.style.display = 'none';
            } else {
                selectedImage.style.display = 'none';
            }
        });
    </script>
</body>
</html>
''')

@app.route('/')
def index():
    # Get list of image files
    images = [f for f in os.listdir('.') if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('index.html', images=images)

@app.route('/image/<filename>')
def serve_image(filename):
    return Response(open(filename, 'rb').read(), mimetype='image/jpeg')

@app.route('/process/<filename>')
def process_image(filename):
    try:
        # Read image
        frame = cv2.imread(filename)
        if frame is None:
            return jsonify({'success': False, 'message': f'Could not read image {filename}'})
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Create a blob from the frame for face detection
        face_blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        
        # Detect faces
        face_net.setInput(face_blob)
        face_detections = face_net.forward()
        
        # Process each detected face
        faces_found = 0
        results = []
        
        for i in range(face_detections.shape[2]):
            confidence = face_detections[0, 0, i, 2]
            
            if confidence > 0.5:
                faces_found += 1
                
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
                
                results.append({'gender': gender, 'age': age})
        
        if faces_found == 0:
            return jsonify({'success': False, 'message': 'No faces detected in the image'})
        
        # Convert result image to base64
        _, buffer = cv2.imencode('.jpg', frame)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Create result message
        message = f"Found {faces_found} face(s):<br>"
        for i, result in enumerate(results):
            message += f"Face #{i+1}: {result['gender']}, {result['age']}<br>"
        
        return jsonify({
            'success': True,
            'message': message,
            'image': img_str,
            'faces': faces_found,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    print("Starting Gender and Age Detection Web Interface...")
    print("Open your browser and navigate to http://127.0.0.1:5000")
    app.run(debug=True)
