# Gender and Age Detection System

A real-time **Gender and Age Detection System** built using **Python, OpenCV, Deep Learning, and Flask**. The project detects faces from images or webcam input and predicts the **gender** and **approximate age range** of each detected person.

---

## Project Overview

This project uses OpenCV's **Deep Neural Network (DNN)** module and pre-trained deep learning models to perform face detection, gender classification, and age estimation.

The system can process images as well as real-time webcam/video input. Detected faces are highlighted with bounding boxes along with the predicted gender and age range.

---

## Features

* Real-time face detection using webcam
* Multiple face detection
* Gender classification
* Approximate age range prediction
* Image-based age and gender detection
* Flask-based web interface
* Pre-trained deep learning models
* Real-time prediction using OpenCV DNN
* Visual display of prediction results
* Basic OpenCV and model testing scripts

---

##  Technologies Used

* **Python**
* **OpenCV**
* **OpenCV DNN**
* **NumPy**
* **Flask**
* **Deep Learning**
* **Computer Vision**
* **Pre-trained CNN Models**

---

## Age Categories

The model predicts one of the following approximate age ranges:

```text
0-2
4-6
8-12
15-20
25-32
38-43
48-53
60-100
```

---

## Gender Categories

The system classifies detected faces into:

```text
Male
Female
```

---

## Project Structure

```text
Gender-and-Age-Detection/
│
├── models/
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   ├── gender_net.caffemodel
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
│
├── detect.py
├── simple_detect.py
├── web_interface.py
├── test_basic.py
├── test_opencv.py
├── requirements.txt
├── requirements_web.txt
└── README.md
```

> **Note:** Some model files may need to be downloaded separately and placed inside the `models` directory according to the paths used by the scripts.

---

## Requirements

Before running the project, make sure you have:

* Python 3.6 or higher
* Webcam (for real-time detection)
* pip
* Required Python libraries

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Gender-and-Age-Detection.git
```

Move into the project directory:

```bash
cd Gender-and-Age-Detection
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

For the main application:

```bash
pip install -r requirements.txt
```

For the Flask web interface:

```bash
pip install -r requirements_web.txt
```

---

## Model Files

The project requires pre-trained deep learning models for:

1. Face Detection
2. Age Detection
3. Gender Detection

Place the required model files inside the appropriate directories.

---


### Face Detection

```text
deploy.prototxt
res10_300x300_ssd_iter_140000.caffemodel
```

or the alternative OpenCV face detector:

```text
opencv_face_detector.pbtxt
opencv_face_detector_uint8.pb
```

---


### Age Detection

```text
age_deploy.prototxt
age_net.caffemodel
```

---

### Gender Detection

```text
gender_deploy.prototxt
gender_net.caffemodel
```

---

## Running the Project

### Real-Time Webcam Detection

Run the main detection program:

```bash
python detect.py
```

The application will open the webcam and detect faces in real time.

Press:

```text
q
```

to close the application.

---


## Image Detection

The project also supports image-based detection.

Run:

```bash
python simple_detect.py
```

You can modify the image path inside the script to process a different image.

Example:

```python
image_path = "girl1.jpg"
```

---

##  Web Interface

The project also contains a Flask-based web interface.

Run:

```bash
python web_interface.py
```

After starting the Flask application, open the local address shown in the terminal, usually:

```text
http://127.0.0.1:5000/
```

The web interface allows you to select an image and process it for age and gender prediction.

---

## Examples 

<p><b>NOTE:- I downloaded the images from Google. You can use any image for testing.</b></p

<img width="480" height="720" alt="girl1" src="https://github.com/user-attachments/assets/2b677c4a-0a11-4d9f-bc12-58eafffed189" />
<img width="623" height="395" alt="girl2" src="https://github.com/user-attachments/assets/1f508e7b-f4f2-4e0f-bc60-22042a44075f" />
<img width="960" height="645" alt="kid1" src="https://github.com/user-attachments/assets/acc6d5f5-9705-4e3a-ab05-ca47b5e70992" />
<img width="800" height="687" alt="kid2" src="https://github.com/user-attachments/assets/479795f5-7b48-4ed0-aeed-a8da2f6938f0" />
<img width="612" height="408" alt="man1" src="https://github.com/user-attachments/assets/7a466778-0817-46a3-ac0d-8a757c9c8d78" />
<img width="1000" height="667" alt="man2" src="https://github.com/user-attachments/assets/496d88cc-545b-4f03-92fd-bb8c6513040c" />

---

##  How the System Works

The complete detection pipeline works as follows:

```text
Input Image / Webcam
        ↓
   Face Detection
        ↓
   Extract Face ROI
        ↓
 ┌───────────────┐
 │               │
 ↓               ↓
Gender Model   Age Model
 │               │
 ↓               ↓
Gender        Age Range
 │               │
 └───────┬───────┘
         ↓
 Display Prediction
```

### Step 1 — Capture Input

The system receives an image or video frame from the webcam.

### Step 2 — Detect Faces

The OpenCV DNN face detection model identifies faces in the input.

### Step 3 — Extract Face Region

For every detected face, the system extracts the corresponding region of interest (ROI).

### Step 4 — Gender Prediction

The extracted face is passed to the gender classification model.

### Step 5 — Age Prediction

The same face region is passed to the age prediction model to estimate the person's age range.

### Step 6 — Display Results

The predicted gender and age range are displayed along with a bounding box around the detected face.

---


##  Testing

The project contains testing scripts for checking the Python environment, OpenCV installation, webcam access, image loading, and model availability.

Run the basic test:

```bash
python test_basic.py
```

Run the OpenCV and model test:

```bash
python test_opencv.py
```

These scripts can help identify missing dependencies, unavailable model files, or webcam access problems.

---


## Example Output

The application displays predictions in the following format:

```text
Male, (25-32)
```

or

```text
Female, (15-20)
```

The prediction is displayed directly on the detected face.

---

## Limitations

* Age prediction is an **approximation**, not an exact age.
* Predictions can be affected by lighting, image quality, camera angle, and face orientation.
* Gender classification may not always be accurate.
* Performance depends on the computer's CPU/GPU and camera quality.
* The pre-trained models may perform differently on faces that differ significantly from their training data.

---

## Privacy Note

This project processes images and webcam frames locally when run on your computer. It is recommended to obtain appropriate consent before using facial analysis on other people.

---

## Future Improvements

Possible improvements include:

* Improve prediction accuracy using newer deep learning models
* Add support for GPU acceleration
* Add confidence scores for predictions
* Support video file input
* Improve the Flask web interface
* Add image upload functionality
* Add real-time statistics and analytics
* Add support for more advanced face recognition techniques
* Deploy the web application online

---

## Applications

This project demonstrates concepts that can be useful for:

* Computer Vision
* Deep Learning
* Facial Analysis
* Human-Computer Interaction
* AI-based Image Processing
* Real-Time Video Processing
* Python AI/ML Applications

---

## Author

**Rishav Kumar**

MCA Student | Software Development & AI/ML Enthusiast

---

## License

This project is intended for educational and demonstration purposes. Please check the licenses of the original OpenCV models and any third-party resources before redistributing them.

---

## Support

If you find this project useful, consider giving the repository on GitHub.
