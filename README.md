# Face Recognition System Using AI

## Overview

This project is an AI-based Face Recognition System built using classical computer vision and machine learning techniques. The system captures facial images through a webcam, extracts Local Binary Pattern Histogram (LBPH) features, trains a Random Forest classifier, and performs real-time face recognition through a Flask web interface.

The project is designed to work completely offline without any cloud services and can be used for applications such as:

* Attendance systems
* Access control systems
* Student identification
* Small-scale security systems

---

## Features

* Capture face datasets using webcam
* Face detection using Haar Cascade Classifier
* Custom implementation of LBP and LBPH feature extraction
* Random Forest based face classification
* Real-time face recognition
* Unknown face detection using confidence thresholding
* Flask web interface
* Model saving and loading functionality
* Confusion matrix generation for performance evaluation

---

## Project Structure

```text
Face-Recognition-System/
│── app.py
│── config.py
│── lbp.py
│── dataset_creator.py
│── train_model.py
│── recognize.py
│── model.pkl
│── requirements.txt
│── README.md
│── dataset/
│── templates/
│    └── index.html
│── static/
```

---

## Technologies Used

### Programming Language

* Python

### Libraries and Frameworks

* Flask
* OpenCV
* NumPy
* Scikit-learn
* Matplotlib

### Machine Learning Techniques

* Local Binary Pattern (LBP)
* Local Binary Pattern Histogram (LBPH)
* Random Forest Classifier

### Face Detection

* Haar Cascade Classifier

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/Face-Recognition-System.git
cd Face-Recognition-System
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Project

Start Flask application:

```bash
python app.py
```

Open browser:

```text
http://127.0.0.1:5000
```

---

## How to Use

### Step 1: Register New Person

* Enter person name
* Click Capture Dataset
* Webcam starts automatically
* Face images will be captured and saved

### Step 2: Train Model

* Click Train Model
* System extracts features
* Random Forest model is trained
* Model file is saved

### Step 3: Start Recognition

* Click Start Recognition
* Webcam starts
* Real-time face recognition begins

---

## Performance

| Operation                | Performance |
| ------------------------ | ----------- |
| Haar Cascade Detection   | ~20 ms      |
| LBP Computation          | ~140 ms     |
| LBPH Feature Extraction  | ~40 ms      |
| Random Forest Prediction | ~5 ms       |
| Total Recognition Speed  | ~5 FPS      |
| Classification Accuracy  | 92.5%       |

---

## Future Improvements

* Add liveness detection
* Improve recognition speed
* Add database integration
* Build attendance management system
* Multi-camera support
* Deploy using cloud services

---

## Author

Adarsh Kumar

---

## License

This project is created for educational and academic purposes.
