
---

# Emotion Detection 🎭

A real-time facial emotion detection system utilizing deep learning techniques with TensorFlow and OpenCV. This project classifies human emotions from facial expressions captured via webcam or images.

![Emotion Detection Demo](emoition_detection.png)

## 📌 Features

* 🎯 Detects seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.
* 🧠 Employs a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.
* 📷 Real-time emotion recognition through webcam integration.
* 🖼️ Supports emotion detection in static images.
* 🛠️ Modular codebase for training, evaluation, and testing.

## 🧰 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Sourodyuti/Emotion-Detection.git
   cd Emotion-Detection
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## 🧪 Usage

### 1. **Training the Model**

To train the emotion detection model:

```bash
python TrainEmotionDetector.py
```

Ensure that the FER-2013 dataset is placed in the appropriate directory as expected by the training script.

### 2. **Evaluating the Model**

To evaluate the performance of the trained model:

```bash
python EvaluateEmotionDetector.py
```

### 3. **Testing with Webcam**

To perform real-time emotion detection using your webcam:

```bash
python TestEmotionDetector.py
```

### 4. **Testing with Images**

Modify the `TestEmotionDetector.py` script to load and process a static image instead of webcam input.

## 🗂️ Project Structure

```
Emotion-Detection/
├── haarcascades/
│   └── haarcascade_frontalface_default.xml
├── model/
│   └── emotion_model.h5
├── TrainEmotionDetector.py
├── EvaluateEmotionDetector.py
├── TestEmotionDetector.py
├── emoition_detection.png
├── requirements.txt
└── README.md
```

* **`haarcascades/`**: Contains Haar Cascade classifiers for face detection.
* **`model/`**: Directory to save and load the trained emotion detection model.
* **`TrainEmotionDetector.py`**: Script to train the CNN model on the FER-2013 dataset.
* **`EvaluateEmotionDetector.py`**: Script to evaluate the trained model's performance.
* **`TestEmotionDetector.py`**: Script to perform real-time emotion detection using webcam or images.
* **`emoition_detection.png`**: Sample image demonstrating the emotion detection output.
* **`requirements.txt`**: List of Python dependencies required to run the project.

## 📊 Dataset

The model is trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), which contains 35,887 grayscale images of 48x48 pixels categorized into seven emotion classes.

## 📈 Model Architecture

The Convolutional Neural Network (CNN) architecture includes:

* Multiple convolutional layers with ReLU activation.
* MaxPooling layers to reduce spatial dimensions.
* Dropout layers to prevent overfitting.
* Fully connected (Dense) layers leading to a Softmax output layer for classification.

## ✅ Results

The trained model achieves satisfactory accuracy on the FER-2013 test set, effectively recognizing emotions in real-time scenarios.

## 📌 Dependencies

* Python 3.x
* TensorFlow
* OpenCV
* NumPy
* Matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For any inquiries or feedback, please contact [Sourodyuti](https://github.com/Sourodyuti).

---

