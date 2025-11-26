-----

# Real-Time Facial Emotion Recognition

## Project Overview

This repository houses a computer vision application designed to detect and classify human facial expressions in real-time video streams. By leveraging deep learning techniques, specifically Convolutional Neural Networks (CNNs), the system analyzes facial features to predict emotional states with high accuracy.

The model is trained to recognize seven universal emotional classes: **Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.**

-----

## Technical Architecture

The application operates on a sequential pipeline that processes raw video input into classified emotional labels.

### 1\. Data Acquisition & Preprocessing

The system captures video frames via the default webcam interface. Each frame undergoes preprocessing to ensure compatibility with the neural network:

  * **Grayscale Conversion:** Reduces computational complexity by converting 3-channel RGB images to single-channel tensors.
  * **Face Detection:** Utilizes **Haar Cascade Classifiers** to scan the frame for facial structures. This step isolates the Region of Interest (ROI), discarding background noise.
  * **Normalization:** The ROI is resized to the target input dimension (typically 48x48 pixels) and pixel values are normalized to a range of 0-1 to improve model convergence.

### 2\. Deep Learning Inference (CNN)

The core classification engine is a deep Convolutional Neural Network. The architecture follows a standard pattern for image classification tasks:

  * **Convolutional Layers:** Extract spatial features such as edges, textures, and shapes from the input image.
  * **Pooling Layers (Max Pooling):** Reduce dimensionality, making the model more robust to variations in position and rotation while decreasing computational load.
  * **Dropout Layers:** Applied to prevent overfitting during the training phase.
  * **Dense (Fully Connected) Layers:** Interpret the high-level features extracted by the convolutional layers.
  * **Softmax Activation:** The final layer outputs a probability distribution across the seven emotion classes.

### 3\. Visualization Module

Post-inference, the system overlays the classification results onto the original video feed. This includes drawing a bounding box around the detected face and labeling it with the predicted emotion and its associated confidence score.

-----

## System Capabilities

  * **Low-Latency Inference:** Optimized for real-time performance, capable of maintaining high FPS on standard CPU architectures without the need for dedicated GPU hardware.
  * **Multi-Subject Detection:** The detection algorithm is scalable, capable of identifying and classifying multiple faces within a single frame simultaneously.
  * **Robust Feature Extraction:** The model prioritizes structural facial landmarks over pixel intensity, allowing for consistent performance across varying lighting conditions.
  * **Modular Codebase:** The project is structured to allow for easy swapping of model architectures or integration into larger systems (e.g., security monitoring or market research tools).

-----

## Repository Structure

  * `main.py`: **Application Entry Point.** Handles video stream initialization, invokes the detection pipeline, and manages the GUI display window.
  * `model.py`: **Network Architecture.** Defines the structure of the CNN, including layer configurations and hyperparameter settings.
  * `train.py`: **Training Script.** Manages the data loading, model compilation, and training loop. Includes checkpointing to save the best-performing model weights.
  * `utils.py`: **Utility Functions.** Contains helper methods for image processing, loading pre-trained weights, and visualization tasks.

-----

## Future Scope

  * **Temporal Analysis:** Implementation of LSTM (Long Short-Term Memory) networks to analyze sequences of frames, improving stability by using context from previous frames.
  * **Multi-Modal Integration:** Incorporating audio analysis to process vocal intonations alongside visual data for increased classification accuracy.
  * **Edge Deployment:** Optimization for deployment on edge devices (e.g., Raspberry Pi, Jetson Nano) using TensorFlow Lite.
