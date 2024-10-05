# Wake-Watch: Driver Drowsiness Detection System

## Overview

Wake-Watch is an intermediate Python project that detects driver drowsiness to prevent accidents caused by fatigue. The system uses computer vision and deep learning techniques to monitor a driver's eyes and alert them when signs of drowsiness are detected.

## Features

- Real-time eye state detection (open/closed) using a webcam
- Convolutional Neural Network (CNN) model for accurate classification
- Audio alert system when drowsiness is detected
- User-friendly interface with live video feed and status display

## Image of the Grapgh


<img width="1033" alt="Results of Graph" src="https://github.com/user-attachments/assets/f15ad2d4-8b4d-4b8e-a4d0-d43f0249b317">




## Prerequisites

- Python 3.6 or higher
- Webcam

## Installation

1. Clone the repository:
2. Install required packages:

  
3. ## Usage
1. Navigate to the project directory:
2. Run the main script:
3. Position yourself in front of the webcam and keep your eyes open.
4. The system will alert you with an alarm sound if it detects that your eyes are closed for too long.

## Project Structure

- `drowsiness detection.py`: Main script for real-time drowsiness detection
- `model.py`: CNN model architecture and training script
- `models/cnnCat2.h5`: Pre-trained model weights
- `haar_cascade_files/`: XML files for face and eye detection
- `alarm.wav`: Audio file for drowsiness alert
- `data/`: Directory containing the dataset (not included in repository)

## How It Works

1. Capture video input from the webcam
2. Detect face and eyes using Haar cascades
3. Preprocess and feed eye images to the CNN classifier
4. Determine if eyes are open or closed
5. Calculate a drowsiness score based on eye state over time
6. Trigger an alarm if the drowsiness score exceeds a threshold

## Dataset

The model was trained on a custom dataset of approximately 7000 images of eyes in various lighting conditions. The dataset is not included in this repository.

## Contributing

Contributions to improve Wake-Watch are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgements

- OpenCV for computer vision capabilities
- TensorFlow and Keras for deep learning functionalities
- Pygame for audio playback
