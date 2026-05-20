# Facial Emotion Recognition

A real-time facial emotion recognition system using MediaPipe Face Mesh and machine learning. This implementation is based on the approach from [REWTAO/Facial-emotion-recognition-using-mediapipe](https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe).

## Features

- **Real-time emotion detection** using webcam
- **7 emotion classes**: happy, sad, angry, surprised, neutral, fear, disgust
- **MediaPipe Face Mesh** for accurate facial landmark detection
- **Custom training** with your own emotion data
- **Image collection tool** for building custom datasets
- **TensorFlow Lite model** for efficient inference

## Quick Start

Python Scripts are under the assumption that you are at the root of the repository

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Collect Training Data

Run the data collection script to capture facial expressions for each emotion:

```bash
python -m apps.facial_recognition.collect_images
```

**Controls:**
- `SPACE`: Save image and keypoints
- `N`: Next emotion
- `K`: Toggle keypoint logging mode (required for training)
- `0-6`: Direct emotion selection (0=happy, 1=sad, 2=angry, 3=surprised, 4=neutral, 5=fear, 6=disgust)
- `Q`: Quit

**Important:** Make sure to enable "Keypoint logging mode" (press `K`) to save the facial landmarks needed for training!

### 3. Train the Model

Once you have collected sufficient data for each emotion:

```bash
python -m apps.facial_recognition.train
```

This will:
- Load your collected keypoint data
- Train a neural network model
- Generate evaluation metrics and plots
- Save the trained model as `emotion_classifier.tflite`

### 4. Run Real-time Recognition

```bash
python -m apps.facial_recognition.main
```

The system will start your webcam and display detected emotions in real-time.

## File Structure

```
facial_recognition/
├── collect_images.py          # Data collection script
├── train.py                   # Model training script
├── main.py                    # Real-time inference
├── README.md                  # This file
└── csv/
        ├── keypoint.csv                    # Training data
        └── emotion_classifier_label.csv    # Emotion labels
```

## How It Works

1. **Face Detection**: MediaPipe Face Mesh detects 468 facial landmarks
2. **Preprocessing**: Landmarks are normalized to relative coordinates
3. **Classification**: A multilayer perceptron (MLP) neural network classifies the emotion
4. **Real-time Display**: Results are overlaid on the video feed

## Data Collection Tips

- **Collect diverse expressions**: Try different intensities of each emotion
- **Good lighting**: Ensure your face is well-lit and clearly visible
- **Multiple angles**: Collect data from slightly different head positions
- **Consistent background**: Use a clean, uncluttered background
- **Sufficient samples**: Aim for at least 50-100 samples per emotion for good results

## Model Performance

The model achieves high accuracy on collected data, but performance depends heavily on:
- **Data quality**: Clear, well-lit facial expressions
- **Data quantity**: More samples generally improve performance
- **Data diversity**: Variety in expressions, lighting, and angles

## Troubleshooting

### "No face detected!" message
- Ensure good lighting
- Position your face clearly in the camera frame
- Check camera permissions

### Poor emotion recognition accuracy
- Collect more training data
- Ensure keypoint logging mode was enabled during collection
- Retrain the model with more diverse expressions

### Import errors
- Install all requirements: `pip install -r requirements.txt`
- Ensure you're using compatible versions of the dependencies

## Credits

This implementation is based on the excellent work by:
- [REWTAO/Facial-emotion-recognition-using-mediapipe](https://github.com/REWTAO/Facial-emotion-recognition-using-mediapipe)
- [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
- Google's MediaPipe framework

## License

This project follows the same Apache 2.0 license as the original implementation.
