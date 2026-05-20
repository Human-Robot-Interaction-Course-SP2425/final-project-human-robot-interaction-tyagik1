# Hand Gesture Recognition

A keypoint-based hand gesture recognition system using MediaPipe Hand Landmarks and machine learning. This implementation is based on the approach from [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) but simplified to remove Docker dependencies.

## Features

- **Real-time gesture detection** using webcam
- **5 gesture classes**: thumbs_up, peace_sign, closed_fist, open_palm, none
- **MediaPipe Hand Landmarks** for accurate hand tracking (21 keypoints)
- **Custom training** with your own gesture data
- **Keypoint-based approach** - no images stored, only landmark coordinates
- **TensorFlow Lite model** for efficient inference
- **No Docker required** - pure Python implementation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Python Scripts are under the assumption that you are at the root of the repository

### 2. Collect Training Data

Run the keypoint collection script:

```bash
python -m apps.gesture_recognition.collect_images
```

**Controls:**
- `SPACE`: Save keypoints for current gesture
- `N`: Next gesture
- `Q`: Quit

The script will save hand landmark coordinates to `apps/gesture_recognition/csv/keypoint.csv`.

### 3. Train the Model

Once you have collected sufficient data for each gesture:

```bash
python -m apps.gesture_recognition.train
```

This will:
- Load your collected keypoint data
- Train a neural network model
- Generate evaluation metrics and plots
- Save the trained model as `gesture_classifier.tflite`

### 4. Run Real-time Recognition

```bash
python -m apps.gesture_recognition.main
```

The system will start your webcam and display detected gestures in real-time.

## File Structure

```
gesture_recognition/
├── train.py                   # Model training script
├── main.py                    # Real-time inference
├── collect_images.py          # Legacy image collection (not used)
├── README.md                  # This file
├── csv/
    ├── keypoint.csv                    # Training data
    └── gesture_classifier_label.csv    # Gesture labels

```

## How It Works

1. **Hand Detection**: MediaPipe Hand Landmarks detects 21 hand keypoints
2. **Preprocessing**: Landmarks are normalized to relative coordinates
3. **Classification**: A multilayer perceptron (MLP) neural network classifies the gesture
4. **Real-time Display**: Results are overlaid on the video feed with bounding boxes

## Advantages Over Image-Based Approaches

- **Faster**: No image processing, direct coordinate analysis
- **Smaller Dataset**: Only coordinates stored, not images
- **More Robust**: Invariant to lighting and background changes
- **Privacy-friendly**: No images stored, just mathematical coordinates
- **Efficient**: Smaller model size and faster inference

## Data Collection Tips

- **Clear Hand Positioning**: Keep your hand clearly visible in the frame
- **Consistent Gestures**: Make sure gestures are distinct and repeatable
- **Good Lighting**: Ensure your hand is well-lit for MediaPipe detection
- **Multiple Angles**: Collect data from slightly different hand positions
- **Sufficient Samples**: Aim for at least 100-200 samples per gesture

## Model Performance

The model achieves high accuracy on collected data. Performance depends on:
- **Data quality**: Clear, consistent gesture samples
- **Data quantity**: More samples generally improve performance
- **Gesture distinctiveness**: More different gestures are easier to classify

## Troubleshooting

### "No hand detected!" message
- Ensure good lighting
- Position your hand clearly in the camera frame
- Check camera permissions

### Poor gesture recognition accuracy
- Collect more training data
- Ensure gestures are distinct and consistent
- Retrain the model with more diverse samples

### Import errors
- Install all requirements: `pip install -r requirements.txt`
- Ensure you're using compatible versions of the dependencies

## Credits

This implementation is based on the excellent work by:
- [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
- Google's MediaPipe framework

## License

This project follows the same Apache 2.0 license as the original implementation.
