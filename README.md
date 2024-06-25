# Eye-blinking-tester
### Project Overview: Real-Time Eye State Detection System

**Objective**: The aim of this project is to develop a real-time system capable of detecting whether a person's eyes are open or closed using computer vision techniques.

**Technologies Used**:
- **Dlib**: Utilized for face detection and facial landmark prediction.
- **OpenCV**: Employed for video capture, frame processing, and visualization.
- **NumPy**: Used for numerical operations, particularly for calculating the Eye Aspect Ratio (EAR).

**Key Features**:

1. **Real-Time Eye State Detection**:
   - Implemented a real-time system that uses dlib's frontal face detector and 68-point facial landmark predictor.
   - Calculated the Eye Aspect Ratio (EAR) to determine eye state, setting an EAR threshold of 0.3 for closed eyes.
   - Efficiently processes video feed to identify eye states with high accuracy.

2. **Frame Capture and Processing**:
   - Captured a series of frames from the video feed, spaced 400 milliseconds apart, to ensure diverse sampling.
   - Each frame was converted to grayscale and processed to detect faces and extract eye regions.
   - Extracted 12 key points (6 for each eye) from the facial landmarks to evaluate the eye states accurately.

3. **Visualization and Output**:
   - Utilized OpenCV to draw convex hulls around the eye regions for clear visualization of the detected eye states.
   - Captured and saved frames showing both open and
