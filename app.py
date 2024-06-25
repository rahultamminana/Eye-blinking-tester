import cv2
import dlib
import numpy as np

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Function to draw eye boundaries
def draw_eye_boundary(image, eye_points):
    hull = cv2.convexHull(eye_points)
    cv2.drawContours(image, [hull], -1, (0, 255, 0), 1)


# Function to determine if eye is closed based on eye aspect ratio (EAR)
def is_eye_closed(eye_points, facial_landmarks):
    ear = (np.linalg.norm(facial_landmarks[eye_points[1]] - facial_landmarks[eye_points[5]]) + np.linalg.norm(facial_landmarks[eye_points[2]] - facial_landmarks[eye_points[4]])) / (2.0 * np.linalg.norm(facial_landmarks[eye_points[0]] - facial_landmarks[eye_points[3]]))
    return ear < 0.3  # EAR threshold


# Start video capture
cap = cv2.VideoCapture(0)
frames = []

# Capture a few well-spaced frames (simplified for demonstration)
for _ in range(5):  # Adjust number of frames based on your requirement
    ret, frame = cap.read()
    if ret:
        frames.append(frame)
    cv2.waitKey(400)  # Wait for 400 ms between frames

eye_open_frame = None
eye_close_frame = None

# Process each frame
for frame in frames:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.matrix([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        left_eye_closed = is_eye_closed(list(range(36, 42)), landmarks)
        right_eye_closed = is_eye_closed(list(range(42, 48)), landmarks)

        if eye_open_frame is None and not left_eye_closed and not right_eye_closed:
            eye_open_frame = frame.copy()
            draw_eye_boundary(eye_open_frame, left_eye)
            draw_eye_boundary(eye_open_frame, right_eye)
        elif eye_open_frame is not None and (left_eye_closed or right_eye_closed):
            eye_close_frame = frame.copy()
            draw_eye_boundary(eye_close_frame, left_eye)
            draw_eye_boundary(eye_close_frame, right_eye)
            break  # Found the frames we needed

    if eye_close_frame is not None:
        break

# Release video capture
cap.release()

# Show and save the frames if both are captured
if eye_open_frame is not None and eye_close_frame is not None:
    cv2.imshow('Eye Open', eye_open_frame)
    cv2.imshow('Eye Closed', eye_close_frame)
    cv2.imwrite('eye_open.jpg', eye_open_frame)
    cv2.imwrite('eye_closed.jpg', eye_close_frame)
    cv2.waitKey(0)  # Wait for a key press to exit
    cv2.destroyAllWindows()
else:
    print("Did not capture both eye states.")
