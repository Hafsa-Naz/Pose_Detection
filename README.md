Pose Estimation – Activity Detection

This project uses MediaPipe Pose and OpenCV to detect human body pose from a video and classify activities as Standing or Squatting using joint angle analysis.

The system extracts key body landmarks and computes joint angles (knee, hip, elbow) using vector geometry. The knee angle is used as the main feature for classification. A simple rule is applied: knee angle > 140° = Standing, otherwise Squatting.

To improve stability, a Savitzky–Golay filter is used to smooth noisy angle signals. The project also generates a real-time skeleton overlay, joint angle visualization plot, and an accuracy report compared with ground truth labels.

Results:
Accuracy achieved: 90.6% (29 correct out of 32 frames), with minor errors during movement transitions.

Requirements:
opencv-python, mediapipe, numpy, matplotlib, scipy
