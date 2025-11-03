# Dynamic Face-Fitting AR Overlay Framework
Computer Vision + Augmented Reality implementation

Detects facial landmarks from a video and determines facial geometry such as head width, chin position, and eye spacing using MediaPipe. 
Automatically adjusts and scales a given overlay image to match the detected facial dimensions and position.

Image background is removed using rembg, followed by an OpenCV-based transparency cleanup to eliminate white or near-white regions. 

The system uses MediaPipe for face tracking, OpenCV for image processing and blending, Pillow for image handling, and rembg for background removal. The result is an automatically fitted overlay that tracks facial movements frame by frame.
