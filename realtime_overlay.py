import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import io
import tkinter as tk
from tkinter import filedialog

# all the functions created here

# will remove the white stuff from ur image so that the background is not there
def removebg(img_bgra):
    if img_bgra.shape[2] == 4:
        b, g, r, a = cv2.split(img_bgra)
    else:
        b, g, r = cv2.split(img_bgra)
        a = np.ones(b.shape, dtype=np.uint8) * 255

    white_mask = (r > 240) & (g > 240) & (b > 240)

    kernel = np.ones((3,3), np.uint8)
    white_mask = cv2.erode(white_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    a[white_mask] = 0
    return cv2.merge([b, g, r, a])

# will apply the image on ur face ie overlaying
def overlayit(frame, overlay, x, y):
    h, w = overlay.shape[:2]
    if x >= frame.shape[1] or y >= frame.shape[0] or x + w <= 0 or y + h <= 0:
        return frame

    y1, y2 = max(0, y), min(frame.shape[0], y + h)
    x1, x2 = max(0, x), min(frame.shape[1], x + w)
    oy1, oy2 = max(0, -y), min(h, frame.shape[0] - y)
    ox1, ox2 = max(0, -x), min(w, frame.shape[1] - x)

    alpha_overlay = overlay[oy1:oy2, ox1:ox2, 3] / 255.0
    alpha_frame = 1.0 - alpha_overlay

    for c in range(3):
        frame[y1:y2, x1:x2, c] = (
            alpha_overlay * overlay[oy1:oy2, ox1:ox2, c] +
            alpha_frame * frame[y1:y2, x1:x2, c]
        )

    return frame

# will get actual coordinates of different locations of face
def get_point(lms, idx, w, h):
    return int(lms[idx].x * w), int(lms[idx].y * h)

# calculates euclidean distance
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# decides overlay anchor position + overlay width based on face landmarks and chosen placement
def get_config(lms, w, h, placement):
    left_eye_outer = get_point(lms, 133, w, h)
    right_eye_outer = get_point(lms, 362, w, h)
    left_ear = get_point(lms, 234, w, h)
    right_ear = get_point(lms, 454, w, h)
    nose_tip = get_point(lms, 1, w, h)
    top_head = get_point(lms, 10, w, h)
    chin = get_point(lms, 152, w, h)
    mouth_left = get_point(lms, 61, w, h)
    mouth_right = get_point(lms, 291, w, h)

    ear_to_ear = distance(left_ear, right_ear)
    top_to_chin = distance(top_head, chin)
    eye_center = ((left_eye_outer[0]+right_eye_outer[0])//2,
                  (left_eye_outer[1]+right_eye_outer[1])//2)
    mouth_center = ((mouth_left[0]+mouth_right[0])//2,
                    (mouth_left[1]+mouth_right[1])//2)

    if placement == "head":
        anchor = (eye_center[0], top_head[1] - int(0.1 * top_to_chin))
        fit_width = ear_to_ear * 1.1
    elif placement == "forehead":
        anchor = (eye_center[0], top_head[1] + int(0.25 * top_to_chin))
        fit_width = ear_to_ear * 0.9
    elif placement == "eyes":
        anchor = eye_center
        fit_width = ear_to_ear * 0.8
    elif placement == "nose":
        anchor = nose_tip
        fit_width = ear_to_ear * 0.4
    elif placement == "mouth":
        anchor = mouth_center
        fit_width = ear_to_ear * 0.6
    elif placement == "chin":
        anchor = (chin[0], chin[1] - int(0.05 * top_to_chin))
        fit_width = ear_to_ear * 1.2
    elif placement == "both_cheeks":
        anchor = None
        fit_width = ear_to_ear * 0.4
    else:
        anchor = nose_tip
        fit_width = ear_to_ear * 0.6

    return anchor, fit_width, left_ear, right_ear



# gui
overlay_img = None

def select_overlay():
    global overlay_img
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select overlay image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.webp")]
    )
    root.destroy()

    if not path:
        return

    img = Image.open(path).convert("RGBA")
    arr = np.array(img)

    bgra = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
    bgra = removebg(bgra)

    overlay_img = bgra
    print("Overlay loaded:", path)

# mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

placements = ["head","forehead","eyes","nose","mouth","chin","both_cheeks"]
placement = "nose"

# real-time loop
cap = cv2.VideoCapture(0)

print("\nREAL-TIME AR READY")
print("Controls:")
print("  o = choose overlay image")
print("  1-7 = change placement")
print("  q = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks and overlay_img is not None:
        for face_landmarks in results.multi_face_landmarks:
            lms = face_landmarks.landmark

            anchor, fit_width, left_ear, right_ear = get_config(lms, w, h, placement)
            aspect_ratio = overlay_img.shape[0] / overlay_img.shape[1]

            new_w = int(fit_width)
            new_h = int(new_w * aspect_ratio)

            resized_overlay = cv2.resize(overlay_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if placement == "both_cheeks":
                for pos in [left_ear, right_ear]:
                    x = int(pos[0] - new_w/2)
                    y = int(pos[1] - new_h/2)
                    frame = overlayit(frame, resized_overlay, x, y)
            else:
                x = int(anchor[0] - new_w/2)
                y = int(anchor[1] - new_h/2)
                frame = overlayit(frame, resized_overlay, x, y)

    cv2.putText(frame, f"Placement: {placement}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("realTime AR", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('o'):
        select_overlay()

    if key in [ord(str(i)) for i in range(1,8)]:
        placement = placements[int(chr(key)) - 1]

cap.release()
cv2.destroyAllWindows()
