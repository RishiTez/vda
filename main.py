import cv2
import numpy as np
import mediapipe as mp
import os
import urllib.request
import time

files = {
    "yolov3-tiny.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
    "yolov3-tiny.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
}

print("Checking for AI models (YOLOv3-Tiny)...")
for file_name, url in files.items():
    if not os.path.exists(file_name):
        print(f"Downloading {file_name}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(file_name, 'wb') as out_file:
                out_file.write(response.read())
            print(f"Successfully downloaded {file_name}")
        except Exception as e:
            print(f"\n[ERROR] Failed to download {file_name}: {e}")
            print(f"Please manually download it from: {url}")

CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

net = None
try:
    if os.path.exists("yolov3-tiny.weights") and os.path.exists("yolov3-tiny.cfg"):
        net = cv2.dnn.readNetFromDarknet("yolov3-tiny.cfg", "yolov3-tiny.weights")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("YOLOv3-Tiny Model Loaded Successfully!")
    else:
        print("Model files missing. Object detection will not work.")
except Exception as e:
    print(f"Error loading model: {e}")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

img_canvas = np.zeros((720, 1280, 3), np.uint8)
xp, yp = 0, 0
scan_points = [] 
draw_color = (255, 0, 255)  
scan_mode = False           
detected_object_label = ""  
scan_timer = 0
brush_thickness = 15
eraser_thickness = 50

def draw_ui(img):
    """Draws the UI elements."""
    # Clear Button
    cv2.rectangle(img, (1100, 10), (1250, 80), (50, 50, 50), cv2.FILLED)
    cv2.putText(img, "CLEAR", (1130, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # SCAN Button
    btn_color = (200, 200, 200) if scan_mode else (100, 0, 100)
    text_color = (0, 0, 0) if scan_mode else (255, 255, 255)
    cv2.rectangle(img, (900, 10), (1050, 80), btn_color, cv2.FILLED)
    cv2.putText(img, "SCAN", (935, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # Color Boxes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)] 
    x_start = 40
    for i, color in enumerate(colors):
        cv2.rectangle(img, (x_start, 10), (x_start + 140, 80), color, cv2.FILLED)
        if draw_color == color and not scan_mode:
             cv2.rectangle(img, (x_start, 10), (x_start + 140, 80), (255, 255, 255), 3)
        x_start += 160
    
    # Eraser
    cv2.rectangle(img, (700, 10), (850, 80), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "ERASER", (720, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if draw_color == (0, 0, 0) and not scan_mode:
        cv2.rectangle(img, (700, 10), (850, 80), (255, 255, 255), 3)
        
    # Result Display
    if detected_object_label:
        overlay = img.copy()
        cv2.rectangle(overlay, (400, 300), (880, 420), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, f"{detected_object_label}", (420, 370), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    if scan_mode:
        cv2.putText(img, "Draw box around object...", (420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

def make_square(img):
    """Pads an image with black borders to make it square without distortion."""
    h, w = img.shape[:2]
    if h == w: return img
    size = max(h, w)
    square = np.zeros((size, size, 3), np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = img
    return square

def recognize_object(img, points):
    global detected_object_label
    if len(points) < 10 or net is None: return

    points_np = np.array(points)
    x, y, w, h = cv2.boundingRect(points_np)
    
    h_img, w_img, _ = img.shape
    pad = 40 
    y1 = max(0, y - pad)
    y2 = min(h_img, y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(w_img, x + w + pad)
    
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return

    roi_square = make_square(roi)
    cv2.imshow("AI Vision (Crop)", roi_square)

    blob = cv2.dnn.blobFromImage(roi_square, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
         output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
         
    outputs = net.forward(output_layers)

    class_ids = []
    confidences = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.3: # Threshold
                class_ids.append(class_id)
                confidences.append(float(confidence))

    if len(class_ids) > 0:
        max_idx = np.argmax(confidences)
        label = CLASSES[class_ids[max_idx]]
        confidence = confidences[max_idx]
        detected_object_label = f"{label} {int(confidence*100)}%"
    else:
        detected_object_label = "Unsure (Try closer)"

print("Starting AI Canvas... Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])
            
            if len(lm_list) != 0:
                x1, y1 = lm_list[8][1:]   # Index
                x2, y2 = lm_list[12][1:]  # Middle
                
                # Check fingers
                fingers = []
                if lm_list[8][2] < lm_list[6][2]: fingers.append(1)
                else: fingers.append(0)
                if lm_list[12][2] < lm_list[10][2]: fingers.append(1)
                else: fingers.append(0)

                # --- SELECTION MODE ---
                if fingers[0] == 1 and fingers[1] == 1:
                    xp, yp = 0, 0
                    if y1 < 100:
                        if 900 < x1 < 1050: # SCAN
                            if time.time() - scan_timer > 1:
                                scan_mode = not scan_mode
                                detected_object_label = ""
                                scan_points = []
                                scan_timer = time.time()
                                cv2.destroyWindow("AI Vision (Crop)")
                        elif 1100 < x1 < 1250: # CLEAR
                            img_canvas = np.zeros((720, 1280, 3), np.uint8)
                            detected_object_label = ""
                            cv2.destroyWindow("AI Vision (Crop)")
                        elif not scan_mode: # Colors
                            if 40 < x1 < 180: draw_color = (255, 0, 0)
                            elif 200 < x1 < 340: draw_color = (0, 255, 0)
                            elif 360 < x1 < 500: draw_color = (0, 0, 255)
                            elif 520 < x1 < 660: draw_color = (0, 255, 255)
                            elif 700 < x1 < 850: draw_color = (0, 0, 0)

                    cv2.rectangle(img, (x1-15, y1-25), (x2+15, y2+25), draw_color, cv2.FILLED)

                if fingers[0] == 1 and fingers[1] == 0:
                    if scan_mode:
                        cv2.circle(img, (x1, y1), 10, (255, 255, 255), cv2.FILLED)
                        if xp == 0 and yp == 0: xp, yp = x1, y1
                        cv2.line(img, (xp, yp), (x1, y1), (255, 255, 255), 5)
                        scan_points.append((x1, y1))
                    else:
                        cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
                        if xp == 0 and yp == 0: xp, yp = x1, y1
                        if draw_color == (0, 0, 0):
                            cv2.line(img, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                            cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, eraser_thickness)
                        else:
                            cv2.line(img, (xp, yp), (x1, y1), draw_color, brush_thickness)
                            cv2.line(img_canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                    xp, yp = x1, y1
                
                if scan_mode and fingers[0] == 0 and len(scan_points) > 30:
                    recognize_object(img.copy(), scan_points)
                    scan_points = []
                    scan_mode = False 
    else:
        xp, yp = 0, 0

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, img_canvas)

    draw_ui(img)
    cv2.imshow("AI Air Canvas", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
