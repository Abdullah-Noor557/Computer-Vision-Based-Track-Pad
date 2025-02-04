import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import pyautogui
import time

# --- Adjust PYTHONPATH ---
FILE = Path(__file__).resolve()
YOLO_ROOT = FILE.parents[2]
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.general import check_img_size
from utils.torch_utils import select_device

# ----- CONFIGURATION -----
MODEL_PATH = r'put your abs path for the best.pt here'
DEVICE = '0'
CAMERA_SOURCE = 3
IMG_SIZE = 224

CONF_THRESHOLD = 0.2

SCROLL_MULTIPLIER = 35
MIN_MOVEMENT_THRESHOLD = 15
SMOOTHING_FACTOR = 0.1
MAX_SCROLL = 100
FLOW_SPEED_NORM = 10
BRIGHTNESS_THRESHOLD = 160  # Reduced for more visible arrows
FLOW_FRAME_SKIP = 1         # Compute optical flow every 2 frames
# --------------------------

# Initialize model and device
device = select_device(DEVICE)
model = DetectMultiBackend(MODEL_PATH, device=device)
stride = model.stride
IMG_SIZE = check_img_size((IMG_SIZE, IMG_SIZE), s=stride)

# Video capture
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    raise RuntimeError("Failed to open camera")

# Visualization parameters
COLOR_ACTIVE = (76, 175, 80)       # Green
COLOR_INACTIVE = (244, 67, 54)     # Red
COLOR_SECONDARY = (33, 150, 243)   # Blue
COLOR_BG = (45, 45, 45)            # Dark Gray
COLOR_TEXT = (250, 250, 250)       # Off-white text
COLOR_HUD = (0, 255, 255)          # Neon Cyan

HUD_ALPHA = 0.4                    # Transparency for HUD panel
HUD_PADDING = 10                   # Padding inside the HUD panel

# Performance optimizations
prev_gray_small = None
smoothed_scroll = 0
last_scroll_time = 0
flow_frame_counter = 0  # Counter for optical flow frame skipping

# Explicit class names (modify according to your model)
CLASS_NAMES = {
    0: "Inactive",
    1: "Active"
}

def draw_flow_arrows(frame, flow, gray_small):
    """
    Draw arrows representing optical flow in bright regions.
    Arrows are colored based on flow direction and magnitude.
    """
    step = 20  # Increased step size for efficiency
    h, w = gray_small.shape[:2]
    
    # Create mask for bright areas
    mask = (gray_small > BRIGHTNESS_THRESHOLD).astype(np.uint8) * 255
    
    for y in range(0, h, step):
        for x in range(0, w, step):
            if mask[y, x] == 0:
                continue

            fx, fy = flow[y, x]
            mag = np.sqrt(fx**2 + fy**2)
            if mag < 0.5:  # Skip very small movements
                continue

            # Convert to HSV for better color representation
            angle = np.arctan2(fy, fx)
            hue = ((np.degrees(angle) + 180) % 180)  # 0-180 range
            saturation = 255
            value = min(int(255 * (mag / 2.0)), 255)  # Brighter with greater mag
            
            hsv_color = np.uint8([[[hue / 2, saturation, value]]])
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]

            # Scale coordinates back to original frame size
            scale = 2  # Downscaled by 0.5, so multiply by 2
            start = (x * scale, y * scale)
            end = (int(x * scale + fx * scale * 8),  # Enhanced for visibility
                   int(y * scale + fy * scale * 8))
            
            cv2.arrowedLine(
                frame, 
                start, 
                end, 
                bgr_color.tolist(), 
                2, 
                tipLength=0.3
            )

def draw_sci_fi_hud(frame, label, max_conf, smoothed_scroll, fps):
    """
    Draw a semi-transparent HUD panel in the top-left with classification info, 
    plus a net scroll direction arrow in the center of the frame.
    """

    # Panel for text (semi-transparent)
    overlay = frame.copy()
    panel_width = 300
    panel_height = 130
    cv2.rectangle(
        overlay,
        (0, 0),
        (panel_width, panel_height),
        COLOR_BG,  # fill color
        -1
    )
    # Blend panel onto the frame
    cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

    # Text offsets inside the panel
    text_x = HUD_PADDING
    text_y = HUD_PADDING + 30

    # 1. Label & Confidence
    label_text = f"{label}: {max_conf * 100:.1f}%"
    color_label = COLOR_ACTIVE if label == "Active" else COLOR_INACTIVE
    cv2.putText(frame, label_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_label, 2)

    # Confidence Bar (0-100%)
    bar_x = text_x
    bar_y = text_y + 10
    bar_width = 200
    bar_height = 10
    # Outline
    cv2.rectangle(frame, (bar_x, bar_y + 15), (bar_x + bar_width, bar_y + 15 + bar_height),
                  (100, 100, 100), 2)
    filled_width = int(bar_width * max_conf)
    # Fill
    cv2.rectangle(frame, (bar_x, bar_y + 15),
                  (bar_x + filled_width, bar_y + 15 + bar_height),
                  color_label, -1)

    # 2. Scroll speed
    scroll_val = int(round(smoothed_scroll))
    scroll_text = f"Scroll: {scroll_val} px"
    cv2.putText(frame, scroll_text, (text_x, bar_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SECONDARY, 2)

    # Scroll Speed Bar (-100 to +100)
    scroll_bar_x = text_x
    scroll_bar_y = bar_y + 60
    scroll_bar_width = 200
    scroll_bar_height = 10
    cv2.rectangle(frame,
                  (scroll_bar_x, scroll_bar_y + 15),
                  (scroll_bar_x + scroll_bar_width, scroll_bar_y + 15 + scroll_bar_height),
                  (100, 100, 100), 2)
    # Normalize from -100..+100 => 0..200
    normalized_scroll = np.clip(scroll_val + 100, 0, 200)
    cv2.rectangle(
        frame,
        (scroll_bar_x, scroll_bar_y + 15),
        (scroll_bar_x + normalized_scroll, scroll_bar_y + 15 + scroll_bar_height),
        COLOR_SECONDARY,
        -1
    )

    # 3. FPS
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (text_x, scroll_bar_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)

    # 4. Net direction arrow in the center
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Determine arrow length and direction (up or down)
    arrow_scale = 1.0    # 1 pixel per 1 scroll unit
    arrow_length = int(np.clip(abs(scroll_val) * arrow_scale, 0, 80))
    sign = np.sign(scroll_val)
    # If scroll_val > 0 => arrow down. If scroll_val < 0 => arrow up.

    start_pt = (center_x, center_y)
    end_pt = (center_x, center_y + arrow_length * sign)

    # Draw the arrow only if there's noticeable scroll
    if arrow_length > 2:
        cv2.arrowedLine(frame, start_pt, end_pt, COLOR_HUD, 3, tipLength=0.2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # --- Classification ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    classification_input = cv2.resize(frame_rgb, (IMG_SIZE[0], IMG_SIZE[1]))
    img = classification_input.astype('float32') / 255.0
    im = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    im = im.half() if model.fp16 else im.float()

    # Warmup model if needed (to stabilize first frames)
    if prev_gray_small is None:
        for _ in range(2):
            model(im)
    
    results_yolo = model(im)
    prob = F.softmax(results_yolo, dim=1)[0]
    max_conf, max_idx = torch.max(prob, dim=0)
    max_conf = max_conf.item()
    max_idx = max_idx.item()
    label = CLASS_NAMES[max_idx] if max_conf >= CONF_THRESHOLD else "Low confidence"

    # --- Optical Flow (for scroll) ---
    if label == "Active":
        # Downscaled grayscale for optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        
        flow_frame_counter += 1
        if prev_gray_small is not None and (flow_frame_counter % FLOW_FRAME_SKIP == 0):
            # Compute flow every FLOW_FRAME_SKIP frames
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray_small, gray_small, None,
                pyr_scale=0.5, 
                levels=2,
                winsize=10,
                iterations=2,
                poly_n=3,
                poly_sigma=1.1,
                flags=0
            )
            
            avg_dy = np.mean(flow[..., 1]) * 100
            if abs(avg_dy) < MIN_MOVEMENT_THRESHOLD:
                avg_dy = 0
            raw_scroll = -int(avg_dy * SCROLL_MULTIPLIER)
            smoothed_scroll = SMOOTHING_FACTOR * raw_scroll + (1 - SMOOTHING_FACTOR) * smoothed_scroll
            
            # Throttle scroll events to ~50Hz max
            if time.time() - last_scroll_time > 0.02:
                scroll_to_apply = int(round(smoothed_scroll))
                if scroll_to_apply != 0:
                    pyautogui.scroll(scroll_to_apply)
                    last_scroll_time = time.time()

            # Draw optical flow vectors
            draw_flow_arrows(frame, flow, gray_small)

        prev_gray_small = gray_small.copy()
    else:
        # Reset optical flow if not active
        prev_gray_small = None
        smoothed_scroll = 0
        flow_frame_counter = 0

    # --- UI Rendering ---
    fps = 1 / (time.time() - start_time)
    draw_sci_fi_hud(frame, label, max_conf, smoothed_scroll, fps)

    cv2.imshow("Sci-Fi Touch Interface", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
