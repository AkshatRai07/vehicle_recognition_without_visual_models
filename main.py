import cv2
import numpy as np
import math
import os
import glob
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import math

class EuclideanDistTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.id_count = 0
        self.max_disappeared = 20  
        self.min_hits = 4          

    def update(self, objects_rect):
        objects_bbs_ids = []
        used_ids = set()

        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            min_dist = 99999
            closest_id = -1

            for id, data in self.tracked_objects.items():
                if id in used_ids: continue

                pt = data['center']
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < min_dist:
                    min_dist = dist
                    closest_id = id

            search_radius = max(150, h * 3)

            if min_dist < search_radius and closest_id != -1:
                prev_cy = self.tracked_objects[closest_id]['center'][1]
                
                current_dy = cy - prev_cy
                
                old_dy = self.tracked_objects[closest_id]['avg_dy']
                self.tracked_objects[closest_id]['avg_dy'] = (old_dy * 0.6) + (current_dy * 0.4)

                self.tracked_objects[closest_id]['center'] = (cx, cy)
                self.tracked_objects[closest_id]['bbox'] = (x, y, w, h)
                self.tracked_objects[closest_id]['disappeared_count'] = 0
                self.tracked_objects[closest_id]['seen_count'] += 1
                
                start_y = self.tracked_objects[closest_id]['start_cy']
                self.tracked_objects[closest_id]['total_dist'] = abs(cy - start_y)

                used_ids.add(closest_id)
                
                if self.tracked_objects[closest_id]['seen_count'] >= self.min_hits:
                    objects_bbs_ids.append([x, y, w, h, closest_id])
            else:
                self.tracked_objects[self.id_count] = {
                    'center': (cx, cy), 
                    'start_cy': cy,
                    'seen_count': 1,
                    'disappeared_count': 0,
                    'bbox': (x, y, w, h),
                    'avg_dy': 0,
                    'total_dist': 0
                }
                self.id_count += 1

        clean_tracked_objects = {}
        for id, data in self.tracked_objects.items():
            if id in used_ids:
                clean_tracked_objects[id] = data
            else:
                data['disappeared_count'] += 1
                if data['disappeared_count'] <= self.max_disappeared:
                    clean_tracked_objects[id] = data
        
        self.tracked_objects = clean_tracked_objects.copy()
        return objects_bbs_ids

class Solution:
    def __init__(self):
        self.tracker = EuclideanDistTracker()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        
        self.counted_ids = set()
        self.flow_direction_locked = False
        self.flow_dy_sum = 0
        self.global_dy_sign = 0
        self.frames_processed = 0

        self.kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    def get_roi_mask(self, frame_shape):
        height, width = frame_shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        pts = np.array([
            [int(width * 0.05), height], 
            [int(width * 0.95), height], 
            [int(width * 0.80), int(height * 0.30)], 
            [int(width * 0.20), int(height * 0.30)]
        ])
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def is_valid_vehicle(self, w, h, dy):
        ratio = float(w) / h
        if ratio < 0.28:
            return False
        
        if abs(dy) < 1.5: 
            return False
            
        return True

    def forward(self, video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return 0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        roi_mask = self.get_roi_mask((height, width))
        
        self.tracker = EuclideanDistTracker()
        self.counted_ids = set()
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
        self.flow_direction_locked = False
        self.flow_dy_sum = 0
        self.global_dy_sign = 0
        self.frames_processed = 0

        while True:
            ret, frame = cap.read()
            if not ret: break
            self.frames_processed += 1

            masked = cv2.bitwise_and(frame, frame, mask=roi_mask)
            blurred = cv2.GaussianBlur(masked, (7, 7), 0)
            mask = self.bg_subtractor.apply(blurred)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            
            mask = cv2.erode(mask, self.kernel_erode, iterations=1)
            mask = cv2.dilate(mask, self.kernel_dilate, iterations=2)
            mask = cv2.morphologyEx(mask, self.kernel_close)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                cy = y + h // 2
                
                scale = (cy - (height * 0.2)) / (height * 0.8)
                scale = max(0.1, min(1.0, scale))
                min_area = 150 + (1200 * scale)

                if area > min_area:
                    detections.append([x, y, w, h])

            boxes_ids = self.tracker.update(detections)
            
            for box in boxes_ids:
                x, y, w, h, id = box
                obj_data = self.tracker.tracked_objects[id]
                avg_dy = obj_data['avg_dy']
                cy = obj_data['center'][1]

                if not self.flow_direction_locked:
                    if abs(avg_dy) > 1:
                        self.flow_dy_sum += avg_dy
                    
                    if self.frames_processed > 30 and abs(self.flow_dy_sum) > 10:
                        self.global_dy_sign = 1 if self.flow_dy_sum > 0 else -1
                        self.flow_direction_locked = True
                
                else:
                    if (avg_dy * self.global_dy_sign) < 0:
                        continue
                    
                    if not self.is_valid_vehicle(w, h, avg_dy):
                        continue

                    zone_start = int(height * 0.40)
                    zone_end = int(height * 0.60)
                    
                    if zone_start < cy < zone_end:
                        if id not in self.counted_ids:
                            if obj_data['total_dist'] > (height * 0.10):
                                self.counted_ids.add(id)

        cap.release()
        return len(self.counted_ids)

def process_wrapper(video_path):
    """
    Helper function to run inside each process.
    Instantiates Solution locally to avoid pickling issues/state conflicts.
    """
    try:
        solve = Solution()
        result = solve.forward(video_path)
        return f"[{os.path.basename(video_path)}] Completed: {result}"
    except Exception as e:
        return f"[{os.path.basename(video_path)}] Error: {e}"

if __name__ == "__main__":
    video_paths = glob.glob("./dataset/*.avi")
    video_paths2 = glob.glob("./dataset/*.mp4")
    video_paths.extend(video_paths2)

    print(f"Found {len(video_paths)} videos. Starting parallel processing...")
    print(video_paths)

    with ProcessPoolExecutor(max_workers=13) as executor:
        results = list(executor.map(process_wrapper, video_paths))

    print("\n--- Final Results ---")
    for res in results:
        print(res)
