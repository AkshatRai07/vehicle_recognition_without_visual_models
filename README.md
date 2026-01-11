# **Vehicle Counter (Computer Vision)**

A lightweight, parallelized Python application that estimates vehicle counts from video footage using classical computer vision techniques. It utilizes background subtraction, morphological operations, and centroid tracking to detect and count moving vehicles without relying on heavy deep learning models.

---

## **Features**

* **Classical CV Approach:** Uses MOG2 Background Subtraction and Euclidean Distance Tracking (no GPU required).
* **Parallel Processing:** Leverages `ProcessPoolExecutor` to process multiple video files simultaneously, maximizing CPU core usage.
* **Dynamic filtering:**
* **Region of Interest (ROI):** Applies a trapezoidal mask to focus on the road area.
* **Direction Locking:** Automatically detects the primary traffic flow direction to filter out noise or reverse traffic.
* **Aspect Ratio & Size Filtering:** Rejects non-vehicle objects (e.g., pedestrians, noise) based on bounding box dimensions.
* **Format Support:** Automatically detects and processes `.avi` and `.mp4` files.

---

## **Prerequisites**

Ensure you have Python 3.x installed. The project relies on the following external libraries:

* **OpenCV** (`cv2`): For image processing and video manipulation.
* **NumPy**: For matrix operations and efficient calculations.

---

## **Installation**

1. **Download this repo**


2. **Create and activate a virtual environment (Recommended):**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```


3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## **Project Structure**

The script expects a specific directory layout to locate input videos. Ensure your folder looks like this:

```text
├── main.py               # The main script provided
├── venv/                 # Virtual environment
└── dataset/              # [CRITICAL] Put your video files here
    ├── video1.mp4
    ├── video2.avi
    └── ...
```

---

## **Usage**

1. **Prepare Data:** Place all video files you want to analyze inside the `dataset` folder.
2. **Run the script:**
```bash
python main.py
```

3. **Output:**
The script will print the processing status of detected videos and output the final count for each file once completed.
```text
Found 5 videos. Starting parallel processing...
['./dataset/traffic_cam_1.mp4', ...]

--- Final Results ---
[traffic_cam_1.mp4] Completed: 42
[traffic_cam_2.avi] Completed: 15
...
```

---

## **Configuration**

You can tweak the following parameters in `main.py` to optimize performance for your specific hardware or video conditions:

* **Concurrency:**
* Update `max_workers=13` in the `if __name__ == "__main__":` block to match your CPU's thread count for optimal parallel performance.

* **Detection Sensitivity:**
* `min_area`: Adjust inside `Solution.forward` to filter smaller or larger objects.
* `varThreshold`: Adjust inside `bg_subtractor` (default `25`) to change how sensitive the background subtractor is to pixel changes.

* **Counting Line:**
* Adjust `zone_start` (0.40) and `zone_end` (0.60) in the `forward` method to change where in the frame vehicles are counted.

---

## **How It Works**

1. **Preprocessing:** A Region of Interest (ROI) mask is applied to ignore sky/scenery. The frame is blurred to reduce noise.
2. **Object Detection:** `MOG2` subtracts the static background. Morphological operations (Erode, Dilate, Close) clean the resulting binary mask to form solid blobs.
3. **Tracking:** Centroids of blobs are calculated. The `EuclideanDistTracker` associates centroids across frames to assign unique IDs.
4. **Counting:**
* The system calculates the average vertical movement (`avg_dy`) of an object.
* Once a global traffic flow is established, objects moving against the flow or "jittering" in place are ignored.
* When a valid tracked object crosses the defined vertical zone, the counter increments.