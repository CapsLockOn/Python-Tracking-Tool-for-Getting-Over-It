import cv2
import csv
import threading
import numpy as np
import time
import os
from yt_dlp import YoutubeDL
from urllib.parse import urlparse, parse_qs
import multiprocessing

# ---------- CONFIGURATION ---------- #
video_list = [
    "https://www.youtube.com/watch?v=6V6vlWyMlYE",
    "https://www.youtube.com/watch?v=wSONmEHR8r4",
    "https://www.youtube.com/watch?v=LPmlYQ1Z-OQ",
    "https://www.youtube.com/watch?v=vzLUm4Hz-ZQ",
    "https://www.youtube.com/watch?v=J5BrXsh8Dj8",
    "https://www.youtube.com/watch?v=qkIVbVEmbSU",
    "https://www.youtube.com/watch?v=i4C67Voq3SM",
    "https://www.youtube.com/watch?v=LgnCIgXSR2Y",
    "https://www.youtube.com/watch?v=QtzxP-eC2Bk",
    "https://www.youtube.com/watch?v=9YVUeef6bZE"
]
video_list = list(dict.fromkeys(video_list))  # Remove duplicates

output_dir = "video_tracking_data"
download_dir = "downloaded_videos"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(download_dir, exist_ok=True)

fall_threshold = 20
writeFreq = 2
ignore_region = (0, 0, 200, 200)

farneback_params = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# ---------- HELPERS ---------- #
def is_url(path):
    return urlparse(path).scheme in ("http", "https")

def extract_start_time_seconds(url):
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    if 't' in query_params:
        time_str = query_params['t'][0]
        if time_str.endswith('s'):
            return int(time_str[:-1])
        else:
            return int(time_str)
    return 0

def download_youtube_video(url):
    try:
        start_time = extract_start_time_seconds(url)

        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': os.path.join(download_dir, '%(title).100s.%(ext)s'),
            'quiet': True,
            'skip_download': True  # Only check for existence
        }

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            filename = ydl.prepare_filename(info)
            if not filename.endswith(".mp4"):
                filename = filename.rsplit(".", 1)[0] + ".mp4"

            if os.path.exists(filename):
                print(f"[INFO] Video already downloaded: {filename}")
                return filename, start_time

            print(f"[INFO] Downloading video: {filename}")
            ydl_opts['skip_download'] = False
            with YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([url])

            return filename, start_time

    except Exception as e:
        print(f"[ERROR] Failed to download video from {url}: {e}")
        return None, 0

def extract_frame_features(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.ones_like(frame_gray)
    x, y, w, h = ignore_region
    mask[y:y+h, x:x+w] = 0
    return cv2.bitwise_and(frame_gray, frame_gray, mask=mask)

# ---------- MAIN TRACKING FUNCTION ---------- #
def process_video(video_path, start_time=0):
    print(f"[START] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    if start_time > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        print(f"[INFO] Jumped to start time: {start_time} seconds")

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    file_path = os.path.join(output_dir, f"{base_name}_output.csv")

    csvfile = open(file_path, 'w', newline='')
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Time", "Progress", "Event"])

    progress = 0
    timeVariable = 0
    tracking_initialized = False
    prev_frame_gray = None
    timer_stop_flag = False

    start_processing_time = time.time()
    max_processing_duration = 3600  # 1 hour in seconds

    def timer():
        nonlocal timeVariable
        if not timer_stop_flag:
            csv_writer.writerow([timeVariable, progress])
            timeVariable += writeFreq
            threading.Timer(writeFreq, timer).start()

    timer()

    while cap.isOpened():
        if time.time() - start_processing_time >= max_processing_duration:
            print(f"[INFO] Reached 1-hour limit for: {video_path}")
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not tracking_initialized:
            prev_frame_gray = extract_frame_features(frame)
            tracking_initialized = True
            continue

        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, frame_gray, None, **farneback_params)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        avg_movement_y = np.mean(flow[..., 1])

        direction = ""
        if avg_movement_y > 3:
            direction = "Up"
            progress += 1
        elif avg_movement_y < -3:
            direction = "Down"
            progress -= 1

        prev_frame_gray = frame_gray.copy()

    timer_stop_flag = True
    cap.release()
    csvfile.close()
    print(f"[DONE] Finished processing: {video_path}")

# ---------- PROCESSING FUNCTION FOR MULTIPROCESSING ---------- #
def process_item(item):
    print(f"[INFO] Preparing to process: {item}")
    if is_url(item):
        local_path, start_time = download_youtube_video(item)
        if not local_path:
            print(f"[ERROR] Skipping video (download failed): {item}")
            return
    else:
        local_path = item
        start_time = 0

    base_name = os.path.splitext(os.path.basename(local_path))[0]
    output_file_path = os.path.join(output_dir, f"{base_name}_output.csv")
    lock_file_path = output_file_path + ".lock"

    # Lock file check
    if os.path.exists(output_file_path):
        print(f"[SKIP] Already processed: {base_name}")
        return
    if os.path.exists(lock_file_path):
        print(f"[SKIP] Currently being processed: {base_name}")
        return

    try:
        with open(lock_file_path, 'w') as f:
            f.write("processing")

        print(f"[INFO] Starting processing for: {base_name}")
        process_video(local_path, start_time=start_time)

    except Exception as e:
        print(f"[ERROR] Failed to process {base_name}: {e}")

    finally:
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)

# ---------- ENTRY POINT ---------- #
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    num_processes = 3

    print(f"[SYSTEM] Using {num_processes} processes to speed up video processing...\n")

    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.map(process_item, video_list)
