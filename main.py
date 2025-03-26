import numpy as np
import supervision as sv
import cv2
import time
import threading

from collections import deque, defaultdict
from collections import deque
from ultralytics import YOLO
from activity import ActivityDetection
from database import HistoryDatabase
from homography import get_homography_mtx, project_point

    
def generate_random_color(class_id):
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, 3).tolist())


def run_tracker_in_thread(video_path, output_path, model_path, camera_id, homography, src_zone=None):
    # Getting the video information
    video_info = sv.VideoInfo.from_video_path(video_path)
    w, h = video_info.resolution_wh

    # Model initialization
    detection_model = YOLO(model_path)
    tracker_model = sv.ByteTrack(frame_rate=60)
    smoother = sv.DetectionsSmoother()
    activity_detector = ActivityDetection(waiting_frames=30, distance_threshold=1)

    # Initalize polygon zone
    if src_zone is not None:
        detection_zone = sv.PolygonZone(src_zone, frame_resolution_wh=video_info.resolution_wh)

    # Database initialization
    db = HistoryDatabase("history.db", "table_cam_" + str(camera_id))
    current_time = time.time()
    next_capture_time = current_time + 5

    # Coordinates initialization
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    # Video capture
    cap = cv2.VideoCapture(video_path)

    # Output video writer initialization
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, video_info.fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply detection and tracking models
        results = detection_model(frame, iou=0.5, conf=0.6, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(results)
        if src_zone is not None:
            detections = detections[detection_zone.trigger(detections)]
        detections = tracker_model.update_with_detections(detections=detections)

        # Apply smoothing and activity detector
        detections = smoother.update_with_detections(detections)
        activity_status = activity_detector.update_activity(detections)

        today_date = time.strftime("%Y-%m-%d")
        time_hms = time.strftime("%H:%M:%S")
        time_hms_dashed = time.strftime("%H-%M-%S")

        data_collection = []

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=src_zone, color=sv.Color.RED)
        
        for track_id, class_id, box, class_name in zip(detections.tracker_id, detections.class_id, detections.xyxy, detections.data["class_name"]):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(class_id)
            track_id = int(track_id)
            status = activity_status.get(track_id, "idle")

            # Obtain the bottom coordinate point
            bottom_y_offset = 5 if class_name == "worker" else 15
            bottom_y = int(y2) - bottom_y_offset
            center_x = (int(x1) + int(x2)) // 2
            bottom_point = (center_x, bottom_y)

            # Draw bounding box and bottom point
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), generate_random_color(class_id), 2)
            cv2.circle(annotated_frame, bottom_point, 3, generate_random_color(class_id), -1)

            # Location processing
            if class_name != "excavator":
                if class_name == "excavator_body":
                    class_name = "excavator"

                projected_points = project_point(bottom_point=bottom_point, homography_mtx=homography)

                # Draw the label and bounding box
                if class_name == "worker":
                    speed = 0
                    label = f"{track_id}-{class_name}"
                else:
                    coordinates[track_id].append(projected_points)
                    if len(coordinates[track_id]) < video_info.fps / 2:
                        speed = 0
                        label = f"{track_id}-{class_name}-{status}"
                    elif status == "idle":
                        speed = 0
                        label = f"{track_id}-{class_name}-{status}-{speed:.2f} km/h"
                    else:
                        # Calculating velocity
                        coordinate_start = coordinates[track_id][0]
                        coordinate_end = coordinates[track_id][-1]
                        distance = np.linalg.norm(np.array(coordinate_start) - np.array(coordinate_end))
                        distance = distance / 1000
                        sec = len(coordinates[track_id]) / video_info.fps
                        speed = (distance / sec * 3.6)
                        if speed < 1:
                            speed = 0
                            if class_name == "dumptruck":
                                status = "idle"
                        if speed > 30:
                            speed = 30
                        label = f"{track_id}-{class_name}-{status}-{speed:.2f} km/h"

                # Draw the label of the object
                (label_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + label_width, y1), generate_random_color(class_id), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

                # Collecting data for database
                data = [track_id, class_name, camera_id, today_date, time_hms, projected_points[0], projected_points[1], -12000, status, speed]
                data_collection.append(data)

        # Data storing
        if time.time() >= next_capture_time:
            cv2.imwrite("outputs/ch" + f"{camera_id}-{time_hms_dashed}" + ".jpg", annotated_frame)
            print(f"DATA DB CAM {camera_id}: ", data_collection)
            try:
                db.add_many(data_collection)
                print("Data stored successfully")
            except:
                print("Skipped storing data")
                pass

            next_capture_time += 5

        cv2.imshow("Annotated Frame cam_id: " + str(camera_id), annotated_frame)

        # Write the annotated frame to the output video
        output_video.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

if __name__ == "__main__":
    # Create the tracker thread
    video_path1 = "video/path"
    video_path2 = "video/path"
    output_path1 = "output/path/file"
    output_path2 = "output/path/file"
    detection_model = "model/weight"

    # Camera 1 homography (Example)
    src_list_cam1 = [[504, 475], [1055, 444], [1026, 475], 
                     [1574, 520], [1537, 479], [1845, 800], 
                     [3, 675]] # Frame
    dst_list_cam1 = [[-44814, 59663], [-10912, 66264], [-17528, 52952], 
                     [10522, 38840], [17322, 52025], [-8282, 1230], 
                     [-60900, 27767]] # BIM
    
    homography1 = get_homography_mtx(src_list_cam1, dst_list_cam1)

    # Camera 2 homography (Example)
    src_list_cam2 = [[1136, 555], [1545, 563], [1757, 559], 
                     [1661, 480], [1980, 722], [218, 869], 
                     [726, 602]] # Frame
    dst_list_cam2 = [[8348, -43749], [9543, -25680], [7665, -20179], 
                     [-31908, -24103], [30730, -6514], [58662, -47987], 
                     [21430, -58173]] # BIM

    homography2 = get_homography_mtx(src_list_cam2, dst_list_cam2)
    detection_zone_cam2 = np.array([[723, 584], [1132, 566], [1259, 531], 
                                    [1517, 554], [1920, 575], [1920, 1080], 
                                    [0, 1080], [218, 814]]) # Optional

    # Thread used for the video file
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread,
                                       args=(video_path1, output_path1, detection_model, 1, homography1),
                                       daemon=True)
    tracker_thread2 = threading.Thread(target=run_tracker_in_thread,
                                       args=(video_path2, output_path2, detection_model, 2, homography2, detection_zone_cam2),
                                       daemon=True)

    # Start thread
    tracker_thread1.start()
    tracker_thread2.start()
    tracker_thread1.join()
    tracker_thread2.join()

    # Clean up and close windows
    cv2.destroyAllWindows()

    print("All threads have finished processing")
