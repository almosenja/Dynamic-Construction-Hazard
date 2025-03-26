from collections import deque, defaultdict
import numpy as np

class ActivityDetection:
    def __init__(self, waiting_frames, distance_threshold):
        self.tracks = defaultdict(lambda: deque(maxlen=2))
        self.tracks_class = defaultdict()
        self.status_history = defaultdict(lambda: deque(maxlen=waiting_frames))
        self.distance_threshold = distance_threshold
        self.area_threshold = 0.9

    def update_activity(self, detections):
        # Check if the detections contain tracker_id
        if not hasattr(detections, 'tracker_id') or detections.tracker_id is None:
            print("Activity status detection is not possible without tracker_id")
            return []

        active_ids = set()
        class_names = {}

        # Update tracks with the current detections
        for idx in range(len(detections)):
            tracker_id = detections.tracker_id[idx]
            class_name = detections.data["class_name"][idx]
            bbox = detections.xyxy[idx]

            self.tracks[tracker_id].append(bbox)
            class_names[tracker_id] = class_name
            active_ids.add(tracker_id)

        # Mark tracks not in the current frame as None
        for track_id in list(self.tracks.keys()):
            if track_id not in active_ids:
                self.status_history[track_id].append(None)
            elif len(self.tracks[track_id]) > 1:
                self.update_status(track_id)

        # Remove tracks that are not active for a certain number of frames
        for track_id in list(self.tracks.keys()):
            if all([d is None for d in self.tracks[track_id]]):
                del self.tracks[track_id]

        equipment_status = self.get_activity_status()
        excavator_associations = self.associate_boxes(class_names)

        for body_id, excavator_id in excavator_associations.items():
            if equipment_status.get(excavator_id) == 'active':
                equipment_status[body_id] = 'active'
            else:
                equipment_status[body_id] = 'idle'

        return equipment_status

    def update_status(self, track_id):
        # Calculate the distance between the current and previous points
        xc1, yc1, xc2, yc2 = map(int, self.tracks[track_id][1])
        xp1, yp1, xp2, yp2 = map(int, self.tracks[track_id][0])
        current_point = ((xc1 + xc2) // 2, (yc1 + yc2) // 2)
        previous_point = ((xp1 + xp2) // 2, (yp1 + yp2) // 2)
        distance = np.linalg.norm(np.array(current_point) - np.array(previous_point))

        # Update the status history
        if distance > self.distance_threshold:
            self.status_history[track_id].append("active")
        else:
            self.status_history[track_id].append("idle")

    def get_activity_status(self):
        # Get the activity status for each track
        activity_status = {}
        for track_id, status in self.status_history.items():
            if "active" in status:
                activity_status[track_id] = "active"
            else:
                activity_status[track_id] = "idle"
        return activity_status
    
    def is_box_within_another(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2

        area_box1 = max(0, x2 - x1) * max(0, y2 - y1)
        area_box2 = max(0, x4 - x3) * max(0, y4 - y3)
        
        # Calculate intersection area
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Check if at least `threshold`% of the smaller box is within the intersection area
        return (intersection_area / area_box2) >= self.area_threshold
    
    def associate_boxes(self, class_names):
        associations = {}

        for exc_id in list(class_names.keys()):
            if class_names[exc_id] == "excavator":
                bbox_exc = self.tracks[exc_id][-1]

                for body_id in list(class_names.keys()):
                    if class_names[body_id] == "excavator_body":
                        bbox_body = self.tracks[body_id][-1]

                        if self.is_box_within_another(bbox_exc, bbox_body):
                            associations[body_id] = exc_id
                            break

        return associations