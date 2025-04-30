from ultralytics import YOLO
import cv2
import pickle
import sys
sys.path.append('../') # 1 folder back
from utils import get_center_of_bbox, measure_distance

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    # Choose the closest humans from the court
    def choose_and_filter_players(self, court_keypoints, player_detections):
        player_detections_first_frame = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []

        for player_dict in player_detections:
            filtered_player_dict = {track_id : bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        
        return filtered_player_detections


    def choose_players(self, court_keypoints, player_dict):
        distances = []

        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            
            min_distance = float('inf')
            # # Calculate distance between player and every keypoints of the court
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i+1]) # x,y
                distance = measure_distance(player_center, court_keypoint)
                # Get the shortest distance between the player and any key points of the court 
                # i.e., The referee or the ball boys might be close to the court too, but the inner court keypoints have a shorter distance to the players
                if distance < min_distance:
                    min_distance = distance
            
            distances.append((track_id, min_distance))

        # Sort the distances in ascending order
        distances.sort(key=lambda x : x[1])
        # Take the top 2 tracks - shortest distance from the keypoints
        chosen_players = [distances[0][0], distances[1][0]]

        return chosen_players

    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)
        
        return player_detections

    # each frame is basically an image
    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0] # persist=True, tells the model to persist the tracking through different frames
        id_name_dict = results.names

        player_dict = {}
        # Only select the bounding boxes that are people not objects or anything else
        for box in results.boxes:
            track_id = int(box.id.tolist()[0]) # tracker id
            result = box.xyxy.tolist()[0] # bounding box
            object_cls_id = box.cls.tolist()[0] # object class id
            object_cls_name = id_name_dict[object_cls_id] # object class name
            # if it is a person
            if object_cls_name == 'person':
                player_dict[track_id] = result
        
        return player_dict
    

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            # Draw bounding boxes
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2) # only the borders, not filled

            output_video_frames.append(frame)

        return output_video_frames