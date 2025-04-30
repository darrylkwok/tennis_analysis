def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    return (center_x, center_y)

# Measure distances between any 2 different points
def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1])**2)**0.5


# Get the player's foot position
def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1+x2)/2), y2)

def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')

    current_keypoint_idx = keypoint_indices[0] #Initialise the first candidate keypoint

    for keypoint_idx in keypoint_indices:
        keypoint = keypoints[keypoint_idx*2], keypoints[keypoint_idx*2+1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            current_keypoint_idx = keypoint_idx
    
    return current_keypoint_idx


def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1] # y_max - y_min


def measure_xy_distance(p1,p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def get_center_of_bbox(bbox):
    return (int((bbox[0]+bbox[2])/2),int((bbox[1]+bbox[3])/2))