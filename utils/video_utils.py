import cv2

# Read the video by passing it the video path
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        # if there are no more frames, ret will return False
        if not ret:
            break # break when there are no more frames left
        
        frames.append(frame)
    cap.release()

    return frames

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # output it as a 24 frames per second
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    
    out.release()




