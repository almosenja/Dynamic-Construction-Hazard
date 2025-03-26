import numpy as np
import cv2
from tqdm import tqdm

def undistort_video(video_path, output_path, intrinsics, dist):
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare the output video writer
    output_video_path = output_path
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Compute the optimal new camera matrix and ROI
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist, (frame_width, frame_height), 1, (frame_width, frame_height))

    # Process the video with a progress bar
    with tqdm(total=frame_count, desc="Processing Video", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            undistorted_frame = cv2.undistort(frame, intrinsics, dist, None, new_camera_matrix)
            undistorted_frame = cv2.resize(undistorted_frame, (frame_width, frame_height))

            out.write(undistorted_frame)
            pbar.update(1)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    out.release()

    print("Video processing complete. The undistorted video is saved as", output_path)


# Test
if __name__ == "__main__":
    video_path = "video/source/path"
    output_path = "video/output/path"

    intrinsics = np.array([[1.6533334e+03, 0.0000000e+00, 9.6000000e+02],
                        [0.0000000e+00, 1.3950000e+03, 5.4000000e+02],
                        [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
    dist = np.array([[-0.175, 0, 0, 0, 0]])
    undistort_video(video_path, output_path, intrinsics, dist)