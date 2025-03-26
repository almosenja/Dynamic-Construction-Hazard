import cv2
import numpy as np

def get_homography_mtx(src_list, dst_list):
    src_pts = np.array(src_list).reshape(-1, 1, 2)
    dst_pts = np.array(dst_list).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return np.array(H)

def project_point(bottom_point, homography_mtx):
    x = bottom_point[0]
    y = bottom_point[1]
    pts = np.float32([[x, y]])
    pts_projected = cv2.perspectiveTransform(pts[None, :, :], homography_mtx)
    xo = int(pts_projected[0][0][0])
    yo = int(pts_projected[0][0][1])

    return (xo, yo)

# Test
if __name__ == "__main__":
    src_list = [[1135, 562], [1516, 577], [1781, 559], [1664, 485], [1914, 1074], [41, 893], [743, 605]]
    dst_list = [[411, 582], [414, 500], [407, 476], [237, 493], [696, 442], [652, 608], [474, 655]]
    
    homography_mtx = get_homography_mtx(src_list=src_list, dst_list=dst_list)
    print(f"Homography matrix for projection:\n{homography_mtx}")