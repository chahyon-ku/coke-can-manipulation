import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def make_and_save_templates():
    # importing template images
    temp_image_m = np.array(cv2.imread('0_color.png', cv2.IMREAD_GRAYSCALE))
    temp_image_l = np.array(cv2.imread('4_color.png', cv2.IMREAD_GRAYSCALE))
    temp_image_r = np.array(cv2.imread('1_color.png', cv2.IMREAD_GRAYSCALE))


    # Apply local threshholding to make the images more invariant
    # Normalizing images
    thresh_m = cv2.ximgproc.niBlackThreshold(temp_image_m, maxValue=255, type=cv2.THRESH_BINARY, blockSize=41, k=-0.2)
    thresh_l = cv2.ximgproc.niBlackThreshold(temp_image_l, maxValue=255, type=cv2.THRESH_BINARY, blockSize=41, k=-0.2)
    thresh_r = cv2.ximgproc.niBlackThreshold(temp_image_r, maxValue=255, type=cv2.THRESH_BINARY, blockSize=41, k=-0.2)

    # Templates
    template_m = thresh_m[112:198, 480:546]
    template_l = thresh_l[242:364, 211:311]
    template_r = thresh_r[145:249, 573:639]

    os.makedirs('templates', exist_ok=True)
    cv2.imwrite('templates/0.png', template_m)
    cv2.imwrite('templates/1.png', template_l)
    cv2.imwrite('templates/2.png', template_r)


def main():
    # intrinsic matrix
    intrinsic_matrix = get_intrinsic()
    f = intrinsic_matrix[0, 0]
    cam_pose = get_;

    templates = [np.array(cv2.imread('templates/0.png', cv2.IMREAD_GRAYSCALE)),
                 np.array(cv2.imread('templates/1.png', cv2.IMREAD_GRAYSCALE)),
                 np.array(cv2.imread('templates/2.png', cv2.IMREAD_GRAYSCALE))]

    # image we want to know the can location of

    current_image = np.array(cv2.imread('7_color.png', cv2.IMREAD_GRAYSCALE))
    current_depth_image = np.array(cv2.imread('7_depth.png', cv2.IMREAD_ANYDEPTH))
    position_image = cv2.ximgproc.niBlackThreshold(current_image, maxValue=255, type=cv2.THRESH_BINARY, blockSize=41, k=-0.2)

    # Similarity checking
    similarities = [cv2.matchTemplate(position_image[300:540, 200:800], template, cv2.TM_CCORR_NORMED)
                    for template in templates]

    max_peak = None
    peaks = []
    for i_similarity, similarity in enumerate(similarities):
        flat = np.reshape(similarity, -1)
        value = np.max(flat)
        loc = np.argmax(flat)
        peaks.append((value, 300 + templates[i_similarity].shape[0] // 2 + loc // similarity.shape[1],
                      200 + templates[i_similarity].shape[1] // 2 + loc % similarity.shape[1]))
        if max_peak is None:
            max_peak = peaks[-1]
        elif max_peak[0] < peaks[-1][0]:
            max_peak = peaks[-1]

    z = current_depth_image[max_peak[1], max_peak[2]] * 0.00025
    x = z * (max_peak[2] - 480) / f;
    y = z * (max_peak[1] - 270) / f;

    camRaruco = R.from_euler('xyz', cam_pose[3:]).as_matrix()

    camTaruco = np.zeros((4, 4))
    camTaruco[:3, :3] = camRaruco
    camTaruco[:3, 3] = cam_pose[:3]
    camTaruco[3, 3] = 1

    camTsoda = np.zeros((4, 4))
    camTsoda[:3, :3] = camRaruco
    camTsoda[:3, 3] = [x, y, z]
    camTsoda[3, 3] = 1


    arucoTsoda = np.linalg.inv(camTaruco) @ camTsoda

    arucorvecsoda, _ = cv2.Rodrigues(arucoTsoda[:3, :3])
    arucotvecsoda = arucoTsoda[:3, 3]

    camrvecaruco, _ = cv2.Rodrigues(camTaruco[:3, :3])
    camtvecaruco = camTaruco[:3, 3]
    camrvecsoda, _ = cv2.Rodrigues(camTsoda[:3, :3])
    camtvecsoda = camTsoda[:3, 3]


    intrinsic_matrix = np.array(((672.66, 0, 480), (0, 672.66, 270), (0, 0, 1)))
    current_image = cv2.drawFrameAxes(current_image, intrinsic_matrix*(960/1000), .01, camrvecaruco, camtvecaruco, .1)
    current_image = cv2.drawFrameAxes(current_image, intrinsic_matrix, .01, camrvecsoda, camtvecsoda, .1)
    plt.imshow(current_image)

    pose = np.zeros((1, 6))
    pose[0, :2] = arucotvecsoda[:2]
    return pose

    # for T in [camTaruco, camTsoda]:
    #     origin = T @ np.array([0., 0., 0., 1.])
    #     x_axis = T @ np.array([.1, 0., 0., 1.])
    #     y_axis = T @ np.array([0., .1, 0., 1.])
    #     z_axis = T @ np.array([0., 0., .1, 1.])
    #     origin = [origin[0] / origin[2] * f + 480, origin[1] / origin[2] * f + 270]
    #     x_axis = [x_axis[0] / x_axis[2] * f + 480, x_axis[1] / x_axis[2] * f + 270]
    #     y_axis = [y_axis[0] / y_axis[2] * f + 480, y_axis[1] / y_axis[2] * f + 270]
    #     z_axis = [z_axis[0] / z_axis[2] * f + 480, z_axis[1] / z_axis[2] * f + 270]
    #
    #     print(origin, x_axis, y_axis, z_axis)
    #
    #     plt.plot((origin[0], x_axis[0]), (origin[1], x_axis[1]), color='red', marker='o')
    #     plt.plot((origin[0], y_axis[0]), (origin[1], y_axis[1]), color='green', marker='o')
    #     plt.plot((origin[0], z_axis[0]), (origin[1], z_axis[1]), color='blue', marker='o')
    plt.show()


def get_aruco_t_soda(color_image, depth_image, intrinsic, aruco_t_camera):
    # intrinsic matrix
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    ppx = intrinsic[0, 2]
    ppy = intrinsic[1, 2]

    templates = [np.array(cv2.imread('templates/0.png', cv2.IMREAD_GRAYSCALE)),
                 np.array(cv2.imread('templates/1.png', cv2.IMREAD_GRAYSCALE)),
                 np.array(cv2.imread('templates/2.png', cv2.IMREAD_GRAYSCALE))]

    # image we want to know the can location of
    position_image = cv2.ximgproc.niBlackThreshold(color_image, maxValue=255, type=cv2.THRESH_BINARY, blockSize=41, k=-0.2)

    # Similarity checking
    similarities = [cv2.matchTemplate(position_image[300:540, 200:800], template, cv2.TM_CCORR_NORMED)
                    for template in templates]

    max_peak = None
    peaks = []
    for i_similarity, similarity in enumerate(similarities):
        flat = np.reshape(similarity, -1)
        value = np.max(flat)
        loc = np.argmax(flat)
        peaks.append((value, 300 + templates[i_similarity].shape[0] // 2 + loc // similarity.shape[1],
                      200 + templates[i_similarity].shape[1] // 2 + loc % similarity.shape[1]))
        if max_peak is None:
            max_peak = peaks[-1]
        elif max_peak[0] < peaks[-1][0]:
            max_peak = peaks[-1]

    z = depth_image[max_peak[1], max_peak[2]] * 0.00025
    x = z * (max_peak[2] - ppx) / fx
    y = z * (max_peak[1] - ppy) / fy

    aruco_T_camera = np.zeros((4, 4))
    aruco_T_camera[:3, :3] = R.from_euler('xyz', aruco_t_camera[3:]).as_matrix()
    aruco_T_camera[:3, 3] = aruco_t_camera[:3]
    aruco_T_camera[3, 3] = 1

    camera_T_soda = np.zeros((4, 4))
    camera_T_soda[:3, :3] = np.transpose(aruco_T_camera[:3, :3])
    camera_T_soda[:3, 3] = [x, y, z]
    camera_T_soda[3, 3] = 1

    aruco_T_soda = aruco_T_camera @ camera_T_soda
    aruco_tvec_soda = aruco_T_soda[:3, 3]
    aruco_rvec_soda = cv2.Rodrigues(aruco_T_soda[:3, :3])

    aruco_t_soda = np.concatenate((aruco_tvec_soda, aruco_rvec_soda), -1)
    aruco_t_soda = np.reshape(aruco_t_soda, -1)
    return aruco_t_soda


if __name__ == '__main__':
    main()