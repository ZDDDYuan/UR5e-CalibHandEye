
import os
import cv2
import time
import numpy as np
import cv2.aruco as aruco
import os.path as osp
from datetime import datetime
from typing import Tuple

from utils.realsense_d435i import RealSenseD435iCamera
from utils.ur5_robot import UR5Robot

ARUCO_DICT = aruco.DICT_4X4_50
MARKER_ID = 2
MARKER_LENGTH = 0.05

SUBFOLDER_NAME = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_DIR = osp.join("/home/zoom3d/project/UR-Robotics/calib_handeye_data", SUBFOLDER_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)

camera = RealSenseD435iCamera()
robot = UR5Robot(is_use_robotiq3f=False)

R_g2b, t_g2b = [], []
R_t2c, t_t2c = [], []

gripper2base_list, target2camera_list = [], []

def detect_aruco_pose(
    img_bgr: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    aruco_dict: int,
    marker_id: int,
    marker_length: float,
    ) -> Tuple[np.ndarray, np.ndarray] | None:
    """
    Detect ArUco marker and return (rvec, tvec).
    Returns None if detection fails.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
    parameters = cv2.aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or MARKER_ID not in ids.flatten():
        return None

    # index of our marker id
    idx = int(np.where(ids.flatten() == marker_id)[0][0])
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        [corners[idx]], marker_length, camera_matrix, dist_coeffs
    )
    return rvec, tvec

def rvec_tvec_transform(rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform the rvec and tvec to rotation matrix and translation vector, respectively
    """
    R, _ = cv2.Rodrigues(rvec[0][0])
    t = tvec[0][0].reshape(3, 1)
    return R, t

def take_samples(init_tcp_pose: np.ndarray) -> None:
    """
    Take samples for eye-in-hand calibration
    """
    rx_raid_range = [np.deg2rad(-10), np.deg2rad(30)]
    ry_raid_range = [np.deg2rad(-20), np.deg2rad(20)]
    rz_raid_range = [np.deg2rad(-10), np.deg2rad(10)]

    workspace_rx = np.linspace(rx_raid_range[0], rx_raid_range[1], 3)
    workspace_ry = np.linspace(ry_raid_range[0], ry_raid_range[1], 4)
    workspace_rz = np.linspace(rz_raid_range[0], rz_raid_range[1], 2)

    grid_rx, grid_ry, grid_rz = np.meshgrid(workspace_rx, workspace_ry, workspace_rz)
    calib_rotations = np.stack([grid_rx.ravel(), grid_ry.ravel(), grid_rz.ravel()], axis=-1)
    calib_rotations = np.vstack([init_tcp_pose[3:].ravel(), calib_rotations])

    delta_trans = 0.01
    
    num_samples = calib_rotations.shape[0]
    print(f"[INFO] The number of samples for calibration: {num_samples}")

    for idx, calib_rot in enumerate(calib_rotations):
        if idx % 2 == 0 and idx != 0:
            calib_trans = init_tcp_pose[:3] + delta_trans
        elif idx % 2 != 0:
            calib_trans = init_tcp_pose[:3] - delta_trans
        elif idx == 0:
            calib_trans = init_tcp_pose[:3]
        calib_tcp_pose = np.append(calib_trans, calib_rot)
        robot.move_j_p(calib_tcp_pose, k_acc=2.0, k_vel=2.0)
        print(f"[INFO] Ready to take sample {idx + 1}/{num_samples} — pose: {calib_tcp_pose.tolist()}")

        while True:
            img_rgb, _ = camera.get_frames()
            img_show = img_rgb.copy()
            camera_matrix, dist_coeffs = camera.intrinsics, camera.coeffs
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            result = detect_aruco_pose(
                img_bgr,
                camera_matrix,
                dist_coeffs,
                aruco_dict=ARUCO_DICT,
                marker_id=MARKER_ID,
                marker_length=MARKER_LENGTH,
            )

            if result is not None:
                rvec, tvec = result
                cv2.drawFrameAxes(img_show, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

            cv2.putText(img_show, "Press 's' to take sample, 'q' to quit", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("Aruco markers", img_show)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'):
                print(f"[INFO] Sample {idx + 1} captured.")

                # Get T_gripper2base
                curr_tcp_pose = np.array(robot.get_current_tcp())
                R_gripper2base, _ = cv2.Rodrigues(curr_tcp_pose[3:])
                t_gripper2base = curr_tcp_pose[:3].reshape(3, 1)
                R_g2b.append(R_gripper2base)
                t_g2b.append(t_gripper2base)

                # Get T_target2camera
                if result is not None:
                    rvec, tvec = result
                    R_target2camera, t_target2camera = rvec_tvec_transform(rvec, tvec)
                    R_t2c.append(R_target2camera)
                    t_t2c.append(t_target2camera)

                break
            elif key == ord('q'):
                print("[INFO] Sampling interrupted by user.")
                cv2.destroyAllWindows()
                return

    print("[INFO] Sampling complete.")
    cv2.destroyAllWindows()

    if len(R_g2b) and len(R_t2c):
        print("[INFO] Saving transformation data...")
        save_transformation()
        print("[INFO] All data saved to", SAVE_DIR)

def save_transformation():
    global gripper2base_list, target2camera_list
    for i in range(len(R_g2b)):
        T_gripper2base = np.eye(4)
        T_gripper2base[:3, :3] = R_g2b[i]
        T_gripper2base[:3, 3] = t_g2b[i].ravel()
        gripper2base_list.append(T_gripper2base)

        T_target2camera = np.eye(4)
        T_target2camera[:3, :3] = R_t2c[i]
        T_target2camera[:3, 3] = t_t2c[i].ravel()
        target2camera_list.append(T_target2camera)

    np.save(osp.join(SAVE_DIR, "gripper2base.npy"), np.array(gripper2base_list))
    np.save(osp.join(SAVE_DIR, "target2cam.npy"), np.array(target2camera_list))

def calibrate_eye_in_hand():
    R_camera2gripper, t_camera2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        method=cv2.CALIB_HAND_EYE_DANIILIDIS
    )
    T_camera2gripper = np.eye(4)
    T_camera2gripper[:3, :3] = R_camera2gripper
    T_camera2gripper[:3, 3] = t_camera2gripper.ravel()

    np.savetxt(osp.join(SAVE_DIR, "camera2gripper.txt"), T_camera2gripper, delimiter=" ")
    print("[INFO] T_cam2gripper has saved to", SAVE_DIR)

def main():
    # Move to initial pose
    init_tcp_pose = np.array([0.15, -0.30, 0.18, 0.0, 0.0, 0.0])
    robot.move_j_p(init_tcp_pose, k_acc=2.0, k_vel=3.0)
    time.sleep(2)
    print("[INFO] UR5e Robot has reached initial pose...")

    input("[INFO] Press Enter to start sampling...")

    # Take samples for calibration
    take_samples(init_tcp_pose)
    # Calculate T_camera2gripper
    if len(R_g2b) and len(R_t2c):
        calibrate_eye_in_hand()
    else:
        print("[INFO] There is not sufficient data for calibration.")


if __name__ == "__main__":
    main()