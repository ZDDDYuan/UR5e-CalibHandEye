import cv2
import numpy as np
from threading import Thread
from scipy.spatial.transform import Rotation as R

from utils.realsense_d435i import RealSenseD435iCamera
from utils.ur5_robot import UR5Robot
    

# --------------- Setup options ---------------
init_tcp_pose = [0.15, -0.28039290004612755, 0.1833946785635515, 
                 -0.004800856693334966, 0.021274021001852574, 0.04240988913491252]
init_place_tcp_pose = [0.16839538732615553, -0.7216761768618855, 0.080325445839348846, 
                  -0.00527696813364512, 0.021188515201630422, 0.04248750641256021]
place_position_delta = [[0.0, 0.0, 0.0], 
                        [0.10, 0.0, 0.0], 
                        [0.0, -0.05, 0.0], 
                        [0.10, -0.05, 0.0]]
place_index = 0
tool_orientation = [-0.004800856693334966, 0.021274021001852574, 0.04240988913491252]

# UR5 Robot init
robot = UR5Robot()
# RealSense Camera init
camera = RealSenseD435iCamera()

robot.move_j_p(init_tcp_pose)

# Callback function for clicking on OpenCV window
click_point_pix = ()
camera_color_img, camera_depth_img = camera.get_frames()

def pick_and_place(x, y):
    global camera, robot, click_point_pix, place_index
    click_point_pix = (x,y)

    place_xyz = np.array(init_place_tcp_pose[:3]) + np.array(place_position_delta[place_index])
    place_rpy = init_place_tcp_pose[3:]
    place_tcp_pose = np.append(place_xyz, place_rpy).tolist()
    place_index = (place_index + 1) % len(place_position_delta)

    # Get click point in camera coordinates
    depth_val = camera_depth_img[y][x]
    if depth_val == 0:
        print("Invalid depth")
        return

    click_z = (depth_val * robot.cam_depth_scale).item()

    fx = camera.intrinsics[0][0]
    fy = camera.intrinsics[1][1]
    cx = camera.intrinsics[0][2]
    cy = camera.intrinsics[1][2]

    click_x = ((x - cx) * click_z / fx).item()
    click_y = ((y - cy) * click_z / fy).item()

    p_cam = np.array([[click_x], [click_y], [click_z], [1.0]])

    T_camera2gripper = robot.cam2gripper_mat

    curr_tcp_pose = robot.get_current_tcp()
    curr_tcp_position = curr_tcp_pose[:3]
    curr_tcp_orientation, _ = cv2.Rodrigues(np.array(curr_tcp_pose[3:]))
    T_gripper2base = np.eye(4)
    T_gripper2base[:3, :3] = curr_tcp_orientation
    T_gripper2base[:3, 3] = curr_tcp_position

    tool_position = T_gripper2base @ T_camera2gripper @ p_cam
    tool_position[1] += 0.012
    
    target_tcp_pose = np.append(tool_position.flatten()[:3], tool_orientation)
    print(f"[INFO] target tcp pose: {target_tcp_pose.tolist()}")

    # Move above the target
    aboved_tcp_pose = target_tcp_pose.copy()
    aboved_tcp_pose[2] = init_tcp_pose[2]
    robot.move_j_p(aboved_tcp_pose, k_acc=1.5, k_vel=2)

    robot.move_l(target_tcp_pose, k_acc=1.5, k_vel=2)
    robot.close_gripper(force=[50, 50, 50], wait=False)
    robot.move_j_p(place_tcp_pose)
    robot.open_gripper(wait=False)
    robot.move_j_p(init_tcp_pose)

def mouseclick_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        Thread(target=pick_and_place, args=(x, y)).start()


if __name__ == "__main__":
    # Show color and depth frames
    cv2.namedWindow('color')
    cv2.setMouseCallback('color', mouseclick_callback)
    cv2.namedWindow('depth')

    while True:
        camera_color_img, camera_depth_img = camera.get_frames()
        rgb_data = camera_color_img

        if len(click_point_pix) != 0:
            rgb_data = cv2.circle(camera_color_img, click_point_pix, 7, (0,0,255), 2)
        cv2.imshow('color', rgb_data)
        cv2.imshow('depth', camera_depth_img)
        
        if cv2.waitKey(1) == ord('q'):
            robot.move_j_p(init_tcp_pose)
            break
        elif cv2.waitKey(1) == ord(' '):
            robot.move_j_p(init_tcp_pose)
            continue

    cv2.destroyAllWindows()
    robot.close()