import time
import socket
import struct
import numpy as np
from typing import Sequence, Dict, Tuple, List

from utils.robotiq3f_gripper import Robotiq3FGripper
# from realsense_d435i import RealSenseD435iCamera


class UR5Robot:
    def __init__(
        self,
        tcp_host: str = "192.168.1.20",
        tcp_port: int = 30003,
        is_use_robotiq3f: bool =True,
        is_use_camera: bool = True,
    ):
        # Initialization
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.is_use_robotiq3f = is_use_robotiq3f
        self.is_use_camera = is_use_camera
        self.tcp_socket = None

        # Define the acceleration and velocity of joint and tool
        self.joint_acc = 0.5
        self.joint_vel = 0.2
        self.tool_acc = 0.5
        self.tool_vel = 0.2

        # Define the tolerance of joint and tool pose
        self.joint_tolerance = 0.01
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]
        
        self.cam2gripper_mat = np.loadtxt("./calib_handeye_data/20250709_202641/camera2gripper.txt")
        self.cam_depth_scale = 0.0010000000474974513

        if is_use_robotiq3f:
            self.gripper = Robotiq3FGripper()
            self.gripper.activate()
            self.open_gripper(wait=False)

    def close(self) -> None:
        """
        Close the connection to UR5 robot.
        """
        if self.tcp_socket:
            self.tcp_socket.close()
            self.tcp_socket = None
            
        if self.is_use_robotiq3f:
            self.gripper.close()

    def get_current_tcp(self) -> Tuple[float, ...]:
        """
        Get current tcp pose.
        """
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host, self.tcp_port))
        data: bytes = self.tcp_socket.recv(1108)
        self.tcp_socket.close()
        return self.parse_tcp_data(data)["Tool vector actual"]
        
    def get_current_joint_config(self) -> Tuple[float, ...]:
        """
        Get current joint configuration.
        """
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host, self.tcp_port))
        data: bytes = self.tcp_socket.recv(1108)
        self.tcp_socket.close()
        return self.parse_tcp_data(data)["q actual"]
        
    def move_j(
        self,
        target_joint_config: Sequence[float],
        k_acc: float = 1.0,
        k_vel: float = 1.0,
        t: float = 0.0,
        r: float = 0.0,
    ) -> None:
        """
        Move the UR5 robot to a specified joint configuration.
        """
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host, self.tcp_port))

        joint_ang = ", ".join(f"{ang: .6f}" for ang in target_joint_config)
        tcp_command = (
            f"movej([{joint_ang}], "
            f"a={k_acc * self.joint_acc: .6f}, v={k_vel * self.joint_vel: .6f}, "
            f"t={t: .6f}, r={r: .6f})\n"
        )
        self.tcp_socket.send(str.encode(tcp_command))

        curr_joint_config = self.get_current_joint_config()
        while not all([np.abs(curr_joint_config[i] - target_joint_config[i]) < self.joint_tolerance for i in range(6)]):
            curr_joint_config = self.get_current_joint_config()
            time.sleep(0.01)
        self.tcp_socket.close()

    def move_j_p(
        self,
        target_tcp_pose: Sequence[float],
        k_acc: float = 1.0,
        k_vel: float = 1.0,
        t: float = 0.0,
        r: float = 0.0,
    )-> None:
        """
        Move the UR5 robot to a specified joint configuration by solving inverse kinematics.
        """
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host, self.tcp_port))

        x, y, z, roll, pitch, yaw = target_tcp_pose

        tcp_command = "def process():\n"
        tcp_command += f"  array = rpy2rotvec([{roll:.6f}, {pitch:.6f}, {yaw:.6f}])\n"
        tcp_command += (
            f"  movej(get_inverse_kin(p[{x:.6f}, {y: .6f}, {z: .6f}, "
            f"array[0], array[1], array[2]]), "
            f"a={k_acc * self.joint_acc:.6f}, "
            f"v={k_vel * self.joint_vel:.6f}, "
            f"t={t:.6f}, r={r:.6f})\n"
        )
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        curr_tcp_pose = self.get_current_tcp()
        while not all([np.abs(curr_tcp_pose[i] - target_tcp_pose[i]) < self.tool_pose_tolerance[i] for i in range(3)]):
            curr_tcp_pose = self.get_current_tcp()
            time.sleep(0.01)
        self.tcp_socket.close()

    def move_l(
        self,
        target_tcp_pose: Sequence[float],
        k_acc: float = 1.0,
        k_vel: float = 1.0,
        t: float = 0.0,
        r: float = 0.0,
    ) -> None:
        """
        Linear move the UR5 robot to a specified position.
        """
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_socket.connect((self.tcp_host, self.tcp_port))

        x, y, z, roll, pitch, yaw = target_tcp_pose

        tcp_command = "def process():\n"
        tcp_command += f"  array = rpy2rotvec([{roll:.6f}, {pitch:.6f}, {yaw:.6f}])\n"
        tcp_command += (
            f"  movel(p[{x:.6f}, {y: .6f}, {z: .6f}, "
            f"array[0], array[1], array[2]], "
            f"a={k_acc * self.joint_acc:.6f}, "
            f"v={k_vel * self.joint_vel:.6f}, "
            f"t={t:.6f}, r={r:.6f})\n"
        )
        tcp_command += "end\n"
        self.tcp_socket.send(str.encode(tcp_command))

        curr_tcp_pose = self.get_current_tcp()
        while not all([np.abs(curr_tcp_pose[i] - target_tcp_pose[i]) < self.tool_pose_tolerance[i] for i in range(3)]):
            curr_tcp_pose = self.get_current_tcp()
            time.sleep(0.01)
        self.tcp_socket.close()

    def parse_tcp_data(self, data: bytes) -> Dict[str, Tuple[float, ...]]:
        """
        Parse the binary TCP data stream received from the UR5 robot into a dictionary of named values.
        """
        recv_dict = {"MessageSize": "i", "Time": "d", "q target": "6d", "qd target": "6d", "qdd target": "6d", 
            "I target": "6d",
            "M target": "6d", "q actual": "6d", "qd actual": "6d", "I actual": "6d", "I control": "6d",
            "Tool vector actual": "6d", "TCP speed actual": "6d", "TCP force": "6d", "Tool vector target": "6d",
            "TCP speed target": "6d", "Digital input bits": "d", "Motor temperatures": "6d", "Controller Timer": "d",
            "Test value": "d", "Robot Mode": "d", "Joint Modes": "6d", "Safety Mode": "d", "empty1": "6d",
            "Tool Accelerometer values": "3d",
            "empty2": "6d", "Speed scaling": "d", "Linear momentum norm": "d", "SoftwareOnly": "d",
            "softwareOnly2": "d", "V main": "d",
            "V robot": "d", "I robot": "d", "V actual": "6d", "Digital outputs": "d", "Program state": "d",
            "Elbow position": "d", "Elbow velocity": "3d"}
        
        buffer_dict: Dict[str, Tuple[float, ...]] = {}
        for k in recv_dict:
            fmtsize = struct.calcsize(recv_dict[k])
            data_extracted, data = data[0:fmtsize], data[fmtsize:]
            fmt = "!" + recv_dict[k]
            buffer_dict[k] = struct.unpack(fmt, data_extracted)
        return buffer_dict
    
    def open_gripper(
        self,
        target_position: List[int] = [0, 0, 0],
        speed: List[int] = [250, 250, 250],
        force: List[int] = [250, 250, 250],
        mode: str = "Pinch",
        individual_control: bool = True,
        wait: bool = True,
        timeout_sec: float = 5.0,
        tolerance: int = 5,
    ) -> None:
        """
        Open gripper to target position. Optionally wait until motion is complete.
        """
        target_position = [target_position[0]] * 3 if not individual_control else target_position

        self.gripper.command_gripper(
            rPRA=target_position,
            rSP=speed, 
            rFR=force, 
            rMOD=mode, 
            rICF=individual_control,
        )
        
        if not wait:
            return  # No-blocking mode

        timeout = time.time() + timeout_sec
        while time.time() < timeout:
            self.gripper.status()

            # Motion complete or get stuck by an object (gIMC != 0)
            if hasattr(self.gripper, "gIMC") and self.gripper.gIMC != 0:
                print(f"[INFO] Gripper motion complete (gIMC = {self.gripper.gIMC})")
                return

            # Or relying on FingerX_Position to jump out block
            a, b, c = self.gripper.FingerA_Position, self.gripper.FingerB_Position, self.gripper.FingerC_Position
            diffs = [abs(a - target_position[0]), abs(b - target_position[1]), abs(c - target_position[2])]
            if all(d <= tolerance for d in diffs):
                print("[INFO] Gripper fingers reached target (within tolerance).")
                return
            time.sleep(0.1)

        print("[WARNING] Gripper open timeout — finger did not reach target or gIMC stuck.")

    def close_gripper(
        self,
        target_position: List[int] = [100, 100, 100],
        speed: List[int] = [250, 250, 250],
        force: List[int] = [250, 250, 250],
        mode: str = "Pinch",
        individual_control: bool = True,
        wait: bool = True,
        timeout_sec: float = 5.0,
        tolerance: int = 5,
    ) -> None:
        """
        Close gripper to target position. Optionally wait until motion is complete.
        """
        target_position = [target_position[0]] * 3 if not individual_control else target_position

        self.gripper.command_gripper(
            rPRA=target_position,
            rSP=speed, 
            rFR=force, 
            rMOD=mode, 
            rICF=individual_control,
        )

        if not wait:
            return  # No-blocking mode

        timeout = time.time() + timeout_sec
        while time.time() < timeout:
            self.gripper.status()

            # Motion complete or get stuck by an object (gIMC != 0)
            if hasattr(self.gripper, "gIMC") and self.gripper.gIMC != 0:
                print(f"[INFO] Gripper motion complete (gIMC = {self.gripper.gIMC})")
                return

            # Or relying on FingerX_Position to jump out block
            a, b, c = self.gripper.FingerA_Position, self.gripper.FingerB_Position, self.gripper.FingerC_Position
            diffs = [abs(a - target_position[0]), abs(b - target_position[1]), abs(c - target_position[2])]
            if all(d <= tolerance for d in diffs):
                print("[INFO] Gripper fingers reached target (within tolerance).")
                return
            time.sleep(0.1)

        print("[WARNING] Gripper close timeout — finger did not reach target or gIMC stuck.")

    

if __name__ == "__main__":
    ur5_robot = UR5Robot(is_use_robotiq3f=False)
    curr_tcp = ur5_robot.get_current_tcp()
    print(curr_tcp)
    curr_joint_config = ur5_robot.get_current_joint_config()
    print(np.degrees(curr_joint_config))

    target_tcp_pose = [0.15, -0.28039290004612755, 0.1833946785635515, 
                       0.0, 0.0, 0.0]
    ur5_robot.move_j_p(target_tcp_pose)
    ur5_robot.close()