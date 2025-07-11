# =====================================================================
# Based on original code from: https://github.com/baha2r/robotiq3f_py
# Original license: Apache 2.0
# Date: 2025-07-10
# =====================================================================

import time
import warnings
import threading
from typing import List
from pyModbusTCP.client import ModbusClient


class Robotiq3FGripper:
    def __init__(
        self, 
        host: str = "192.168.1.11", 
        port: int = 502, 
        unit_id: int = 9, 
        update_interval: float = 0.1,
    ):
        # Initialize Modbus client and update thread
        self.client = ModbusClient(host=host, port=port, auto_open=True, unit_id=unit_id)
        self.client.open()
        self.update_interval = update_interval
        self._running = True
        self._thread = threading.Thread(target=self._update_loop)
        self._thread.start()

    def _update_loop(self) -> None:
        """Continuously update gripper status."""
        while self._running:
            self._update_status()
            time.sleep(self.update_interval)

    def _update_status(self) -> None:
        """Update and store the gripper status from Modbus server."""
        try:
            readData = self.client.read_input_registers(0, 8)
            
            if not readData or len(readData) < 8:
                print(f"[ERROR] Invalid Modbus response: {readData}")
                return

            self.gSTA, self.gIMC, self.gGTO, self.gMOD, self.gACT, self.gDTS, self.gDTC, self.gDTB, self.gDTA = \
                self.stat(self.add_leading_zeros(bin(readData[0])))

            self.FaultStatus, self.FingerA_PositionReqEcho = self.Byte_status(self.add_leading_zeros(bin(readData[1])))
            self.FingerA_Position, self.FingerA_Current = self.Byte_status(self.add_leading_zeros(bin(readData[2])))
            self.FingerB_PositionReqEcho, self.FingerB_Position = self.Byte_status(self.add_leading_zeros(bin(readData[3])))
            self.FingerB_Current, self.FingerC_PositionReqEcho = self.Byte_status(self.add_leading_zeros(bin(readData[4])))
            self.FingerC_Position, self.FingerC_Current = self.Byte_status(self.add_leading_zeros(bin(readData[5])))
            self.Scissor_PositionReqEcho, self.Scissor_Position = self.Byte_status(self.add_leading_zeros(bin(readData[6])))
            self.Scissor_Current, _ = self.Byte_status(self.add_leading_zeros(bin(readData[7])))

        except Exception as e:
            print(f"[EXCEPTION] Failed to update gripper status: {e}")

        
    @staticmethod
    def add_leading_zeros(bin_num: str, total_length: int = 16) -> str:
        """Ensure binary number string has correct number of digits."""
        bin_str = str(bin_num)[2:]  # remove '0b' prefix
        return bin_str.zfill(total_length)

    @staticmethod
    def Byte_status(variable: int) -> str:
        """Split and parse byte status."""
        B1 = int(variable[0:8], 2)
        B2 = int(variable[8:16], 2)
        return B1, B2

    @staticmethod
    def stat(variable: int) -> str:
        """Split and parse status."""
        # Define gripper modes
        modes = ["Basic Mode", "Pinch Mode", "Wide Mode", "Scissor Mode"]

        # Split and parse status bits
        gSTA = int(variable[0:2], 2)
        gIMC = int(variable[2:4], 2)
        gGTO = int(variable[4], 2)
        gMOD = int(variable[5:7], 2)
        gMOD = modes[gMOD]  # Translate mode code to string
        gACT = int(variable[7], 2)
        gDTS = int(variable[8:10], 2)
        gDTC = int(variable[10:12], 2)
        gDTB = int(variable[12:14], 2)
        gDTA = int(variable[14:16], 2)
        
        return gSTA, gIMC, gGTO, gMOD, gACT, gDTS, gDTC, gDTB, gDTA
    
    def activate(self) -> None:
        """Activate the gripper."""
        # response = self.client.write_multiple_registers(0, [self._action_req_variable(rACT=1), 0, 0])
        self.client.write_multiple_registers(
                0,
                [self._action_req_variable(rACT=1, rGTO=1, rMOD=0, rICF=0),
                self._position_req_variable(0),
                self._write_req_variable(0, 0)]
            )
        print("[INFO] Gripper activate")
        time.sleep(1)

    def command_gripper(
        self, 
        rPRA: List[int] = [1, 1, 1], 
        rSP: List[int] = [250, 250, 250], 
        rFR: List[int] = [250, 250, 250], 
        rMOD: str = "Basic", 
        rICF: bool = False
    ) -> None:
        """Send a command to the gripper."""
        # self.status()
        modes = {
            "Basic": 0, 
            "Pinch": 1, 
            "Wide": 2, 
            "Scissor": 3
        }
        rMOD = modes[rMOD]
        if rICF:
            for var in [rPRA, rSP, rFR]:
                if isinstance(var, int):
                    self.close()
                    raise ValueError("Input variables must be 3d vectors when using Individual Control Flag.")
            response = self.client.write_multiple_registers(
                0,
                [self._action_req_variable(rACT=1, rGTO=1, rMOD=rMOD, rICF=1),
                self._position_req_variable(rPRA[0]),
                self._write_req_variable(rSP[0], rFR[0]),
                self._write_req_variable(rPRA[1],rSP[1]),
                self._write_req_variable(rFR[1],rPRA[2]),
                self._write_req_variable(rSP[2], rFR[2]),]
            )
        else:
            for var in [rPRA, rSP, rFR]:
                if isinstance(var, list):
                    warnings.warn("only first value of 3d vector will be used when not using Individual Control Flag.") 
            response = self.client.write_multiple_registers(
                0,
                [self._action_req_variable(rACT=1, rGTO=1, rMOD=rMOD, rICF=0),
                self._position_req_variable(rPRA[0]),
                self._write_req_variable(rSP[0], rFR[0])]
            )
        
    def status(self) -> None:
        readData = self.client.read_input_registers(0,8)
        if readData:
            self.gSTA, self.gIMC, self.gGTO, self.gMOD, self.gACT, self.gDTS, self.gDTC, self.gDTB, self.gDTA = self.stat(self.add_leading_zeros(bin(readData[0])))
            self.FaultStatus, self.FingerA_PositionReqEcho = self.Byte_status(self.add_leading_zeros(bin(readData[1])))
            self.FingerA_Position, self.FingerA_Current = self.Byte_status(self.add_leading_zeros(bin(readData[2])))
            self.FingerB_PositionReqEcho, self.FingerB_Position = self.Byte_status(self.add_leading_zeros(bin(readData[3])))
            self.FingerB_Current, self.FingerC_PositionReqEcho = self.Byte_status(self.add_leading_zeros(bin(readData[4])))
            self.FingerC_Position, self.FingerC_Current = self.Byte_status(self.add_leading_zeros(bin(readData[5])))
            self.Scissor_PositionReqEcho, self.Scissor_Position = self.Byte_status(self.add_leading_zeros(bin(readData[6])))
            self.Scissor_Current, RES = self.Byte_status(self.add_leading_zeros(bin(readData[7])))
        else:
            print("Error reading data in status.")

    def _action_req_variable(self, rARD: int = 0, rATR: int = 0, rGTO: int = 0, rACT: int = 0, rMOD:int = 0, rICS:int = 0, rICF:int = 0 ) -> str:
        """Build action request variable."""
        # Check if the input variables are either 0 or 1
        for var in [rARD, rATR, rGTO, rACT]:
            if var not in [0, 1]:
                raise ValueError("Input variables must be either 0 or 1.")
        rMOD = bin(rMOD).replace("0b", "").zfill(2)
        # Construct the string variable
        string_variable = f"0b00{rARD}{rATR}{rGTO}{rMOD}{rACT}0000{rICS}{rICF}00" 
        
        return int(string_variable,2)
    
    def _position_req_variable(self, rPR: int = 0) -> str:
        """Build position request variable."""
        # Check if the input variables are between 0 or 255
        for var in [rPR]:
            if var not in range(0,256):
                raise ValueError("Input variables must be between 0 and 255.")
        rPR = format(rPR, '08b')

        # Construct the string variable
        string_variable = f"0b00000000{rPR}"
        
        return int(string_variable,2) 
    
    def _write_req_variable(self, X: int = 0, Y: int = 0) -> str:
        """Build write request variable."""
        # Check if the input variables are between 0 to 255
        for var in [X, Y]:
            if var not in range(0,256):
                raise ValueError("Input variables must be between 0 and 255.")
        X = format(X, '08b')
        Y = format(Y, '08b')
        # Construct the string variable
        string_variable = f"0b{X}{Y}"
        
        return int(string_variable,2)
    
    def close(self) -> None:
        """Stop the update thread and close the Modbus client."""
        self._running = False
        self._thread.join()
        self.client.close()
        print("Connection closed.")


def main():
    # Initialize the GripperController object with the IP address of the server
    gripper = Robotiq3FGripper("192.168.1.11")

    # Activate the gripper
    gripper.activate()

    # Set up the initial parameters for the gripper command
    target_position = [10, 10, 10]  # Desired finger positions
    speed = [250, 205, 250]  # Speed of the movement
    force = [250, 250, 250]  # Force applied by the fingers

    # Select gripper mode - options: "Basic", "Wide", "Pinch", "Scissor"
    mode = "Pinch"
    
    # Define if individual finger control is required
    individual_control = True

    # If individual control is not required, replicate the first element across the list
    target_position = [target_position[0]] * 3 if not individual_control else target_position
        
    # Send command to the gripper with the specified parameters
    gripper.command_gripper(rPRA=target_position, rSP=speed, rFR=force, rMOD=mode, rICF=individual_control)

    # Wait for the gripper to reach the target positions
    while ([gripper.FingerA_Position, gripper.FingerB_Position, gripper.FingerC_Position] != target_position):
        time.sleep(0.1)

    # Output the current finger positions
    print(f"FingerA_Position: {gripper.FingerA_Position} FingerB_Position: {gripper.FingerB_Position} FingerC_Position: {gripper.FingerC_Position}")

    time.sleep(2)

    target_position = [75, 75, 75]  # Desired finger positions
    speed = [250, 205, 250]  # Speed of the movement
    force = [250, 250, 250]  # Force applied by the fingers

    # Select gripper mode - options: "Basic", "Wide", "Pinch", "Scissor"
    mode = "Pinch"
    
    # Define if individual finger control is required
    individual_control = True

    # If individual control is not required, replicate the first element across the list
    target_position = [target_position[0]] * 3 if not individual_control else target_position
        
    # Send command to the gripper with the specified parameters
    gripper.command_gripper(rPRA=target_position, rSP=speed, rFR=force, rMOD=mode, rICF=individual_control)

    # Wait for the gripper to reach the target positions
    while ([gripper.FingerA_Position, gripper.FingerB_Position, gripper.FingerC_Position] != target_position):
        time.sleep(0.1)

    # Output the current finger positions
    print(f"FingerA_Position: {gripper.FingerA_Position} FingerB_Position: {gripper.FingerB_Position} FingerC_Position: {gripper.FingerC_Position}")

    time.sleep(2)

    target_position = [100, 100, 100]  # Desired finger positions
    speed = [250, 205, 250]  # Speed of the movement
    force = [250, 250, 250]  # Force applied by the fingers

    # Select gripper mode - options: "Basic", "Wide", "Pinch", "Scissor"
    mode = "Pinch"
    
    # Define if individual finger control is required
    individual_control = True

    # If individual control is not required, replicate the first element across the list
    target_position = [target_position[0]] * 3 if not individual_control else target_position
        
    # Send command to the gripper with the specified parameters
    gripper.command_gripper(rPRA=target_position, rSP=speed, rFR=force, rMOD=mode, rICF=individual_control)

    # Wait for the gripper to reach the target positions
    while ([gripper.FingerA_Position, gripper.FingerB_Position, gripper.FingerC_Position] != target_position):
        time.sleep(0.1)

    # Output the current finger positions
    print(f"FingerA_Position: {gripper.FingerA_Position} FingerB_Position: {gripper.FingerB_Position} FingerC_Position: {gripper.FingerC_Position}")

    # Close the connection when finished
    # gripper.close()

if __name__ == "__main__":
    main()
