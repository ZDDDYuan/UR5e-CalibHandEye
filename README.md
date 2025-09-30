# Hand-Eye Calibration for UR5e Robot

## Hardware

- **Robot**: UR5e Robot
- **RGB-D Camera**: CameraRealSense D435i
- **Gripper**: Robotiq-3F Gripper

## Coordinate Systems

* **Base**: Robot base coordinate system
* **Gripper**: Robot end-effector coordinate system
* **Target**: Calibration target coordinate system
* **Camera**: Camera coordinate system

## Eye-in-Hand Calibration
### Theory

In an *eye-in-hand* calibration setup, the goal is to estimate the transformation matrix from the camera to the robot end-effector, denoted as:

$$
^{Gripper}T_{Camera}
$$

During the calibration process, the pose of the calibration target with respect to the robot base is assumed to be **constant**, i.e.,

$$
^{Base}T_{Target}
$$

remains unchanged.

By moving the robot end-effector to multiple poses, we can construct the following relationship:

$$
^{Base}T_{End_1} \cdot ^{End_1}T_{Camera_1} \cdot ^{Camera_1}T_{Target} = ^{Base}T_{End_2} \cdot ^{End_2}T_{Camera_2} \cdot ^{Camera_2}T_{Target}
$$

The problem of estimating the transformation from the camera coordinate frame to the end-effector frame becomes equivalent to solving the hand-eye calibration problem in the form of:

$$
AX = XB
$$

Where:

* $A = ^{Base}T_{End_2}^{-1} \cdot ^{Base}T_{End_1}$
* $B = ^{Camera_2}T_{Target} \cdot ^{Camera_1}T_{Target}^{-1}$

Substituting into the equation:

$$
^{Base}T_{End_2}^{-1} \cdot ^{Base}T_{End_1} \cdot ^{End_1}T_{Camera_1} = ^{End_2}T_{Camera_2} \cdot ^{Camera_2}T_{Target} \cdot \left(^{Camera_1}T_{Target}\right)^{-1}
$$

The goal is to solve for:

$$
^{Gripper}T_{Camera}
$$

which defines the transformation from the camera frame to the robot end-effector.

### Application

1. Run the eye-in-hand calibration script:

```python
python calibration.py
```
the transformation matrix $^{Gripper}T_{Camera}$ would be saved in `calib_handeye_data` directory.

2. Evaluate the calibration precision by running
```python
python utils/evaluation.py
```

3. Run a demo for pick-and-place task for testing calibration precision in actual grasping
```python
python click_to_pick_and_place.py
```
