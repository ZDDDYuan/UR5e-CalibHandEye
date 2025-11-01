import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import ace_tools_open as tools

# Load calibration data
calib_data_dir = "calib_handeye_data/20250709_202641"
T_g2b_all = np.load(f"{calib_data_dir}/gripper2base.npy")
T_t2c_all = np.load(f"{calib_data_dir}/target2cam.npy")
T_cam2gripper = np.loadtxt(f"{calib_data_dir}/camera2gripper.txt")

def compute_pose_error(T1, T2):
    """
    Calculate the pose error between two tranform matrix
    """
    dT = np.linalg.inv(T1) @ T2
    trans_error = np.linalg.norm(dT[:3, 3])
    rotvec = R.from_matrix(dT[:3, :3]).as_rotvec()
    rot_error_deg = np.linalg.norm(np.rad2deg(rotvec))
    return trans_error, rot_error_deg

trans_errors = []
rot_errors = []

for i in range(len(T_g2b_all)):
    T_g2b = T_g2b_all[i]
    T_t2c = T_t2c_all[i]

    # Calculated T_target2base
    T_t2b_pred = T_g2b @ T_cam2gripper @ T_t2c

    # Use first calculated T_target2base as ground truth
    if i == 0:
        T_ref = T_t2b_pred.copy()

    # Calculate errors
    t_err, r_err = compute_pose_error(T_ref, T_t2b_pred)
    trans_errors.append(t_err)
    rot_errors.append(r_err)

# Calculate RMSE in translation and rotation
rmse_t = np.sqrt(np.mean(np.square(trans_errors)))
rmse_r = np.sqrt(np.mean(np.square(rot_errors)))

print(f"[RESULT] Translation RMSE: {rmse_t*1000:.2f} mm")
print(f"[RESULT] ROtation RMSE: {rmse_r:.2f} deg")

# Visualization
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(trans_errors, marker='o')
plt.title("Translation Error per Sample (m)")
plt.xlabel("Sample Index")
plt.ylabel("Translation Error")

plt.subplot(1, 2, 2)
plt.plot(rot_errors, marker='x', color='orange')
plt.title("Rotation Error per Sample (deg)")
plt.xlabel("Sample Index")
plt.ylabel("Rotation Error")

plt.tight_layout()
plt.show()

# Lists to store per-axis errors
dxs, dys, dzs = [], [], []
rxs, rys, rzs = [], [], []

# Compute per-axis translation and rotation errors
for i in range(len(T_g2b_all)):
    Tg2b = T_g2b_all[i]
    Tt2c = T_t2c_all[i]
    # Predicted target to base
    T_pred = Tg2b @ T_cam2gripper @ Tt2c
    if i == 0:
        T_ref = T_pred.copy()
    dT = np.linalg.inv(T_ref) @ T_pred
    
    # Translation error per axis
    dxs.append(dT[0, 3])
    dys.append(dT[1, 3])
    dzs.append(dT[2, 3])
    
    # Rotation error per axis (Euler angles)
    rotvec = R.from_matrix(dT[:3, :3]).as_rotvec()
    euler_deg = np.rad2deg(R.from_rotvec(rotvec).as_euler('xyz'))
    rxs.append(euler_deg[0])
    rys.append(euler_deg[1])
    rzs.append(euler_deg[2])

# Create DataFrame for display
df = pd.DataFrame({
    'dx (m)': dxs,
    'dy (m)': dys,
    'dz (m)': dzs,
    'rx (deg)': rxs,
    'ry (deg)': rys,
    'rz (deg)': rzs
})

# Display table
tools.display_dataframe_to_user("Per-axis Pose Errors", df)

# Plot per-axis translation errors
plt.figure(figsize=(12, 4))
for idx, axis in enumerate(['dx (m)', 'dy (m)', 'dz (m)']):
    plt.plot(df[axis], marker='o', label=axis)
plt.title("Translation Error per Axis")
plt.xlabel("Sample Index")
plt.ylabel("Error (m)")
plt.legend()
plt.show()

# Plot per-axis rotation errors
plt.figure(figsize=(12, 4))
for idx, axis in enumerate(['rx (deg)', 'ry (deg)', 'rz (deg)']):
    plt.plot(df[axis], marker='x', label=axis)
plt.title("Rotation Error per Axis")
plt.xlabel("Sample Index")
plt.ylabel("Error (deg)")
plt.legend()
plt.show()