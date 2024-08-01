import numpy as np

from trajectory_io import  *
camera_poses_gpu = read_trajectory("test_segm_gpu.log")  # 读取轨迹文件
camera_poses = read_trajectory("test_segm.log")  # 读取轨迹文件
pose_diff_array = []
# 检查两个轨迹的长度是否相同
if len(camera_poses_gpu) != len(camera_poses):
    print("The trajectories have different lengths.")
else:
    # 打印每个对应位姿的差异
    for i in range(len(camera_poses)):
        pose_diff = camera_poses_gpu[i].pose - camera_poses[i].pose
        pose_diff_array.append(pose_diff)
pose_diff_array = np.array(pose_diff_array)