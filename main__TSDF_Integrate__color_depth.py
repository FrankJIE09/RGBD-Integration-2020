"""
Created on Fri Mar 13 12:28:52 2020
@author: Margarita Chizh

Based on Open3D Tutorial: http://www.open3d.org/docs/release/tutorial/Advanced/rgbd_integration.html

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details
"""

import open3d as o3d              # 导入 Open3D 库并简写为 o3d
import numpy as np                # 导入 NumPy 库并简写为 np
import matplotlib.pyplot as plt   # 导入 Matplotlib 的 pyplot 模块并简写为 plt

plt.close('all')                  # 关闭所有打开的图形窗口
pic_num = 1                       # 设置图片编号初始值为 1

#==============================================================================
#==============================================================================

file_name = 'A_hat1'              # 文件名前缀（位于 Test_data 文件夹中）

# 准备帧编号：
num_Frames = 132                  # 帧总数（在 Test_data 文件夹中）
skip_N_frames = 10                # 可以选择集成的帧范围 - 帧数少则运行速度快
frames_nums = np.arange(0, num_Frames + 1, skip_N_frames)  # 生成帧编号数组

#==============================================================================
#==============================================================================

#  == ICP 函数: ==

# 检查: http://www.open3d.org/docs/release/tutorial/Basic/icp_registration.html

def draw_registration_result(source, target, transformation, title='Title'):
    source_temp = source                         # 临时变量保存 source 点云
    target_temp = target                         # 临时变量保存 target 点云
    source_temp.paint_uniform_color([1, 0.706, 0])  # 给 source 点云着色
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # 给 target 点云着色
    source_temp.transform(transformation)        # 对 source 点云应用变换
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name=title)  # 显示点云

def preprocess_point_cloud(pcd, voxel_size):
#    print("\n- - Preprocessing - -\n:: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)  # 体素下采样

    radius_normal = voxel_size * 2
#    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  # 估计法向量

    radius_feature = voxel_size * 5
#    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))  # 计算 FPFH 特征
    return pcd_down, pcd_fpfh  # 返回下采样后的点云和 FPFH 特征

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
#    print("\n- - Global registration - -\n:: RANSAC registration on downsampled point clouds.")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, False, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))  # RANSAC 配准
    return result  # 返回配准结果

def refine_registration(source, target, voxel_size):
    distance_threshold = voxel_size * 0.4
#    print("\n - - Refine - -\n:: Point-to-plane ICP registration is applied on original point")
    radius_normal = voxel_size * 2
    source.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  # 估计源点云法向量
    target.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))  # 估计目标点云法向量
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())  # 点对平面 ICP 配准
    return result  # 返回精细配准结果

#==============================================================================
##=============##           TSDF 参数                  ##=============##
#==============================================================================

voxel_size  = 0.01   # 体素大小 1cm
trunc       = np.inf  # 最大深度限制，测试深度帧在主体分割期间已被截断
#==============================================================================
##=============##                 相机内参                  ##=============##
#==============================================================================

# 对于 Intel RealSense，可使用命令获取内参：color_frame.profile.as_video_stream_profile().intrinsics
# 内参：
width   = 1280
height  = 720
fx      = 920.003
fy      = 919.888
cx      = 640.124
cy      = 358.495
#scale   = 0.0010000000474974513 # 将[mm]转换为[米]

# 创建所需格式的内参矩阵：
cameraIntrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# print(cameraIntrinsics.intrinsic_matrix)

#==============================================================================
##=============##               /里程计/ ICP               ##=============##
#==============================================================================

# 代替使用不准确的里程计，我们使用 ICP 查找一个帧到下一个帧的变换

num_of_poses = len(frames_nums)  # 获取帧编号长度

#  == 创建一个轨迹 .log 文件: ==
from trajectory_io import *  # 导入自定义轨迹 IO 模块
metadata    = [0, 0, 0]  # 元数据
traj        = []  # 初始化轨迹列表
transform_0 = np.identity(4)  # 第一帧作为参考帧，保存单位变换矩阵
traj.append(CameraPose(metadata, transform_0))  # 添加初始相机位姿

for i in range(num_of_poses-1):
    # 源帧：
    color_source = o3d.io.read_image("Test_data/%s_color_frame%s.jpg"%(file_name, frames_nums[i]))  # 读取源彩色图像
    depth_source = o3d.io.read_image("Test_data/%s_depth_frame%s.png"%(file_name, frames_nums[i]))  # 读取源深度图像

    rgbd_source = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_source, depth_source,
            depth_trunc=trunc,  # 深度截断
            convert_rgb_to_intensity=False)  # 创建 RGBD 图像
    source = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_source, cameraIntrinsics)  # 从 RGBD 图像创建源点云
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)  # 预处理源点云

    # 目标帧：
    color_target = o3d.io.read_image("Test_data/%s_color_frame%s.jpg"%(file_name, frames_nums[i+1]))  # 读取目标彩色图像
    depth_target = o3d.io.read_image("Test_data/%s_depth_frame%s.png"%(file_name, frames_nums[i+1]))  # 读取目标深度图像

    rgbd_target = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_target, depth_target,
            depth_trunc=trunc,  # 深度截断
            convert_rgb_to_intensity=False)  # 创建 RGBD 图像

    target = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_target, cameraIntrinsics)  # 从 RGBD 图像创建目标点云
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)  # 预处理目标点云
    #----------------------------------------------------------------------
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh, voxel_size)  # 执行全局配准
    result_icp = refine_registration(source, target, voxel_size)  # 执行精细配准

    #draw_registration_result(source, target, result_icp.transformation, title='ICP result')

    ## 将结果添加到相机位姿：
    transform = result_icp.transformation  # 获取变换矩阵
    transform = np.dot(transform, transform_0)  # 计算累计变换
    transform_0 = transform  # 更新当前变换
    traj.append(CameraPose(metadata, transform))  # 添加当前相机位姿


# == 从 ICP 变换生成 .log 文件: ==
write_trajectory(traj, "test_segm.log")  # 写入轨迹文件
camera_poses = read_trajectory("test_segm.log")  # 读取轨迹文件


#==============================================================================
##=============##         TSDF 体积集成             ##=============##
#==============================================================================

#  == TSDF 体积集成: ==

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length = 0.01,  # 体素长度（米），约 1cm
    sdf_trunc = 0.05,  # sdf 截断距离（米），约几个体素长度
    color_type = o3d.pipelines.integration.TSDFVolumeColorType.RGB8)  # 设置颜色类型为 RGB8

print('\nnumber of camera_poses:', len(camera_poses),'\n')  # 打印相机位姿数量


num_cam_pose = 0  # 初始化相机位姿计数
for i in frames_nums:
    print("Integrate %s-th image into the volume." % i)  # 打印当前集成的帧编号

    color = o3d.io.read_image("Test_data/%s_color_frame%s.jpg" % (file_name, i))  # 读取彩色图像
    depth = o3d.io.read_image("Test_data/%s_depth_frame%s.png" % (file_name, i))  # 读取深度图像

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color, depth,
    depth_trunc=trunc,  # 深度截断
    convert_rgb_to_intensity=False)  # 创建 RGBD 图像

    volume.integrate(rgbd, cameraIntrinsics, camera_poses[num_cam_pose].pose)  # 集成到 TSDF 体积
    num_cam_pose += 1  # 更新相机位姿计数

#==============================================================================
##=============##              三角网格                ##=============##
#==============================================================================

# #  == 从体积中提取三角网格（使用 Marching Cubes 算法）并可视化： ==

mesh = volume.extract_triangle_mesh()  # 提取三角网格
print(mesh.compute_vertex_normals())  # 计算顶点法线

mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 翻转网格

o3d.visualization.draw_geometries([mesh])  # 显示网格


# 保存生成的三角网格，重新加载并可视化：

o3d.io.write_triangle_mesh('Mesh__%s__every_%sth_%sframes.ply' % (file_name, skip_N_frames, len(camera_poses)), mesh)  # 写入三角网格
meshRead = o3d.io.read_triangle_mesh('Mesh__%s__every_%sth_%sframes.ply' % (file_name, skip_N_frames, len(camera_poses)))  # 读取三角网格
o3d.visualization.draw_geometries([meshRead])  # 显示网格
