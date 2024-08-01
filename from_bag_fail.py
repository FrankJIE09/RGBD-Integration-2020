import time

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from tqdm import tqdm


def read_bag_file(bag_file):
    # 创建一个RGBD视频读取器来读取 .bag 文件
    bag_reader = o3d.t.io.RSBagReader()
    bag_reader.open(bag_file)

    # 获取相机内参
    metadata = bag_reader.metadata
    intrinsics = metadata.intrinsics
    width = intrinsics.width
    height = intrinsics.height
    fx = intrinsics.intrinsic_matrix[0, 0]
    fy = intrinsics.intrinsic_matrix[1, 1]
    cx = intrinsics.intrinsic_matrix[0, 2]
    cy = intrinsics.intrinsic_matrix[1, 2]

    # 根据读取的内参创建相机内参对象
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    # 读取并存储所有帧
    frames = []
    while not bag_reader.is_eof():
        frame = bag_reader.next_frame()
        if frame is None:
            break
        frames.append(frame)

    # 返回帧列表和相机内参
    return frames, camera_intrinsics


def preprocess_point_cloud(pcd, voxel_size):
    # 对点云进行体素降采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    # 估算点云的法线
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(radius=radius_normal, max_nn=30)
    # 计算点云的FPFH特征
    radius_feature = voxel_size * 5

    pcd_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(
        pcd_down, radius=radius_feature, max_nn=100)
    # 返回降采样的点云和FPFH特征
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    # 设置全局配准的距离阈值
    distance_threshold = voxel_size * 1.5
    # 执行基于特征匹配的全局RANSAC配准
    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        return result
    except Exception as e:
        error_message = str(e)
        print("\n".join(error_message.split("\n")[:10]))  # 打印前10行错误信息
    # 返回配准结果


def refine_registration(source, target, voxel_size, result_ransac):
    # 设置ICP配准的距离阈值
    distance_threshold = voxel_size * 0.4
    # 估算源和目标点云的法线
    radius_normal = voxel_size * 2
    source.estimate_normals(radius=radius_normal, max_nn=30)
    target.estimate_normals(radius=radius_normal, max_nn=30)
    # 执行ICP配准
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # 返回ICP配准结果
    return result


def main():
    bag_file = "realsense3.bag"  # RealSense相机的.bag文件路径

    pipeline = rs.pipeline()
    config = rs.config()

    # 启用读取流
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    voxel_size = 0.01  # 体素大小
    trunc = np.inf  # 深度截断

    # 读取.bag文件中的帧和相机内参
    frames, camera_intrinsics = read_bag_file(bag_file)

    transform_0 = np.identity(4)
    traj = [transform_0]

    # 处理每一对连续帧
    for i in tqdm(range(0, len(frames) - 1, 10)):
        source_rgbd = frames[i]
        target_rgbd = frames[i + 1]
        source_rgbd = source_rgbd.cuda()
        target_rgbd = target_rgbd.cuda()
        # 从RGBD图像创建点云
        intrinsic_tensor = o3d.core.Tensor(camera_intrinsics.intrinsic_matrix, dtype=o3d.core.Dtype.Float32)
        source = o3d.t.geometry.PointCloud.create_from_rgbd_image(source_rgbd, intrinsic_tensor, )

        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

        target = o3d.t.geometry.PointCloud.create_from_rgbd_image(target_rgbd, intrinsic_tensor)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

        # 执行全局配准和ICP精细配准
        result_ransac = o3d.t.pipelines.registration.icp(source_down, target_down,
                                                         # init_source_to_target=camera_poses_cpu[i+1].pose,
                                                         estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
                                                         max_correspondence_distance=voxel_size * 1.5,
                                                         voxel_size=voxel_size,
                                                         criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(
                                                             max_iteration=1000))

        # 更新变换矩阵并存储轨迹
        transform = result_ransac.transformation  # 获取变换矩阵
        transform = np.dot(transform, transform_0)
        transform_0 = transform
        traj.append(transform)
    num_cam_pose = 0  # 初始化相机位姿计数

    bag_file = "realsense3.bag"  # Replace with your RealSense .bag file path
    #
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the streams you need to read
    config.enable_device_from_file(bag_file, repeat_playback=False)
    profile = pipeline.start(config)

    # 初始化可扩展的TSDF体积
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,  # 体素长度（米），约 1cm
        sdf_trunc=0.05,  # sdf 截断距离（米），约几个体素长度
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)  # 设置颜色类型为 RGB8
    loop_i = 0
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()
        if not frames:
            break
        if loop_i > 10:
            loop_i =0
            time.sleep(0.01)
            continue
        if loop_i > 0:
            time.sleep(0.01)
            continue
        loop_i =loop_i+1
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth = np.asanyarray(depth_frame.get_data())
        color = np.asanyarray(color_frame.get_data())

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth),
            depth_trunc=trunc,  # 深度截断
            convert_rgb_to_intensity=False)  # 创建 RGBD 图像

        volume.integrate(rgbd, camera_intrinsics, traj[num_cam_pose])  # 集成到 TSDF 体积
        num_cam_pose += 1  # 更新相机位姿计数

    # 提取和显示网格
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh('integrated_mesh.ply', mesh)


if __name__ == '__main__':
    main()
