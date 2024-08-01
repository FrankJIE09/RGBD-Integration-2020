import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def read_bag_file(bag_file):
    # 创建一个RGBD视频读取器来读取 .bag 文件
    bag_reader = o3d.t.io.RSBagReader()
    bag_reader.open(bag_file)

    # 获取相机内参
    metadata = bag_reader.metadata
    intrinsics = metadata.intrinsics
    width = intrinsics.width
    height = intrinsics.height
    fx = intrinsics.intrinsic_matrix[0,0]
    fy =  intrinsics.intrinsic_matrix[1,1]
    cx =  intrinsics.intrinsic_matrix[0,2]
    cy = intrinsics.intrinsic_matrix[1,2]

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    frames = []
    while not bag_reader.is_eof():
        frame = bag_reader.next_frame()
        if frame is None:
            break
        frames.append(frame)

    return frames, camera_intrinsics


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    radius_normal = voxel_size * 2
    source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def main():
    bag_file = 'realsense2.bag'  # 修改为你的 .bag 文件路径
    voxel_size = 0.01  # 体素大小
    trunc = np.inf  # 深度截断

    frames, camera_intrinsics = read_bag_file(bag_file)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    transform_0 = np.identity(4)
    traj = [transform_0]

    for i in range(len(frames) - 1):
        color_image_np = np.asarray(frames[i].color)
        depth_image_np = np.asarray(frames[i].depth)

        # 将 NumPy 数组转换为 Open3D 图像
        color_image_o3d = o3d.geometry.Image(color_image_np)
        depth_image_o3d = o3d.geometry.Image(depth_image_np)

        # 创建 RGBDImage
        source_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image_o3d,
            depth_image_o3d,
            depth_scale=1000.0,  # 根据你的深度数据单位进行调整
            depth_trunc=3.0,  # 根据需要设置最大深度截断值
            convert_rgb_to_intensity=False  # 通常为 False，除非需要将 RGB 转换为灰度强度
        )
        color_image_np = np.asarray(frames[i+1].color)
        depth_image_np = np.asarray(frames[i+1].depth)

        # 将 NumPy 数组转换为 Open3D 图像
        color_image_o3d = o3d.geometry.Image(color_image_np)
        depth_image_o3d = o3d.geometry.Image(depth_image_np)

        # 创建 RGBDImage
        target_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image_o3d,
            depth_image_o3d,
            depth_scale=1000.0,  # 根据你的深度数据单位进行调整
            depth_trunc=3.0,  # 根据需要设置最大深度截断值
            convert_rgb_to_intensity=False  # 通常为 False，除非需要将 RGB 转换为灰度强度
        )



        source = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd, camera_intrinsics)
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

        target = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd, camera_intrinsics)
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        result_icp = refine_registration(source, target, voxel_size, result_ransac)

        transform = result_icp.transformation
        transform = np.dot(transform, transform_0)
        transform_0 = transform
        traj.append(transform)

    for i, frame in enumerate(frames):
        volume.integrate(frame, camera_intrinsics, traj[i])

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh('integrated_mesh.ply', mesh)


if __name__ == '__main__':
    main()
