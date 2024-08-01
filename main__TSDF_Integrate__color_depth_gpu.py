import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from trajectory_io import *  # 导入自定义轨迹 IO 模块

# 关闭所有 matplotlib 图形窗口
plt.close('all')

# 设置图片编号初始值
pic_num = 1

# 文件名前缀和帧编号
file_name = 'A_hat1'
num_Frames = 132
skip_N_frames = 10
frames_nums = np.arange(0, num_Frames + 1, skip_N_frames)

# 体素大小和最大深度限制
voxel_size = 0.01
trunc = np.inf

# 相机内参设置
width, height = 1280, 720
fx, fy, cx, cy = 920.003, 919.888, 640.124, 358.495
# 创建 Open3D 的 PinholeCameraIntrinsic 对象
camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

# 获取 intrinsic_matrix 并转换为 Tensor
intrinsic_matrix = np.array(camera_intrinsics.intrinsic_matrix)
intrinsic_tensor = o3d.core.Tensor(intrinsic_matrix, dtype=o3d.core.Dtype.Float64, device=o3d.core.Device("CUDA:0"))

# 现在 intrinsic_tensor 在 GPU 上，可以用于 GPU 上的进一步处理
print("Intrinsic Tensor on CUDA:", intrinsic_tensor)
# 设置 GPU 设备

device = o3d.core.Device("CUDA:0")
metadata = [0, 0, 0]  # 元数据

traj = []  # 初始化轨迹列表
transform_0 = np.identity(4)  # 第一帧作为参考帧，保存单位变换矩阵
traj.append(CameraPose(metadata, transform_0))  # 添加初始相机位姿


def preprocess_point_cloud(pcd, voxel_size):
    # 体素下采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        radius=voxel_size * 2, max_nn=30)
    pcd_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        radius=voxel_size * 5, max_nn=100)
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    result = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(
        source_down, target_down, source_fpfh, target_fpfh, False,
        voxel_size * 1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(), 4,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
        device=device)
    return result


def refine_registration(source, target, voxel_size, transformation):
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 0.4, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


# 提取数值部分的函数
def extract_numeric_value(entry):
    try:
        return float(entry)
    except ValueError:
        return entry


def integrate_and_extract_mesh(frames, camera_poses):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01, sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8, device=device)
    for rgbd, pose in zip(frames, camera_poses):
        volume.integrate(rgbd, camera_intrinsics, pose)
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh


camera_poses_cpu = read_trajectory("test_segm.log")  # 读取轨迹文件

# 主处理流程（假设 frames_nums 和其他相关数据已经准备好）
camera_poses = []  # 用于存储每一帧的相机位姿
for i in tqdm(range(len(frames_nums) - 1)):
    # 假设已经加载了一些 RGBD 图像到 GPU
    color_image = o3d.io.read_image("Test_data_backup/%s_color_frame%s.jpg" % (file_name, frames_nums[i]))  # 读取源彩色图像
    depth_image = o3d.io.read_image("Test_data_backup/%s_depth_frame%s.png" % (file_name, frames_nums[i]))  # 读取源深度图像

    # 将图像也转换为 GPU 上的 Tensor
    color_tensor = o3d.t.geometry.Image.from_legacy(color_image, device=o3d.core.Device("CUDA:0"))
    depth_tensor = o3d.t.geometry.Image.from_legacy(depth_image, device=o3d.core.Device("CUDA:0"))

    # 创建 RGBDImage
    rgbd_image = o3d.t.geometry.RGBDImage(
        color_tensor, depth_tensor)

    # 使用 GPU 上的内参矩阵 Tensor 和 RGBD 图像来创建点云
    source = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic_tensor)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)

    color_image = o3d.io.read_image("Test_data_backup/%s_color_frame%s.jpg" % (file_name, frames_nums[i + 1]))  # 读取源彩色图像
    depth_image = o3d.io.read_image("Test_data_backup/%s_depth_frame%s.png" % (file_name, frames_nums[i + 1]))  # 读取源深度图像

    # 将图像也转换为 GPU 上的 Tensor
    color_tensor = o3d.t.geometry.Image.from_legacy(color_image, device=o3d.core.Device("CUDA:0"))
    depth_tensor = o3d.t.geometry.Image.from_legacy(depth_image, device=o3d.core.Device("CUDA:0"))

    # 创建 RGBDImage
    rgbd_image = o3d.t.geometry.RGBDImage(
        color_tensor, depth_tensor)

    # 使用 GPU 上的内参矩阵 Tensor 和 RGBD 图像来创建点云
    target = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, intrinsic_tensor)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    result_ransac = o3d.t.pipelines.registration.icp(source_down, target_down,
                                                     # init_source_to_target=camera_poses_cpu[i+1].pose,
                                                     estimation_method = o3d.t.pipelines.registration.TransformationEstimationPointToPoint(),
                                                     max_correspondence_distance=voxel_size * 1.5,
                                                     voxel_size=voxel_size,
                                                     criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000))


    # result_icp = refine_registration(source, target, voxel_size)  # 执行精细配准

    # draw_registration_result(source, target, result_icp.transformation, title='ICP result')

    ## 将结果添加到相机位姿：
    transform = result_ransac.transformation  # 获取变换矩阵

    transformation_float_values = np.eye(4)
    # 遍历张量的每个元素，并将其转换为 float
    for m in range(transform.shape[0]):
        for n in range(transform.shape[1]):
            transformation_float_values[m][n] = transform[m][n].item()
    transform = np.dot(transformation_float_values, transform_0)  # 计算累计变换
    transform_0 = transform  # 更新当前变换
    traj.append(CameraPose(metadata, transform))  # 添加当前相机位姿

    # camera_poses.append(pcd_down.get_center())  # 这里只是一个示例，实际应用中应该是配准后的位姿

# == 从 ICP 变换生成 .log 文件: ==
write_trajectory(traj, "test_segm_gpu.log")  # 写入轨迹文件
camera_poses = read_trajectory("test_segm_gpu.log")  # 读取轨迹文件

# camera_poses = read_trajectory("test_segm.log")  # 读取轨迹文件

# 进行 TSDF 体积集成并提取网格
volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=0.01,  # 体素长度（米），约 1cm
    sdf_trunc=0.05,  # sdf 截断距离（米），约几个体素长度
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)  # 设置颜色类型为 RGB8

print('\nnumber of camera_poses:', len(camera_poses), '\n')  # 打印相机位姿数量

num_cam_pose = 0  # 初始化相机位姿计数
for i in frames_nums:
    print("Integrate %s-th image into the volume." % i)  # 打印当前集成的帧编号

    color = o3d.io.read_image("Test_data_backup/%s_color_frame%s.jpg" % (file_name, i))  # 读取彩色图像
    depth = o3d.io.read_image("Test_data_backup/%s_depth_frame%s.png" % (file_name, i))  # 读取深度图像

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_trunc=trunc,  # 深度截断
        convert_rgb_to_intensity=False)  # 创建 RGBD 图像
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd,intrinsic=camera_intrinsics
    )

    # 可视化点云
    o3d.visualization.draw_geometries([pcd])
    volume.integrate(rgbd, camera_intrinsics, camera_poses[num_cam_pose].pose)  # 集成到 TSDF 体积
    num_cam_pose += 1  # 更新相机位姿计数

# ==============================================================================
##=============##              三角网格                ##=============##
# ==============================================================================

# #  == 从体积中提取三角网格（使用 Marching Cubes 算法）并可视化： ==

mesh = volume.extract_triangle_mesh()  # 提取三角网格
print(mesh.compute_vertex_normals())  # 计算顶点法线

mesh.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])  # 翻转网格

o3d.visualization.draw_geometries([mesh])  # 显示网格

# 保存生成的三角网格，重新加载并可视化：

o3d.io.write_triangle_mesh('Mesh__%s__every_%sth_%sframes_gpu.ply' % (file_name, skip_N_frames, len(camera_poses)),
                           mesh)  # 写入三角网格
meshRead = o3d.io.read_triangle_mesh(
    'Mesh__%s__every_%sth_%sframes_gpu.ply' % (file_name, skip_N_frames, len(camera_poses)))  # 读取三角网格
o3d.visualization.draw_geometries([meshRead])  # 显示网格
