import open3d as o3d
import numpy as np

# 假设这是您的转换矩阵Tensor对象
transformation_tensor = o3d.core.Tensor([[0.9999866121559432, 0.002351604949298301, 0.004609280100534836, -0.0031274554085975697],
                                         [-0.0023595897805917415, 0.9999957237436029, 0.0017276662224256865, -0.0010650859740782266],
                                         [-0.004605197601632161, -0.0017385191029190223, 0.9999878847798016, 0.0004209729704048829],
                                         [0, 0, 0, 1]], dtype=o3d.core.Dtype.Float64)

# 打印原始Tensor
print("Original Tensor:")
print(transformation_tensor)

# 将Open3D Tensor转换为NumPy数组
transformation_numpy = np.asarray(transformation_tensor)

# 打印转换后的NumPy数组
print("\nConverted NumPy Array:")
print(transformation_numpy)
