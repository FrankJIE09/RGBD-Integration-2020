import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

# 假设机械臂的动力学可以用线性状态空间模型表示
# A: 系统矩阵
# B: 输入矩阵
# Q: 状态权重矩阵
# R: 输入权重矩阵

A = np.array([[0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0]])
B = np.array([[0, 0],
              [0, 0],
              [1, 0],
              [0, 1]])
Q = np.eye(A.shape[1])       # 状态权重矩阵，这里使用单位矩阵作为示例
R = np.eye(B.shape[1])       # 输入权重矩阵，这里使用单位矩阵作为示例

# 计算LQR增益
P = la.solve_continuous_are(A, B, Q, R)
K = la.inv(R) @ la.solve_continuous_lyapunov(A.T, B.T,)

# 定义目标状态和初始状态
x_target = np.array([...])  # 目标状态向量
x_initial = np.array([...])  # 初始状态向量

# 使用LQR控制器进行轨迹规划
def lqr_controller(x):
    return -K @ x

# 定义轨迹规划的优化问题
def objective(x):
    # 计算轨迹的代价
    return 0.5 * x.T @ Q @ x + 0.5 * (lqr_controller(x).T @ R @ lqr_controller(x))

# 初始轨迹点
x_trajectory = [x_initial]

# 进行优化，这里使用scipy的minimize函数
result = minimize(objective, x_initial, method='SLSQP')

# 检查优化是否成功
if result.success:
    x_trajectory.append(result.x)
    # 可以继续添加更多的轨迹点，直到达到目标状态
    # ...
else:
    print("Optimization failed:", result.message)

# 打印最终的轨迹
for x in x_trajectory:
    print(x)