"""
In this file, you should implement your trajectory generation class or function.
Your method must generate a smooth 3-axis trajectory (x(t), y(t), z(t)) that 
passes through all the previously computed path points. A positional deviation 
up to 0.1 m from each path point is allowed.

You should output the generated trajectory and visualize it. The figure must
contain three subplots showing x, y, and z, respectively, with time t (in seconds)
as the horizontal axis. Additionally, you must plot the original discrete path 
points on the same figure for comparison.

You are expected to write the implementation yourself. Do NOT copy or reuse any 
existing trajectory generation code from others. Avoid using external packages 
beyond general scientific libraries such as numpy, math, or scipy. If you decide 
to use additional packages, you must clearly explain the reason in your report.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class TrajectoryGenerator:
    def __init__(self, path, average_speed=2.0):
        """
        初始化轨迹生成器，使用三次样条插值
        :param path: A*生成的路径点列表
        :param average_speed: 假设的平均飞行速度 (m/s)
        """
        raw_path = np.array(path)

        # --- [注！] 数据预处理：去除连续重复的点 ---
        if len(raw_path) > 0:
            # 计算相邻点之间的距离
            # np.diff 计算 path[i+1] - path[i]
            diffs = np.diff(raw_path, axis=0)
            # 计算欧几里得距离
            dists = np.linalg.norm(diffs, axis=1)

            # 生成掩码：保留第一个点(True)，以及所有距离大于 0.001 的后续点
            # 1e-3 是一个很小的阈值，用于过滤掉重复点
            mask = np.concatenate(([True], dists > 1e-3))

            self.path = raw_path[mask]
        else:
            self.path = raw_path

        # 安全检查：如果去重后点太少，无法生成样条
        if len(self.path) < 2:
            print("警告：路径点过少，无法生成轨迹，保留原始点。")
            self.time_knots = np.array([0])
            # 防止后续报错，这里做个简单处理
            return

        self.avg_speed = average_speed
        self.time_knots = self._calculate_time_knots()

        # 构建三次样条插值函数
        # bc_type='clamped' 强制起止速度为0
        self.cs_x = CubicSpline(self.time_knots, self.path[:, 0], bc_type='clamped')
        self.cs_y = CubicSpline(self.time_knots, self.path[:, 1], bc_type='clamped')
        self.cs_z = CubicSpline(self.time_knots, self.path[:, 2], bc_type='clamped')

    def _calculate_time_knots(self):
        """根据路径点之间的欧氏距离计算时间戳"""
        diffs = np.diff(self.path, axis=0)
        distances = np.sqrt((diffs ** 2).sum(axis=1))

        # 防止除零错误（虽然预处理已经过滤了，但加一层保险）
        distances = np.maximum(distances, 1e-6)

        times = distances / self.avg_speed
        time_knots = np.hstack(([0], np.cumsum(times)))
        return time_knots

    def solve(self, dt=0.1):
        if len(self.path) < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])

        total_time = self.time_knots[-1]
        t_eval = np.arange(0, total_time, dt)

        # 如果时间恰好整除，arange 可能不包含最后一个点，手动补齐以确保完整
        if t_eval[-1] < total_time:
            t_eval = np.append(t_eval, total_time)

        x_traj = self.cs_x(t_eval)
        y_traj = self.cs_y(t_eval)
        z_traj = self.cs_z(t_eval)

        return t_eval, x_traj, y_traj, z_traj

    def visualize(self, dt=0.1):
        if len(self.path) < 2:
            print("无法可视化：路径点不足")
            return

        t_smooth, x_smooth, y_smooth, z_smooth = self.solve(dt)

        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        fig.suptitle('Generated Trajectory', fontsize=16)

        # Plot X
        axs[0].plot(t_smooth, x_smooth, 'r-', label='Trajectory X(t)')
        axs[0].scatter(self.time_knots, self.path[:, 0], color='black', marker='o', label='Path Points')
        axs[0].set_ylabel('X (m)')
        axs[0].legend()
        axs[0].grid(True)

        # Plot Y
        axs[1].plot(t_smooth, y_smooth, 'g-', label='Trajectory Y(t)')
        axs[1].scatter(self.time_knots, self.path[:, 1], color='black', marker='o', label='Path Points')
        axs[1].set_ylabel('Y (m)')
        axs[1].legend()
        axs[1].grid(True)

        # Plot Z
        axs[2].plot(t_smooth, z_smooth, 'b-', label='Trajectory Z(t)')
        axs[2].scatter(self.time_knots, self.path[:, 2], color='black', marker='o', label='Path Points')
        axs[2].set_ylabel('Z (m)')
        axs[2].set_xlabel('Time (s)')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
