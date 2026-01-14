"""
In this file, you should implement your own path planning class or function.
Within your implementation, you may call `env.is_collide()` and `env.is_outside()`
to verify whether candidate path points collide with obstacles or exceed the
environment boundaries.

You are required to write the path planning algorithm by yourself. Copying or calling 
any existing path planning algorithms from others is strictly
prohibited. Please avoid using external packages beyond common Python libraries
such as `numpy`, `math`, or `scipy`. If you must use additional packages, you
must clearly explain the reason in your report.
"""
            

import heapq
import math


class Node:
    """A* 算法的节点类"""

    def __init__(self, x, y, z, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.z = z
        self.g = g  # 起点到当前点的距离
        self.h = h  # 当前点到终点的启发式距离
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        # 优先队列通过比较 f 值来排序
        return self.f < other.f

    def get_pos(self):
        return (self.x, self.y, self.z)


# 启发式距离计算
def get_dist(pos1, pos2):
    """计算两点间的欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0]) ** 2 +
                     (pos1[1] - pos2[1]) ** 2 +
                     (pos1[2] - pos2[2]) ** 2)


def a_star_search(env, start_pos=(1, 2, 0), goal_pos=(18, 18, 3), step_size=0.5):
    """
    参数:
    env: FlightEnvironment 对象
    start_pos: 起始坐标 tuple (x, y, z)
    goal_pos: 目标坐标 tuple (x, y, z)
    step_size: 搜索步长（精度），越小越精确但越慢
    """

    # 1. 初始化
    start_node = Node(start_pos[0], start_pos[1], start_pos[2], g=0, h=get_dist(start_pos, goal_pos))
    goal_node = Node(goal_pos[0], goal_pos[1], goal_pos[2])

    open_list = []  # 优先队列
    heapq.heappush(open_list, start_node)

    # 使用字典记录已访问节点，key为坐标字符串或元组
    # 这里为了简单，我们用某种方式量化坐标作为key
    closed_set = set()

    # 定义移动方向：x, y, z 的变化量 (-1, 0, 1) 的所有组合
    # 生成 26 个方向的 neighbor (3x3x3 - 1)
    movements = []
    for dx in [-step_size, 0, step_size]:
        for dy in [-step_size, 0, step_size]:
            for dz in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                movements.append((dx, dy, dz))

    while open_list:
        # 2. 取出 f 值最小的节点
        current_node = heapq.heappop(open_list)

        # 检查是否接近目标点 (小于步长即可认为到达)
        if get_dist(current_node.get_pos(), goal_pos) < step_size:
            goal_node.parent = current_node
            goal_node.g = current_node.g + get_dist(current_node.get_pos(), goal_pos)
            return reconstruct_path(goal_node)

        # 离散化坐标用于去重 (例如保留1位小数)
        node_key = (round(current_node.x, 1), round(current_node.y, 1), round(current_node.z, 1))

        if node_key in closed_set:
            continue
        closed_set.add(node_key)

        # 3. 扩展邻居节点
        for dx, dy, dz in movements:
            new_x = current_node.x + dx
            new_y = current_node.y + dy
            new_z = current_node.z + dz
            new_pos = (new_x, new_y, new_z)

            # --- 关键：调用环境接口进行检查 ---
            # 检查1: 是否出界
            if env.is_outside(new_pos):
                continue

            # 检查2: 是否碰撞
            if env.is_collide(new_pos):
                continue

            # 4. 创建新节点并加入 Open List
            g_cost = current_node.g + math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            h_cost = get_dist(new_pos, goal_pos)

            new_node = Node(new_x, new_y, new_z, g=g_cost, h=h_cost, parent=current_node)

            heapq.heappush(open_list, new_node)

    print("未找到路径！")
    return []


def reconstruct_path(node):
    """从终点回溯到起点生成路径"""
    path = []
    current = node
    while current:
        path.append([current.x, current.y, current.z])
        current = current.parent
    return path[::-1]  # 反转列表
