#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-12-09 19:33:06
# @LastEditTime : 2020-12-09 20:19:53
# @Description  : 

import utils


def heuristic_distace(Neighbour_node, target_node):
    x_dist = abs(Neighbour_node[0] - target_node[0])
    y_dist = abs(Neighbour_node[1] - target_node[1])
    H = x_dist + y_dist
    return H


def obs_position(number):
    obs_pos = []
    obstacle_dict = {
        1: [[1, 5]],
        2: [[5, 9]],
        3: [[9, 5]],
        4: [[5, 1]],
        5: [[3, 5], [4, 5]],
        6: [[5, 6], [5, 7]],
        7: [[6, 5], [7, 5]],
        8: [[5, 3], [5, 4]],
    }
    for n in number:
        obs_pos = obs_pos + obstacle_dict[n]
    return obs_pos


def path_finding(start_point, end_point, barrier):
    # start_point = utils.get_location()
    # end_point = [2, 5]
    if start_point == end_point:
        return [start_point]
    obstacle_points = [[2, 2], [2, 5], [2, 8], [5, 2], [5, 5], [5, 8], [8, 2],
                       [8, 5], [8, 8]]
    obstacle_points.extend(obs_position(barrier))
    current_point = start_point
    path_vertices = [start_point]
    Neighbour_vertices = []

    while current_point != end_point:
        x = current_point[0]
        y = current_point[1]
        F = []  # 节点权值 F = g + h
        Neighbour_vertices = [
            [x - 1, y],
            [x, y - 1],
            [x, y],
            [x, y + 1],
            [x + 1, y],
        ]
        # 遍历相邻坐标
        for value in Neighbour_vertices:
            if value[0] in range(1, 6):
                if value[1] in range(1, 6):
                    if value not in obstacle_points + path_vertices:
                        # 如果满足节点 1, 在地图边界内 2, 不在障碍物点和已经经过的点, 计算权重
                        F.append(heuristic_distace(value, end_point) + 1)
                    else:
                        F.append(10000)
                else:
                    F.append(10000)
            else:
                F.append(10000)
        current_point = Neighbour_vertices[F.index(
            min(total_distance for total_distance in F))]
        path_vertices.append(current_point)
    return path_vertices


def walk_to_target(path_vertices, face_direction):
    if len(path_vertices) == 1:
        return
    start_point = path_vertices[0]

    # # face_direction: 0表示向上，1表示向右，2表示向下，3表示向左
    # if start_point[0] == 1:
    #     face_direction = 2
    # elif start_point[0] == 9:
    #     face_direction = 0
    # elif start_point[1] == 1:
    #     face_direction = 1
    # elif start_point[1] == 9:
    #     face_direction = 3
    # else:
    #     raise SystemExit('Wrong Start Point!')

    for i in range(len(path_vertices) - 1):
        current_point = path_vertices[i]
        next_point = path_vertices[i + 1]
        step = [
            next_point[0] - current_point[0], next_point[1] - current_point[1]
        ]
        if step[0] == 1:
            need_direction = 2
        elif step[0] == -1:
            need_direction = 0
        elif step[1] == 1:
            need_direction = 1
        elif step[1] == -1:
            need_direction = 3
        else:
            raise SystemExit('Wrong Direction!')
        angle = need_direction - face_direction
        if angle > 2:
            angle = angle - 4
        if angle < -2:
            angle = angle + 4
        # 旋转
        utils.turn(angle)
        face_direction = need_direction
        # 前进
        current_location = utils.walk_step(current_point, next_point)
        return face_direction, current_location
