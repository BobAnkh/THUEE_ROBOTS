#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-12-09 19:33:06
# @LastEditTime : 2020-12-13 10:02:50
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


def djkstra(graph, start, end):
    path_set = set()    # 已求的路径集合
    priority_dic = {}
    for k in graph.keys():
        priority_dic[k] = [9999, False, ""] # 权重表构建为一个3维数组，分别是：最小路径代价，是否计算过最小边，最小路径
    priority_dic[start][0] = 0

    # 判断权重表中所有路点是否添加完毕
    def isSelectAll():
        ret = True
        for val in priority_dic.values():
            if not val[1]:
                ret = False
                break
        return ret

    while not isSelectAll():
        find_point = start
        find_path = start
        min_distance = 9999
        for path in path_set:
            end_point = path[-2:]
            path_distance = priority_dic[end_point][0]
            if path_distance < min_distance and not priority_dic[end_point][1]:
                find_point = end_point
                find_path = path
                min_distance = path_distance
        find_distance = priority_dic[find_point][0]
        neighbors = graph[find_point]
        for k in neighbors.keys():
            p = find_path + "-" + k
            weight = find_distance + neighbors[k]
            path_set.add(p)
            if weight < priority_dic[k][0]:
                priority_dic[k][0] = weight
                priority_dic[k][2] = p
        priority_dic[find_point][1] = True

    return priority_dic[end]


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
            if value[0] in range(1, 10):
                if value[1] in range(1, 10):
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


def path_finding_d(start_point, end_point, barrier):
    if start_point == end_point:
        return [start_point]
    obstacle_points = [[2, 2], [2, 5], [2, 8], [5, 2], [5, 5], [5, 8], [8, 2],
                       [8, 5], [8, 8]]
    obstacle_points.extend(obs_position(barrier))
    ################################################################
    graph = {}
    for i in range(1, 10):
        for j in range(1, 10):
            if [i, j] in obstacle_points:
                continue
            neighbor_weigh = {}
            neighbors = [[i - 1, j], [i, j + 1], [i + 1, j], [i, j - 1]]
            for neighbor in neighbors:
                if neighbor[0] < 1 or neighbor[0] > 9 or neighbor[1] < 1 or neighbor[1] > 9:
                    continue
                elif neighbor in obstacle_points:
                    continue
                else:
                    neighbor_weigh[str(neighbor[0])+str(neighbor[1])] = 1
            graph[str(i)+str(j)] = neighbor_weigh
    result = djkstra(graph, str(start_point[0])+str(start_point[1]), str(end_point[0])+str(end_point[1]))
    tmp = result[2].split('-')
    path_vertices = []
    for elem in tmp:
        path_vertices.append([int(elem[0]), int(elem[1])])
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
