#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-12-09 19:17:14
# @LastEditTime : 2020-12-13 09:01:36
# @Description  : 
from SixClass import six_class
import Path
import svm_hog
import stair_detect
import os
import canny_pokemon_pos
import cv2


def des_position(number):
    des_pos = []
    destination_dict = {
        1: [[2, 2]],
        2: [[2, 5]],
        3: [[2, 8]],
        4: [[5, 2]],
        5: [[5, 5]],
        6: [[5, 8]],
        7: [[8, 2]],
        8: [[8, 5]],
        9: [[8, 8]]
    }
    for n in number:
        des_pos = des_pos + destination_dict[n]
    return des_pos


def des_to_target(destination):
    dtt_dict = {
        '[2, 2]': [2, 1],
        '[2, 5]': [2, 6],
        '[2, 8]': [2, 9],
        '[5, 2]': [5, 1],
        '[5, 5]': [5, 6],
        '[5, 8]': [4, 8],
        '[8, 2]': [8, 1],
        '[8, 5]': [8, 6],
        '[8, 8]': [8, 9]
    }
    return dtt_dict[str(destination)]


animal_label = [
        'kabi', 'jienigui', 'miaowazhongzi', 'pikachu', 'yibu',
        'xiaohuolong'
    ]

def main():
    # 六分类识别
    orig_img_path = '/home/xilinx/jupyter_notebooks/final/six.jpg'
    os.system(f'fswebcam  --no-banner --no-overlay --save {orig_img_path} -d /dev/video0 2> /dev/null')
    target_animal = six_class(orig_img_path)
    svm_model = '/home/xilinx/jupyter_notebooks/final/svm_model/' + target_animal + '.model'
    print(target_animal)
    # test data
    start_point = [1, 2]
    targets = des_position([2, 5])
    end_point = [3, 9]
    barrier = [1, 5]
    # 挑选合适的起始点位
    target0_point = des_to_target(targets[0])
    target1_point = des_to_target(targets[1])
    path_1 = Path.path_finding_d_d(start_point, target0_point, barrier)
    path_2 = Path.path_finding_d(start_point, target1_point, barrier)
    path_3 = Path.path_finding_d(target0_point, end_point, barrier)
    path_4 = Path.path_finding_d(target1_point, end_point, barrier)
    if (len(path_1) + len(path_3)) > 12 and (len(path_1) + len(path_3)) < 19 and (len(path_2) + len(path_4)) > 12 and (len(path_2) + len(path_4)) < 19:
        if len(path_1) > len(path_2):
            targets = [targets[1], targets[0]]
    else:
        if (len(path_2) + len(path_4)) > 12 and (len(path_2) + len(path_4)) < 19:
            targets = [targets[1], targets[0]]
    # 计算起始朝向
    # face_direction: 0表示向上，1表示向右，2表示向下，3表示向左
    if start_point[0] == 1:
        face_direction = 2
    elif start_point[0] == 9:
        face_direction = 0
    elif start_point[1] == 1:
        face_direction = 1
    elif start_point[1] == 9:
        face_direction = 3
    else:
        raise SystemExit('Wrong Start Point!')
    # 确定台阶侧面以开始搜寻朝向并抵达正面
    detect_point = des_to_target(targets[0])
    path_vertices = Path.path_finding_d(start_point, detect_point, barrier)
    current_direction, current_location = Path.walk_to_target(path_vertices, face_direction)
    current_direction, target_block = stair_detect.stair_detect(current_direction, targets[0], current_location, barrier)
    path_vertices = Path.path_finding_d(current_location, target_block, barrier)
    current_direction, current_location = Path.walk_to_target(path_vertices, current_direction)
    # 朝向
    step = [
            targets[0][0] - current_location[0], targets[0][1] - current_location[1]
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
    angle = need_direction - current_direction
    if angle > 2:
        angle = angle - 4
    if angle < -2:
        angle = angle + 4
    utils.turn(angle)
    current_direction = need_direction
    # TODO: 走的步数不知道
    utils.walk()
    # utils.fall()h
    # 识别小动物
    orig_img_path = '/home/xilinx/jupyter_notebooks/final/animal.jpg'
    os.system(f'fswebcam  --no-banner --no-overlay --save {orig_img_path} -d /dev/video0 2> /dev/null')
    DATA_TEST = '/home/xilinx/jupyter_notebooks/final/animal.jpg'
    temp = svm_hog.SVM_HOG_TEST(DATA_TEST, model_place=svm_model)
    if temp['animal.jpg'] == 'duck':
        detect_point = des_to_target(targets[1])
        path_vertices = Path.path_finding_d(current_location, detect_point, barrier)
        current_direction, current_location = Path.walk_to_target(path_vertices, face_direction)
        current_direction, target_block = stair_detect.stair_detect(current_direction, targets[1], current_location, barrier)
        path_vertices = Path.path_finding_d(current_location, target_block, barrier)
        current_direction, current_location = Path.walk_to_target(path_vertices, current_direction)
        # 朝向
        step = [
                    targets[1][0] - current_location[0], targets[1][1] - current_location[1]
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
        angle = need_direction - current_direction
        if angle > 2:
            angle = angle - 4
        if angle < -2:
            angle = angle + 4
        utils.turn(angle)
        current_direction = need_direction
        # 上台阶
        utils.go_upstairs()
        for turn in range(3):
            img_path = '/home/xilinx/jupyter_notebooks/final/animal_upstair.jpg'
            os.system(f'fswebcam  --no-banner --no-overlay --save {img_path} -d /dev/video0 2> /dev/null')
            img_temp = cv2.imread(img_path)
            cur_pos = canny_pokemon_pos.canny_pokemon_pos(img_temp)
            if cur_pos == 2:
                break
            elif cur_pos < 2:
                utils.moveleft()
            else:
                utils.moveright()
        # 抓取
        utils.down_stairs()
        path_vertices = Path.path_finding_d(current_location, end_point, barrier)
        current_direction, current_location = Path.walk_to_target(path_vertices, current_direction)
        utils.solute()
    else:
        # 上台阶
        utils.go_upstairs()
            for turn in range(3):
        img_path = '/home/xilinx/jupyter_notebooks/final/animal_upstair.jpg'
        os.system(f'fswebcam  --no-banner --no-overlay --save {img_path} -d /dev/video0 2> /dev/null')
        img_temp = cv2.imread(img_path)
        cur_pos = canny_pokemon_pos.canny_pokemon_pos(img_temp)
        if cur_pos == 2:
            break
        elif cur_pos < 2:
            utils.moveleft()
        else:
            utils.moveright()
        utils.down_stairs()
        path_vertices = Path.path_finding_d(current_location, end_point, barrier)
        current_direction, current_location = Path.walk_to_target(path_vertices, current_direction)
        utils.solute()


if __name__ == '__main__':
    main()
