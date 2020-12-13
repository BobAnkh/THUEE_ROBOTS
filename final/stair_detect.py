import utils
import os
import svm_hog
import Path


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


def direction_judge(result, step, target):
    if step == [1, 0]:
        if result == 'back':
            return [target[0] + 1, target[1]]
        elif result == 'front':
            return [target[0] - 1, target[1]]
        elif result == 'left_high':
            return [target[0], target[1] - 1]
        elif result == 'right_high':
            return [target[0], target[1] + 1]
    elif step == [-1, 0]:
        if result == 'back':
            return [target[0] - 1, target[1]]
        elif result == 'front':
            return [target[0] + 1, target[1]]
        elif result == 'left_high':
            return [target[0], target[1] + 1]
        elif result == 'right_high':
            return [target[0], target[1] - 1]
    elif step == [0, 1]:
        if result == 'back':
            return [target[0], target[1] + 1]
        elif result == 'front':
            return [target[0], target[1] - 1]
        elif result == 'left_high':
            return [target[0] + 1, target[1]]
        elif result == 'right_high':
            return [target[0] - 1, target[1]]
    elif step == [0, -1]:
        if result == 'back':
            return [target[0], target[1] - 1]
        elif result == 'front':
            return [target[0], target[1] + 1]
        elif result == 'left_high':
            return [target[0] - 1, target[1]]
        elif result == 'right_high':
            return [target[0] + 1, target[1]]


def stair_detect(face_direction, target, current_location, barrier):
    obstacle_points = [[2, 2], [2, 5], [2, 8], [5, 2], [5, 5], [5, 8], [8, 2],
                       [8, 5], [8, 8]]
    obstacle_points.extend(obs_position(barrier))
    current_direction = face_direction
    step = [
        target[0] - current_location[0], target[1] - current_location[1]
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
        raise SystemExit('Wrong Direction!')
    angle = need_direction - current_direction
    if angle > 2:
        angle = angle - 4
    if angle < -2:
        angle = angle + 4
    utils.turn(angle)
    current_direction = need_direction
    orig_img_path = '/home/xilinx/jupyter_notebooks/final/stair/stair.jpg'
    os.system(f'fswebcam  --no-banner --no-overlay --save {orig_img_path} -d /dev/video0 2> /dev/null')
    temp = svm_hog.SVM_HOG_TEST('/home/xilinx/jupyter_notebooks/final/stair/', model_place='/home/xilinx/jupyter_notebooks/final/stair.model')
    if temp['stair.jpg'] == 'back' or temp['stair.jpg'] == 'front':
        if current_location[0] == target[0]:
            next_point = [target[0]-1, target[1]]
            if next_point in obstacle_points:
                next_point = [target[0]+1, target[1]]
        else:
            next_point = [target[0], target[1]-1]
            if next_point in obstacle_points:
                next_point = [target[0], target[1]+1]
        path_vertices = Path.path_finding_d(current_location, next_point, barrier)
        current_direction, current_location = Path.walk_to_target(path_vertices, current_direction)
        os.system(f'fswebcam  --no-banner --no-overlay --save {orig_img_path} -d /dev/video0 2> /dev/null')
        temp = svm_hog.SVM_HOG_TEST('/home/xilinx/jupyter_notebooks/final/stair/', model_place='/home/xilinx/jupyter_notebooks/final/stair.model')
    target_block = direction_judge(temp['stair.jpg'], step, target)
    # current_location = utils.walk_step(current_point, next_point)
    return current_direction, target_block
