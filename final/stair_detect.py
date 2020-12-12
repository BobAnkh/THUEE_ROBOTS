import utils
import os
import svm_hog


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


def stair_detect(face_direction, target, current_location):
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
    target_block = direction_judge(temp['stair.jpg'], step, target)
    # current_location = utils.walk_step(current_point, next_point)
    return current_direction, target_block
