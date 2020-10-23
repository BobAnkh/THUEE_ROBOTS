import svm_hog  # 模式匹配
import numpy as np
# import utils


def obj_recog(robot_pic, model):
    '''识别机器人相机返回的图片中是什么动物

    Args:
        robot_pic (list): 机器人相机返回的图片
        model (string): 训练好的判别器
    '''
    kind = svm_hog.SVM_HOG_TEST(robot_pic, model)
    print(kind)


def main():
    image = np.zeros((160, 120))
    model = ''
    obj_recog(image, model)
