import svm_hog  # 模式匹配
from PIL import Image as PIL_Image


def obj_recog():
    orig_img_path = '/home/xilinx/jupyter_notebooks/exp2/data/test/robot_pic.jpg'
    # jupyter调用摄像头的代码
#   !fswebcam  --no-banner --no-overlay --save {orig_img_path} -d /dev/video0 2> /dev/null
    pic_addr = 'data/test'  # 照片存储位置
    kind = svm_hog.SVM_HOG_TEST(pic_addr, loglevel='INFO')
    return kind


if __name__ == '__main__':
    kind = obj_recog()
    print(kind)
