from controller import Robot
import os
import sys

libraryPath = os.path.join(os.environ.get("WEBOTS_HOME"), 'projects', 'robots', 'robotis', 'darwin-op', 'libraries',
                           'python37')
libraryPath = libraryPath.replace('/', os.sep)
sys.path.append(libraryPath)
from managers import RobotisOp2GaitManager, RobotisOp2MotionManager


class Walk():
    def __init__(self):
        self.robot = Robot()  # 初始化Robot类以控制机器人
        self.mTimeStep = int(self.robot.getBasicTimeStep())  # 获取当前每一个仿真步所仿真时间mTimeStep
        self.HeadLed = self.robot.getLED('HeadLed')  # 获取头部LED灯
        self.EyeLed = self.robot.getLED('EyeLed')  # 获取眼部LED灯
        self.HeadLed.set(0xff0000)  # 点亮头部LED灯并设置一个颜色
        self.EyeLed.set(0xa0a0ff)  # 点亮眼部LED灯并设置一个颜色
        self.mAccelerometer = self.robot.getAccelerometer('Accelerometer')  # 获取加速度传感器
        self.mAccelerometer.enable(self.mTimeStep)  # 激活传感器，并以mTimeStep为周期更新数值
        self.fup = 0
        self.fdown = 0   # 定义两个类变量，用于之后判断机器人是否摔倒

        self.mGyro = self.robot.getGyro('Gyro')  # 获取陀螺仪
        self.mGyro.enable(self.mTimeStep)  # 激活陀螺仪，并以mTimeStep为周期更新数值

        self.positionSensors = []  # 初始化关节角度传感器
        self.positionSensorNames = ('ShoulderR', 'ShoulderL', 'ArmUpperR', 'ArmUpperL',
                                    'ArmLowerR', 'ArmLowerL', 'PelvYR', 'PelvYL',
                                    'PelvR', 'PelvL', 'LegUpperR', 'LegUpperL',
                                    'LegLowerR', 'LegLowerL', 'AnkleR', 'AnkleL',
                                    'FootR', 'FootL', 'Neck', 'Head')  # 初始化各传感器名
        self.acc_win1 = []  # 加速度传感器检测跌倒模块y轴方向滑窗
        self.acc_win2 = []  # 加速度传感器检测跌倒模块z轴方向滑窗
        self.acc_last_avg1 = 0  # 加速度传感器检测跌倒模块上一个y轴滑窗平均
        self.acc_last_avg2 = 0  # 加速度传感器检测跌倒模块上一个z轴滑窗平均
        self.winlen = 20  # 加速度传感器检测跌倒模块滑窗长度
        # 获取各传感器并激活，以mTimeStep为周期更新数值
        for i in range(0, len(self.positionSensorNames)):
            self.positionSensors.append(self.robot.getPositionSensor(self.positionSensorNames[i] + 'S'))
            self.positionSensors[i].enable(self.mTimeStep)

        self.mKeyboard = self.robot.getKeyboard()  # 初始化键盘读入类
        self.mKeyboard.enable(self.mTimeStep)  # 以mTimeStep为周期从键盘读取

        self.mMotionManager = RobotisOp2MotionManager(self.robot)  # 初始化机器人动作组控制器
        self.mGaitManager = RobotisOp2GaitManager(self.robot, "config.ini")  # 初始化机器人步态控制器

    def myStep(self):
        ret = self.robot.step(self.mTimeStep)
        if ret == -1:
            exit(0)

    def wait(self, ms):
        startTime = self.robot.getTime()
        s = ms / 1000.0
        while s + startTime >= self.robot.getTime():
            self.myStep()

    def run(self):
        print("-------Walk example of ROBOTIS OP2-------")
        print("This example illustrates Gait Manager")
        print("Press the space bar to start/stop walking")
        print("Use the arrow keys to move the robot while walking")
        self.myStep()  # 仿真一个步长，刷新传感器读数

        self.mMotionManager.playPage(9)  # 执行动作组9号动作，初始化站立姿势，准备行走
        self.wait(200)  # 等待200ms

        self.isWalking = False  # 初始时机器人未进入行走状态

        while True:
            self.checkIfFallen()
            self.mGaitManager.setXAmplitude(0.0)  # 前进为0
            self.mGaitManager.setAAmplitude(0.0)  # 转体为0
            key = 0  # 初始键盘读入默认为0
            key = self.mKeyboard.getKey()  # 从键盘读取输入
            if key == 32:  # 如果读取到空格，则改变行走状态
                if (self.isWalking):  # 如果当前机器人正在走路，则使机器人停止
                    self.mGaitManager.stop()
                    self.isWalking = False
                    self.wait(200)
                else:  # 如果机器人当前停止，则开始走路
                    self.mGaitManager.start()
                    self.isWalking = True
                    self.wait(200)
            elif key == 315:  # 如果读取到‘↑’，则前进
                self.mGaitManager.setXAmplitude(1.0)
            elif key == 317:  # 如果读取到‘↓’，则后退
                self.mGaitManager.setXAmplitude(-1.0)
            elif key == 316:  # 如果读取到‘←’，则左转
                self.mGaitManager.setAAmplitude(-0.5)
            elif key == 314:  # 如果读取到‘→’，则右转
                self.mGaitManager.setAAmplitude(0.5)
            self.mGaitManager.step(self.mTimeStep)  # 步态生成器生成一个步长的动作
            self.myStep()  # 仿真一个步长

    def checkIfFallen(self):
        '''
        对加速度计在y轴、z轴方向的读数做滑动平均来检测机器人是否摔倒
        检测到摔倒之后，判断是背摔还是正摔，令机器人站起来
        如果判断错误，机器人未能站起，通过一起执行正摔背摔站起的程序站起

        Args:
            self (Walk): Walk类，代表机器人自身
        '''
        acc = self.mAccelerometer.getValues()  # 通过加速度传感器获取三轴的对应值
        if (len(self.acc_win1) == 0) or (len(self.acc_win2) == 0):  # 在预备姿态下滑窗初始化，不进行其他操作
            self.acc_win1.append(acc[1])
            self.acc_win2.append(acc[2])
            return
        avg1 = sum(self.acc_win1) / len(self.acc_win1)  # 计算y轴方向滑窗平均值
        avg2 = sum(self.acc_win2) / len(self.acc_win2)  # 计算z轴方向滑窗平均值
        if self.acc_last_avg1 != 0 and self.acc_last_avg1 != 0:  # 若上次滑窗均值不为0，则与此时滑窗均值比较
            if abs(avg1 - self.acc_last_avg1) >= 90:  # 若y轴方向上次滑窗均值和本次相差太大，说明起身失败，需强制起身
                self.mMotionManager.playPage(11)  # 倒地起身动作1
                self.mMotionManager.playPage(10)  # 倒地起身动作0
                self.mMotionManager.playPage(9)  # 恢复准备行走姿势
                self.acc_last_avg1 = 0  # 强制起身之后，上次滑窗均值清零防止循环
                self.acc_last_avg2 = 0  # 强制起身之后，上次滑窗均值清零防止循环
                return
        if (acc[1] - avg1) >= 200:  # y轴方向滑窗均值远小于此时y轴方向加速度，机器人后倒
            self.mMotionManager.playPage(11)  # 倒地起身动作1
            self.mMotionManager.playPage(9)  # 恢复准备行走姿势
            self.acc_win1 = []  # 起身之后窗清空
            self.acc_win2 = []
            self.acc_last_avg1 = avg1  # 起身之后记录下上次滑窗均值
            self.acc_last_avg2 = avg2
        elif (acc[1] - avg1) <= -200:  # y轴方向滑窗均值远小于此时y轴方向加速度，机器人前倾
            self.mMotionManager.playPage(10)  # 倒地起身动作0
            self.mMotionManager.playPage(9)  # 恢复准备行走姿势
            self.acc_win1 = []  # 起身之后窗清空
            self.acc_win2 = []
            self.acc_last_avg1 = avg1  # 起身之后记录下上次滑窗均值
            self.acc_last_avg2 = avg2
        else:  # 正常未跌倒情况
            if len(self.acc_win1) <= self.winlen:  # 滑窗长度未达到要求，塞入数据
                self.acc_win1.append(acc[1])
            else:  # 滑窗长度达到要求，更新滑窗
                del self.acc_win1[0]
                self.acc_win1.append(acc[1])
            if len(self.acc_win2) <= self.winlen:  # 滑窗长度未达到要求，塞入数据
                self.acc_win2.append(acc[2])
            else:  # 滑窗长度达到要求，更新滑窗
                del self.acc_win2[0]
                self.acc_win2.append(acc[2])


if __name__ == '__main__':
    walk = Walk()  # 初始化Walk类
    walk.run()  # 运行控制器
