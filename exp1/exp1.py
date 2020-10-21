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
        acc = self.mAccelerometer.getValues()  # 通过加速度传感器获取三轴的对应值
        '''
        add your codes here
        '''
        #self.mMotionManager.playPage(10) # 倒地起身动作0
        #self.mMotionManager.playPage(11) # 倒地起身动作1
        #self.mMotionManager.playPage(9) # 恢复准备行走姿势

if __name__ == '__main__':
    walk = Walk()  # 初始化Walk类
    walk.run()  # 运行控制器
