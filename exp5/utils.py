#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-11-25 21:44:22
# @LastEditTime : 2020-11-25 23:46:26
# @Description  :
import cmd_control as cctl
import send

def get_location():
    '''
    获取当前位置

    Returns:
        list: 位置坐标
    '''
    cmd = "robot1"  # 假定机器人是robot1
    Location_string = send.send_common(cmd)
    Location_num = int(Location_string)
    
    location = [Location_num//10%10, Location_num//1%10]
    return location


def walk_step(current_point, next_point):
    '''
    向前走1格

    Args:
        current_point (list): 当前坐标点
        next_point (list): 下一个坐标点

    Returns:
        int: 1表示成功，0表示失败
    '''
    
    location_now = current_point
    L = 1
    while 1:
        if location_now == current_point:
            cctl.run_action('SlowForward')
            location_now = get_location()
        else:
            for i in range(0, L):
                cctl.run_action('SlowForward')
            break
    return 1


def turn(angle):
    '''
    转向

    Args:
        angle (int): 正表示顺时针，负表示逆时针，数值*90表示转动角度。如2表示顺时针转180度，-1表示逆时针转90度

    Returns:
        int: 1表示成功，0表示失败
    '''
    if angle != 0:
        cctl.run_action('SlowForward')
        time = [3, 3]
        if angle < 0:
            for i in range(0, -angle*time[0]):
                cctl.run_action('LeftTurn')
        else:
            for i in range(0, angle*time[1]):
                cctl.run_action('RightTurn')
    return 1
