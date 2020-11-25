#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Author       : BobAnkh
# @Github       : https://github.com/BobAnkh
# @Date         : 2020-11-25 21:44:22
# @LastEditTime : 2020-11-25 23:46:26
# @Description  :


def get_location():
    '''
    获取当前位置

    Returns:
        list: 位置坐标
    '''
    location = [4, 1]
    print("My location:", location)
    return location


def walk_step(current_point, next_point):
    '''
    向前走1步

    Args:
        current_point (list): 当前坐标点
        next_point (list): 下一个坐标点

    Returns:
        int: 1表示成功，0表示失败
    '''

    return 1


def turn(angle):
    '''
    转向

    Args:
        angle (int): 正表示顺时针，负表示逆时针，数值*90表示转动角度。如2表示顺时针转180度，-1表示逆时针转90度

    Returns:
        int: 1表示成功，0表示失败
    '''

    return 1