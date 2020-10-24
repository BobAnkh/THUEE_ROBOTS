# THUEE_ROBOTS

THUEE Course Intelligent Robots Design and Implementation

## 环境说明

统一采用Python 3.7.6版本

## 说明

第1次小实验的相关材料和代码存放在`exp1`目录下

第2次小实验的相关材料和代码存放在`exp2`目录下，运行代码需要将包含有所有数据集的`data`文件夹置于`exp2`文件夹内，并在`exp2`文件夹下运行

目前总体两类，每类各30个样本图片，可以抽取少数若干张作为测试集，其余留作训练集

`data`文件夹下的文件结构大致如此:

```test
.
|-- test
|   |-- duck1.jpg
|   `-- pikachu1.jpg
`-- train
    |-- duck
    |   |-- 0.jpg
    |   |-- 1.jpg
    |   `-- ...(省略若干张图片)
    `-- pikachu
        |-- 0.jpg
        |-- 1.jpg
        `-- ...(省略若干张图片)

```
