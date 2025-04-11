'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-06
FilePath: /Automated Valet Parking/config/read_config.py
Description: read config and return a dict

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


import yaml
import os


# -> dict 返回值的类型提示：告诉读者/编辑器这个函数返回一个 dict 类型（字典）
# 不写 -> dict 不影响函数运行
def read_config(config_name) -> dict:
    name = config_name + '.yaml'
    # 获取当前 Python 文件所在目录的绝对路径
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, name)
    f = open(yamlPath, 'r', encoding='utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config
