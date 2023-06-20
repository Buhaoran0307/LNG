# coding=utf-8
import os

import pandas as pd
import math
import gc

from Utils import save_pickle_data


def distance(a, b):
    pi = 3.14
    cos = math.cos(a[1] * pi / 180) * math.cos(b[1] * pi / 180) * math.cos((a[0] - b[0]) * pi / 180)
    sin = math.sin(a[1] * pi / 180) * math.sin(b[1] * pi / 180)
    dis = 6371000 * math.acos(cos + sin - 1e-12)
    return dis


def convert_raw_data(source_path, target_path):
    print("[log] 正在转换为csv文件....")
    raw_data = open(source_path)
    with open(target_path, "w") as t:
        t.write('mmsi,timestamp,status,speed,longitude,latitude,waterline' + "\n")
        i = 0
        for line in raw_data:
            print("第", i, "个数据点: ", line)
            t.write(line.replace(" ", ","))
            i += 1
    print("[log] 转换完成 ！")


def gain_low_speed_data(temp_path):
    csv = pd.read_csv(temp_path)
    data = {
        'longitude': [],
        'latitude': [],
        'behavior': []
    }
    attribute_list = ['longitude', 'latitude', 'draft']
    avg_pt = [0, 0]
    tem_pt = [csv[attribute][0] for attribute in attribute_list]
    count = 0
    draft_valid = []
    for i in range(len(csv)):
        pt = [csv[attribute][i] for attribute in attribute_list]
        # 汇聚间距小于5km的点
        if distance(pt, tem_pt) < 5000:
            avg_pt[0] += pt[0]
            avg_pt[1] += pt[1]
            if pt[2] != 0:
                draft_valid.append(pt[2])
            tem_pt = pt
            count += 1
        else:
            if count != 0:
                data['longitude'].append(avg_pt[0] / count)
                data['latitude'].append(avg_pt[1] / count)
                if len(draft_valid) > 0:
                    different = draft_valid[-1] - draft_valid[0]
                    if different > 8:
                        data['behavior'].append(1)
                    elif different < -8:
                        data['behavior'].append(-1)
                    else:
                        data['behavior'].append(0)
                else:
                    data['behavior'].append(0)
            count = 0
            avg_pt = [0, 0]
            tem_pt = pt
            draft_valid = []
    return data


def gain_preprocess_data(source_path):
    temp_path = 'data/temp'
    # 进行数据预处理
    # 读取原始数据
    raw_data = pd.read_csv(source_path)

    # 提取低速和停泊的数据点
    status = raw_data['status'][:]
    moored_status = (status == 1) | (status == 5) | (status == 15)
    moored_data = raw_data[moored_status]
    moored_data_frame = pd.DataFrame(moored_data)
    moored_data_frame.to_csv(temp_path + '/raw_moored_status.csv', index=False, encoding="utf-8")
    save_pickle_data(moored_data, 'data/roaming/moored_data.pkl')
    del raw_data
    del moored_data_frame
    gc.collect()

    # 聚合后的低速和停泊数据点
    selected_moored_data = gain_low_speed_data(temp_path + '/wrong_status.csv')
    selected_moored_data_frame = pd.DataFrame(selected_moored_data)
    selected_moored_data_frame.to_csv('data/selected_moored_data.csv', index=False, encoding="utf-8")
    save_pickle_data(selected_moored_data, 'data/roaming/selected_moored_data.pkl')
    del raw_data
    del selected_moored_data
    gc.collect()
