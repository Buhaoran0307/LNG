# coding=utf-8
import pandas as pd


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


def gain_preprocess_data(source_path):
    raw_data = pd.read_csv(source_path)

    # 数据预处理
    print("[log] 正在进行数据预处理....")
    data = raw_data[(raw_data['waterline'] > 0) & (raw_data['waterline'] <= 300)]
    print("[log] 删除吃水线小于0和大于300的值...done")

    return data
