# coding=utf-8
import pickle

import folium
import numpy as np
import pandas as pd
from folium import plugins


def generate_center_map(centers_frame, path):
    word_map = folium.Map(zoom_start=10, control_scale=True)  # 创建地图
    marker_cluster = plugins.MarkerCluster().add_to(word_map)

    for index, row in centers_frame.iterrows():  # 添加标记点
        folium.Marker(location=[row["latitude"], row["longitude"]]).add_to(marker_cluster)

    word_map.save(path)  # 保存地图


def generate_cluster_map(dbscan, feature_vector, path):
    world_map = folium.Map(zoom_start=10, control_scale=True)  # 创建地图
    for label in set(dbscan.labels_):
        if label == -1:
            # 如果标签为-1，表示噪声点，可以选择是否忽略或单独处理
            continue

        # 获取属于当前标签的数据点的索引
        cluster_indices = np.where(dbscan.labels_ == label)[0]

        # 遍历数据点索引，并添加标记点
        for idx in cluster_indices:
            latitude = feature_vector[idx, 0]  # 根据数据点索引获取纬度
            longitude = feature_vector[idx, 1]  # 根据数据点索引获取经度

            # 创建标记点并添加到地图上
            folium.Marker(location=[latitude, longitude]).add_to(world_map)
    world_map.save(path)  # 保存地图为HTML文件


def save_pickle_data(target, path):
    with open(path, 'wb') as file:  # 保存结果
        pickle.dump(target, file)


def recover_pickle_data(path):
    with open(path, 'rb') as file:  # 打开文件并加载模型
        return pickle.load(file)


def convert_frame(data, column):
    return pd.DataFrame(data, columns=column)  # 转换numpy类型到dataframe


def select_features(data, features):
    feature_vector = data[features]
    save_pickle_data(feature_vector, 'data/roaming/feature_vector.pkl')
    print("[log] 抽取特征值：", features)
    print("[log] 特征数据格式为: " + str(feature_vector.shape))
    return data, feature_vector
