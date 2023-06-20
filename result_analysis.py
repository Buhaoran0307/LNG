# coding=utf-8
import json
import matplotlib.pyplot as plt
import numpy as np


def save_json(data, json_path):
    with open(json_path, 'a') as f:
        json.dump(data, f)


def plot_data_points(data, figure_size, title):
    plt.figure(figsize=figure_size, dpi=100)
    plt.title(title)
    plt.legend(data, (u'Import', u'Export', u'Mooring'))
    plt.savefig("data/" + title + ".png")


def find_type(cluster):
    load_rate = (cluster[:, 2] > 0).sum() / cluster.shape[0]
    discharge_rate = (cluster[:, 2] < 0).sum() / cluster.shape[0]
    center = np.average(cluster, axis=0)

    if discharge_rate >= 0.05 and load_rate >= 0.05:
        port_class = 'port'
    elif discharge_rate >= 0.05:
        port_class = 'import'
    elif load_rate >= 0.05:
        port_class = 'export'
    else:
        port_class = 'none'
    return port_class, center


def clustering_analysis(clustering, data, algorithm_name, filename):
    save_path = 'results'
    # 获取聚类标签
    labels = clustering.labels_
    # 计算非离群点的个数
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # 保存一条结果
    one_result = {
        # {"code":1,"latitude":xxx, "longitude":xxx, "isLNG":true, "IN":false}
    }
    # 备份进出站港口
    another_result = {}
    # 对聚类节点进行分类
    dic = {
        "import": [],
        "export": [],
        "none": []
    }

    j = 0
    backup = []
    with open(save_path + '/' + filename, 'w') as f:
        f.write('{\n')
    for i in range(n_clusters_):
        one_cluster = data[labels == i]
        if one_cluster.size > 0:
            port_class, center = find_type(one_cluster)
            one_result["code"] = i + 1
            one_result["latitude"] = format(center[1], '.6f')
            one_result["longitude"] = format(center[0], '.6f')

            if port_class == 'port':
                dic["import"].append(center)
                one_result["isLNG"] = True
                one_result["IN"] = True
                IN_port = plt.scatter(center[0], center[1], c='r', s=5)

                # 备份该港口
                j += 1
                another_result["code"] = n_clusters_ + j
                another_result["latitude"] = format(center[1], '.6f')
                another_result["longitude"] = format(center[0], '.6f')
                dic["export"].append(center)
                another_result["isLNG"] = True
                another_result["IN"] = False
                backup.append(another_result)
                OUT_port = plt.scatter(center[0], center[1], c='b', s=5)

            elif port_class == 'import':
                dic["import"].append(center)
                one_result["isLNG"] = True
                one_result["IN"] = True
                IN_port = plt.scatter(center[0], center[1], c='r', s=5)
            elif port_class == 'export':
                dic["export"].append(center)
                one_result["isLNG"] = True
                one_result["IN"] = False
                OUT_port = plt.scatter(center[0], center[1], c='b', s=5)
            else:
                dic["mooring"].append(center)
                one_result["isLNG"] = False
                one_result["IN"] = None
                mooring = plt.scatter(center[0], center[1], c='g', s=5)
            with open(save_path + '/' + filename, 'a') as f:
                f.write('\t')
            save_json(one_result, save_path + '/' + filename)
            with open(save_path + '/' + filename, 'a') as f:
                f.write('\n')
    # 绘制可视化聚点
    plot_data_points((IN_port, OUT_port, mooring), (9, 7), algorithm_name)
    # 保存备份结果
    for result in backup:
        with open(save_path + '/' + filename, 'a') as f:
            f.write('\t')
        save_json(result, save_path + '/' + filename)
        with open(save_path + '/' + filename, 'a') as f:
            f.write('\n')
    with open(save_path + '/' + filename, 'a') as f:
        f.write('}')
