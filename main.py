# coding=utf-8
import math

from Utils import generate_center_map, recover_pickle_data, convert_frame, save_pickle_data, select_features, \
    generate_cluster_map
from cluster import kmeans_clustering, DBSCAN_clustering, kmeans_plus_clustering
from pre_process import gain_preprocess_data

# 读取数据并且转换格式
# convert_raw_data("data/lng2.csv", "data/real_lng.csv")

# 进行数据预处理
data = gain_preprocess_data("data/real_lng.csv")
save_pickle_data(data, 'data/roaming/data.pkl')
print("[log] 原数据格式为: " + str(data.shape))

# 选取特征向量
features = ['longitude', 'latitude']
data, feature_vector = select_features(data,features)

# 计算聚类簇的数量
n_clusters: int = int(math.sqrt(feature_vector.shape[0] / 2))
n_init = 20
print("[log] 聚类数量：", n_clusters, "个")

# K-means 聚类
kmeans = kmeans_clustering(feature_vector, n_clusters, n_init)
# kmeans = recover_pickle_data('models/kmeans_model_1647_20.pkl')
centers = kmeans.cluster_centers_
centers_frame = convert_frame(centers, ['longitude', 'latitude'])
generate_center_map(centers_frame, "maps/kmeans.html")

# K-means++ 聚类
kmeans = kmeans_plus_clustering(feature_vector, n_clusters, n_init)
# kmeans = recover_pickle_data('models/kmeans_model_1647_20.pkl')
centers = kmeans.cluster_centers_
centers_frame = convert_frame(centers, ['longitude', 'latitude'])
generate_center_map(centers_frame, "maps/kmeans_plus.html")

# DBSCAN聚类
dbscan = DBSCAN_clustering(feature_vector, 0.05, 3)
# dbscan = recover_pickle_data('models/dbscan_model_0.05_20.3')
generate_cluster_map(dbscan, feature_vector, "maps/dbscan.html")