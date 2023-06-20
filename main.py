# coding=utf-8
import math
import time

from sklearn.cluster import KMeans, DBSCAN

from data_loader import data_loader
from pre_process import gain_preprocess_data, convert_raw_data
from result_analysis import clustering_analysis

# 读取数据并且转换格式
convert_raw_data("data/lng.csv", "data/real_lng.csv")

# 进行数据预处理
selected_moored_data = gain_preprocess_data("data/real_lng.csv")
loader = data_loader("data/selected_moored_data.csv")
data, coord = loader.load()

#使用DBSCAN算法聚合数据
start = time.time()
clustering_dbscan = DBSCAN(eps=0.05, min_samples=3).fit(coord)
end = time.time()
dic = clustering_analysis(clustering_dbscan, data, "DBSCAN", filename="lng_results_list(DBSCAN).json")
print("DBSCAN")
print("总数: ", len(clustering_dbscan.labels_))
print("聚类中心点数: ", len(set(clustering_dbscan.labels_)) - (1 if -1 in clustering_dbscan.labels_ else 0))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")

#使用K-means算法聚合数据
start = time.time()
clustering_kmeans = KMeans(n_clusters=400, init='random').fit(coord)
end = time.time()
dic = clustering_analysis(clustering_kmeans, data, "K-means", filename="lng_results_list(K-means).json")
print("K-means")
print("总数: ", len(clustering_kmeans.labels_))
print("聚类中心点数: ", len(clustering_kmeans.cluster_centers_))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")

#使用K-means++算法聚合数据
start = time.time()
clustering_kmeanspp = KMeans(n_clusters=600, init='k-means++').fit(coord)
end = time.time()
dic = clustering_analysis(clustering_kmeanspp, data, "K-means++", filename="lng_results_list(K-means++).json")
print("K-means++")
print("总数: ", len(clustering_kmeanspp.labels_))
print("聚类中心点数: ", len(clustering_kmeanspp.cluster_centers_))
print()
print("其中，")
print("\tLNG出口点有: ", len(dic["export"]), "个")
print("\tLNG出口点有: ", len(dic["import"]), "个")
print("\t停泊点有: ", len(dic["mooring"]), "个")
print("耗时：", end - start, "秒")