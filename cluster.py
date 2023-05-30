# coding=utf-8
import time

from sklearn.cluster import KMeans, DBSCAN

from Utils import save_pickle_data


def kmeans_clustering(feature_vector, n_clusters, n_init):
    kmeans = KMeans(n_clusters=n_clusters, init='random', verbose=2, n_init=n_init)
    start = time.time()
    print("[log] 正在执行聚类...")
    kmeans.fit(feature_vector.values)  # 执行聚类
    print("[log] 聚类...done")
    # cluster_centers = kmeans.cluster_centers_  # 获取聚类中心坐标
    # print(cluster_centers)
    end = time.time()
    print("[log] 耗时：", end - start, "秒")
    save_pickle_data(kmeans, 'models/kmeans_model_' + str(n_clusters) + "_" + str(n_init) + ".pkl")
    return kmeans


def kmeans_plus_clustering(feature_vector, n_clusters, n_init):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', verbose=2, n_init=n_init)
    start = time.time()
    print("[log] 正在执行聚类...")
    kmeans.fit(feature_vector.values)  # 执行聚类
    print("[log] 聚类...done")
    # cluster_centers = kmeans.cluster_centers_  # 获取聚类中心坐标
    # print(cluster_centers)
    end = time.time()
    print("[log] 耗时：", end - start, "秒")
    save_pickle_data(kmeans, 'models/kmeans_model_' + str(n_clusters) + "_" + str(n_init) + ".pkl")
    return kmeans


def DBSCAN_clustering(feature_vector, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    start = time.time()
    dbscan.fit(feature_vector)
    print("[log] 聚类...done")
    end = time.time()
    print("[log] 耗时：", end - start, "秒")
    save_pickle_data(dbscan, 'models/dbscan_model_' + str(eps) + "_" + str(min_samples) + ".pkl")
    return dbscan
