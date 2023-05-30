# coding=utf-8
def find_cluster_members(data, labels, i):
    return data[labels == i]


def find_ports(data, kmeans):
    dic = {
        "import": [],
        "export": [],
        "mooring": []
    }
