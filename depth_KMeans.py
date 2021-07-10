import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def depth_data_KMeans(y,k):
    for k in range(2,k+1):
        clf = KMeans(n_clusters=k) #设定k  这里就是调用KMeans算法，同时默认参数就是KMeans++
        s = clf.fit(y) #加载数据集合
        numSamples = len(y) 
        centroids_label = clf.labels_
        print (centroids_label, type(centroids_label)) #显示各个点的，聚类标签
        #print (clf.inertia_)  #显示聚类效果 评价指标
        mark_data = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        #画出所有样例点 属于同一分类的绘制同样的颜色
        for i in range(numSamples):
            plt.plot(y[i][0], 0, mark_data[clf.labels_[i]]) #mark[markIndex])
        mark_centroids = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 画出质点，用特殊图型
        mark_xvline = ['r', 'b', 'g', 'k']
        centroids =  clf.cluster_centers_
        for i in range(k):
            plt.plot(centroids[i][0], 0, mark_centroids[i], markersize = 12)
            print('><><><><><><><><><><><>')
            print(centroids[i][0])   #输出聚类中心值
            plt.axvline(centroids[i][0], ymin=0.3, ymax=0.7, color=mark_xvline[i])
        plt.show()
    return 0