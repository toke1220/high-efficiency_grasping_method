import sklearn.metrics as metrics
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 构造自定义函数，用于绘制不同k值和对应轮廓系数的折线图
def k_silhouette(X, clusters):
    K = range(2,clusters+1)
# 构建空列表，用于存储个中簇数下的轮廓系数
    S = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        labels = kmeans.labels_
        S.append(metrics.silhouette_score(X, labels, metric='euclidean'))  # 调用字模块metrics中的silhouette_score函数，计算轮廓系数
   
    plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']# 中文和负号的正常显示
    plt.rcParams['axes.unicode_minus'] = False

    plt.style.use('ggplot')# 设置绘图风格
    
    plt.plot(K, S, 'b*-')  # 绘制K的个数与轮廓系数的关系
    plt.xlabel('the number of chu')
    plt.ylabel('lun kuo xi shu')
    
    plt.show() # 显示图形


#k_silhouette(y, 50)# 自定义函数的调用（指定原始数据和选取范围）