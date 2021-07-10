import numpy as np
import matplotlib.pyplot as plt
from numpy.core.records import array

def detect_outliers(df, step):
    outlier_data = []
    if not type(df) is list:
        df = df.tolist()
    # 1st quartile (25%)
    Q1 = np.percentile(df, 25)
    # 3rd quartile (75%)
    Q3 = np.percentile(df, 75)
    # Interquartile range (IQR)
    IQR = Q3 - Q1

    # outlier step
    outlier_step = step * IQR
    for nu in df:
        if (nu < Q1 - outlier_step) or (nu > Q3 + outlier_step):
            outlier_data.append(nu)
    df = np.setdiff1d(df,outlier_data)
    return df

if __name__ == '__main__':
    df = [-3331,652,653,4,11111,650,623,653,642,658,659,0,689]
    tmp = np.copy(df)
    Outliers_to_drop = detect_outliers(tmp)
    # Drop outliers
    print(Outliers_to_drop)
    labels=['Depth_data']
    flierprops = {'marker':'o','markerfacecolor':'red','color':'black'}
    #flierprops = {'marker':'o','color':'red'}
    plt.boxplot(df, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=1.5)
    plt.show()
    plt.boxplot(Outliers_to_drop, meanline=True, notch=True, labels=labels, flierprops=flierprops, whis=1.5)
    plt.show()
theta = 8.00  
theta = 8.00 
theta = 8.00   
theta = 8.00 
theta = 8.00 
