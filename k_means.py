from sklearn import cluster
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    f = open("k-meansInp.txt","r")
    num_centroid =  int(f.readline())
    data=[]
    while True:
        temp =  f.readline()
        if temp=='':
            break
        temp= [float(i) for i in temp.split(',')]
        data.append(temp)

    data = np.array(data,dtype=float)
    return (num_centroid,data) 

def plot(data,labels):
    l=0
    plt.xlim(-0.006,0.003)
    plt.ylim( -0.0065, 0.0075 )
    for x in data:
        if labels[l] == 0 :
            plt.scatter(x[0],x[1],c='r')
        elif labels[l] == 1:
            plt.scatter(x[0],x[1],c='b')
        elif labels[l]==2:
            plt.scatter(x[0],x[1],c='g')
        else:
            plt.scatter(x[0],x[1],c='k')
        l+=1
        plt.pause(0.001)

    plt.show()
            

def main():
    centroids,data=load_data()
    k_means = cluster.KMeans(n_clusters=centroids,random_state=0)
    k_means.fit(data)
    plot(data,k_means.labels_)
    print(k_means.labels_)
    print(k_means.cluster_centers_)

if __name__ == "__main__":
    main()