import torch
import numpy as np


class Kmeans:
    def __init__(self, n_clusters=2, max_iters=1000) -> None:
        self.n_clusters = n_clusters
        self.max_iters = max_iters

        self.centers = None
        self.labels = None
        self.iters = 0

    def fit(self, x):
        # 隨機從x中挑選出n個點做為初始化cluster的中心
        x_shape = x.shape
        if self.centers is None:
            centers_old = x[np.random.choice(
                x_shape[0], self.n_clusters, replace=False)]
        else:
            centers_old = self.centers

        for i in range(self.max_iters):
            d_clusters = torch.zeros((self.n_clusters, x_shape[0]))
            # 計算每個點跟各cluster center的距離
            for j in range(self.n_clusters):
                d = self.get_distance(centers_old[j:j+1], x)
                d_clusters[j] = d  # (2,n_batch)

            # 求每個點應該屬於哪個cluster
            labels = self.clustering(d_clusters)  # n_batch

            # 更新各cluster center的位置
            centers_new = self.update_center(labels, x)

            # 如果center沒有變化,則視為收斂
            if self.is_converge(centers_old, centers_new):
                break
            else:
                centers_old = centers_new

        self.centers = centers_new
        self.labels = labels
        self.iters = i
        return None

    def get_distance(self, center, x):
        d = torch.sum((x - center)**2, dim=1)/x.shape[1]
        d = torch.sqrt(d)
        return d  # (n_batch)

    def clustering(self, d_clusters):
        labels = torch.argmin(d_clusters, dim=0)
        return labels

    def update_center(self, labels, x):
        centers = []
        for i in range(self.n_clusters):
            mask_i = torch.where(labels == i)
            center = torch.mean(x[mask_i], dim=0)
            centers.append(center)
        return torch.stack(centers)

    def is_converge(self, centers_old, centers_new):
        if torch.equal(centers_old, centers_new):
            return True
        else:
            return False


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    n_samples = 100
    n_features = 5
    centers = 4
    x, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=centers, random_state=42)
    x = torch.tensor(x)
    clf = Kmeans(n_clusters=centers)
    clf.fit(x)
    print(clf.iters)
    print(clf.labels)
    print(y)
