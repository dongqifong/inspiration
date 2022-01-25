import torch


class Kmeans:
    def __init__(self, n_clusters=2, max_iters=10000) -> None:
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        pass

    def fit(self, x):
        x = torch.randn((64, 15))

        # 初始化各cluster中心點
        x_shape = x.shape
        noise = torch.randn((self.n_clusters, x_shape[-1]))  # (2,n_features)
        mu = torch.mean(x, dim=0, keepdim=True)  # (1,n_features)
        centers_old = mu + noise  # (2,n_features)

        # clusters = [[]for i in range(self.n_clusters)]
        # d_clusters = [[]for i in range(self.n_clusters)]

        for i in range(self.max_iters):

            # 計算每個點跟各cluster center的距離
            for i in range(self.n_clusters):
                d = self.get_distance(centers_old[i:i+1], x)
                d_clusters[i] = d  # (2,n_batch)

            # 求每個點應該屬於哪個cluster
            clusters_idx = self.clustering(torch.tensor(d_clusters))  # n_batch
            for i in range(self.n_clusters):
                mask_i = torch.where(clusters_idx == i)
                clusters[i] = x[mask_i]

            # 更新各cluster center的位置
            centers_new = self.update_center(clusters)

            # 如果center沒有變化,則視為收斂
            if self.is_converge(centers_old, centers_new):
                break
            else:
                centers_old = centers_new

    def get_distance(self, center, x):
        d = torch.sum((x - center)**2, dim=1)
        d = torch.sqrt(d)/x.shape[1]
        return d  # (n_batch)

    def clustering(self, d_clusters):
        cluster_idx = torch.argmin(d_clusters, dim=0)
        return cluster_idx

    def update_center(self, clusters):
        centers = []
        for i in range(len(clusters)):
            center = torch.mean(clusters[i], dim=0)
            centers.append(center)
        return torch.tensor(centers)  # (2,n_features)

    def is_converge(centers_old, centers_new):
        if torch.equal(centers_old, centers_new):
            return True
        else:
            return False


a = torch.tensor([[1.1], [2.0]])
b = torch.tensor([[1.0], [2.0]])
print(torch.equal(a, b))
