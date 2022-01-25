import torch


class Kmeans:
    def __init__(self, n_clusters=2) -> None:
        self.n_clusters = n_clusters
        self.centers = [[] for i in range(self.n_clusters)]
        pass

    def fit(self, x):
        x = torch.randn((64, 15))
        x_shape = x.shape
        noise = torch.randn((self.n_clusters, x.shape[-1]))
        mu = torch.mean(x, dim=0, keepdim=True)
        centers = mu + noise
        d_temp = []
        for i in range(self.n_clusters):
            d = self.get_distance(centers[i:i+1], x)
            d_temp.append(d)

    def get_distance(self, center, x):
        d = torch.sum((x - center)**2, dim=1)
        d = torch.sqrt(d)/x.shape[1]
        return d
