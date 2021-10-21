import torch
from torch.autograd import Variable

def si_sdr(gt, out, eps=1e-4):
    gt = torch.where(abs(gt)>eps,gt,eps)
    gt_norm = torch.sum(gt ** 2, dim=-1, keepdims=True)
    # print(gt_norm.shape)  # torch.Size([2, 1, 1])
    optimal_scaling = torch.sum(gt * out, dim=-1, keepdims=True) / gt_norm
    # print(optimal_scaling.shape)  # torch.Size([2, 1, 1])
    projection = optimal_scaling * gt  # torch.Size([2, 1, 1])* torch.Size(([2,1,2])
    # print(projection.shape)  # torch.Size([2, 1, 2])
    noise = out - projection
    noise = torch.where(abs(noise)>eps,noise,eps)
    ratio = torch.sum(projection ** 2, dim=-1) / torch.sum(noise ** 2, dim=-1)
    # print(ratio.shape)  # torch.Size([2, 1])
    return 10 * torch.log10(ratio)


if __name__ == "__main__":
    reference = Variable(torch.tensor(
        [[0, 0], [1, 1]], dtype=torch.float64).reshape(2, 1, 2), requires_grad=True)

    output = Variable(torch.tensor([[1, 1], [2, 3]], dtype=torch.float64).reshape(
        2, 1, 2), requires_grad=True)
    
    s = si_sdr(reference, output).sum()
    print(s)
    s.backward()
