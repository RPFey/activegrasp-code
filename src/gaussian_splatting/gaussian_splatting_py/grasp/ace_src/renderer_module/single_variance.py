import torch
import torch.nn as nn


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        # super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def forward(self, x):
        return torch.ones([len(x), 1]).to(self.device) * torch.exp(self.variance * 10.0)
