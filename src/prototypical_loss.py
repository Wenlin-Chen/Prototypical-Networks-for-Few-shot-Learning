import torch

class PrototypicalLoss(touch.nn.Module):
    def __init__(self, n_support):
        super(PrototypicalLoss).__init__(self)
        self.n_support = n_support

    def forward(self, input, target):
        return self.prototypical_loss(input, target, self.n_support)

    def euclidean_dist(self, x1, x2):
        """
        Euclidean distance (p=2) between two tensors x1, x2
        x1 shape: B x P x M
        x2 shape: B x R x M
        output shape: B x P x R
        """
        return torch.cdist(x1, x2, p=2)

    def prototypical_loss(self, input, target, n_support):
        """
        TO IMPLEMENT:
        Average samples to get prototype center and compute 
        loss w.r.t Euclidean distance between samples and center
        """
        raise NotImplementedError