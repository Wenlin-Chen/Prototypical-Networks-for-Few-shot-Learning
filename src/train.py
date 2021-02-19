from omniglot_dataset import OmniglotDataset
from prototypical_batch_sampler import PrototypicalBatchSampler
import torch
import os
import params

if __name__ == "__main__":

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_dataset = OmniglotDataset(mode='train', root=os.path.dirname(os.getcwd()))
    train_sampler = PrototypicalBatchSampler(labels=train_dataset.y,
                                    classes_per_it=params.classes_per_it,
                                    num_samples=params.num_support_tr + params.num_query_tr,
                                    iterations=params.iterations)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

    tr_iter = iter(dataloader)
    for batch in tr_iter:
        x, y = batch
        print(x,y)