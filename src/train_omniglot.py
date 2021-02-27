from omniglot_dataset import OmniglotDataset
from prototypical_batch_sampler import PrototypicalBatchSampler
import torch
import os
import params
from model import EmbeddingNet
from prototypical_loss import PrototypicalLoss as Loss

if __name__ == "__main__":

    device = 'cuda:0' if torch.cuda.is_available() and params.use_cuda else 'cpu'
    print(device)

    train_dataset = OmniglotDataset(mode='train', root=os.path.dirname(os.getcwd()))
    train_sampler = PrototypicalBatchSampler(labels=train_dataset.y,
                                    classes_per_it=params.classes_per_it_tr,
                                    num_samples=params.num_support_tr + params.num_query_tr,
                                    iterations=params.iterations)

    dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler)

    model = EmbeddingNet(img_channels=1, hidden_channels=64, embedded_channels=64).to(device)
    #print(model)

    loss_fn = Loss(params.num_support_tr).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=params.lr_scheduler_gamma,
                                    step_size=params.lr_scheduler_step)

    for epoch in range(params.epochs):
        tr_iter = iter(dataloader)
        for batch in tr_iter:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            #print(x.size(), y.size())
            x_embed = model(x)
            #print(x_embed.size())
            loss_val, acc_val = loss_fn(x_embed, y)
            loss_val.backward()
            optimizer.step()
        lr_scheduler.step()
        '''
        TODO: Implement validation
        '''


