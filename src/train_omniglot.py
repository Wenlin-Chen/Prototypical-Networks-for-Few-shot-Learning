from omniglot_dataset import OmniglotDataset
from prototypical_batch_sampler import PrototypicalBatchSampler
import torch
import os
import params_omniglot as params
from model import EmbeddingNet
from prototypical_loss import PrototypicalLoss as Loss
import numpy as np


def set_seed(seed):
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_dataloader(mode):
    if mode == 'train':
        classes_per_it = params.classes_per_it_tr
        num_samples = params.num_support_tr + params.num_query_tr
    else:
        classes_per_it = params.classes_per_it_val
        num_samples = params.num_support_val + params.num_query_val

    dataset = OmniglotDataset(mode=mode, root=os.path.dirname(os.getcwd()))
    sampler = PrototypicalBatchSampler(labels=dataset.y,
                                       classes_per_it=classes_per_it,
                                       num_samples=num_samples,
                                       iterations=params.iterations)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    return dataloader

def train(tr_dataloader, model, loss_fn, optimizer, lr_scheduler, val_dataloader, device):

    tr_loss = []
    tr_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(params.epochs):
        print('======= Epoch: {} ======='.format(epoch))

        # train
        model.train()
        tr_iter = iter(tr_dataloader)
        for batch in tr_iter:
            optimizer.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            #print(x.size(), y.size())
            x_embed = model(x)
            #print(x_embed.size())
            loss_val, acc_val = loss_fn(x_embed, y, params.num_support_tr)
            loss_val.backward()
            optimizer.step()

            tr_loss.append(loss_val.item())
            tr_acc.append(acc_val.item())

        lr_scheduler.step()

        avg_loss = np.mean(tr_loss[-params.iterations:])
        avg_acc = np.mean(tr_acc[-params.iterations:])
        print('Average Train Loss: {}, Average Train Accuracy: {}'.format(avg_loss, avg_acc))
        
        # validation
        model.eval()
        with torch.no_grad():
            val_iter = iter(val_dataloader)
            for batch in val_iter:
                x, y = x.to(device), y.to(device)
                x_embed = model(x)
                loss_val, acc_val = loss_fn(x_embed, y, params.num_support_val)
                val_loss.append(loss_val.item())
                val_acc.append(acc_val.item())

        avg_loss = np.mean(val_loss[-params.iterations:])
        avg_acc = np.mean(val_acc[-params.iterations:])
        print('Average Validation Loss: {}, Average Validation Accuracy: {}'.format(avg_loss, avg_acc))
        if avg_acc >= best_acc:
            best_acc = avg_acc
            best_epoch = epoch
            best_model_state = model.state_dict()

    print('Best validation accuracy was {}, achieved in epoch {}'.format(best_acc, best_epoch))

    return tr_loss, tr_acc, val_acc, val_acc, best_epoch, best_acc, best_model_state

def test(test_dataloader, model, loss_fn, device):
    test_acc = []

    model.eval()
    with torch.no_grad():
        for epoch in range(10):
            test_iter = iter(test_dataloader)
            for batch in test_iter:
                x, y = batch
                x, y = x.to(device), y.to(device)
                x_embed = model(x)
                _, acc = loss_fn(x_embed, y, params.num_support_val)
                test_acc.append(acc.item())

    avg_acc = np.mean(test_acc)
    print('Test Accuracy: {}'.format(avg_acc))

    return avg_acc


def run():
    set_seed(params.seed)

    device = 'cuda:0' if torch.cuda.is_available() and params.use_cuda else 'cpu'
    print(device)

    tr_dataloader = get_dataloader('train')
    val_dataloader = get_dataloader('val')
    test_dataloader = get_dataloader('test')

    model = EmbeddingNet(img_channels=1, hidden_channels=64, embedded_channels=64).to(device)

    loss_fn = Loss(device).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=params.lr_scheduler_gamma,
                                    step_size=params.lr_scheduler_step)


    tr_stats = train(tr_dataloader, model, loss_fn, optimizer, lr_scheduler, val_dataloader, device)

    best_model_state = tr_stats[-1]
    best_model = model.load_state_dict(best_model_state)
    test_acc = test(test_dataloader, model, loss_fn, device)


if __name__ == "__main__":
    run()