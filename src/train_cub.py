from cub_dataset import CUB, split_class, get_split
from prototypical_batch_sampler import PrototypicalBatchSampler
import params_cub as params
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np 
import torch

def set_seed(seed):
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_dataloader(mode):
    """
    Mode: 'train', 'val' or 'test'
    """
    PATH = Path(params.CUB_data_path)

    images = pd.read_csv(PATH/"images.txt", header=None, sep=" ")
    images.columns = ["id", "name"]

    labels = pd.read_csv(PATH/"image_class_labels.txt", header=None, sep=" ")
    labels.columns = ["id", "label"]

    class_indices = split_class(params.n_class, params.n_class_train,
                                params.n_class_val, mode)

    x, y = get_split(class_indices, params.samples_per_class, 
                    images, labels)
    
    if mode == 'train':
        classes_per_it = params.classes_per_it_tr
        num_samples = params.num_query_tr # zero-shot
    else:
        classes_per_it = params.classes_per_it_val
        num_samples = params.num_query_val # zero-shot

    is_train = True if mode == 'train' else False

    dataset = CUB(PATH, x, y, train=is_train, transform=True)

    sampler = PrototypicalBatchSampler(labels=dataset.y,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=params.iterations)
    # dataloader = DataLoader(dataset, batch_size=params.batch_size, 
    #                     shuffle=True, num_workers=params.num_workers)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

    return dataloader

# def train(tr_dataloader, model, loss_fn, optimizer, lr_scheduler, val_dataloader, device):
def train(tr_dataloader):
    # tr_loss = []
    # tr_acc = []
    # val_loss = []
    # val_acc = []
    # best_acc = 0.0
    # best_epoch = 0
    # best_model_state = None

    for epoch in range(params.epochs):
        print('======= Epoch: {} ======='.format(epoch))

        # train
        # model.train() # TO DO: GoogLeNet
        tr_iter = iter(tr_dataloader)
        for batch in tr_iter:
            # optimizer.zero_grad()
            x, y = batch
            print(x.shape, y.shape) # [500, 3, 224, 224], [500]
            # x, y = x.to(device), y.to(device)
            #print(x.size(), y.size())
            # x_embed = model(x) # TO DO: GoogLeNet
            #print(x_embed.size())
            # loss_val, acc_val = loss_fn(x_embed, y, params.num_support_tr)
            # loss_val.backward()
            # optimizer.step()

    #         tr_loss.append(loss_val.item())
    #         tr_acc.append(acc_val.item())

    #     lr_scheduler.step()

    #     avg_loss = np.mean(tr_loss[-params.iterations:])
    #     avg_acc = np.mean(tr_acc[-params.iterations:])
    #     print('Average Train Loss: {}, Average Train Accuracy: {}'.format(avg_loss, avg_acc))
        
    #     # validation
    #     model.eval()
    #     with torch.no_grad():
    #         val_iter = iter(val_dataloader)
    #         for batch in val_iter:
    #             x, y = x.to(device), y.to(device)
    #             x_embed = model(x)
    #             loss_val, acc_val = loss_fn(x_embed, y, params.num_support_val)
    #             val_loss.append(loss_val.item())
    #             val_acc.append(acc_val.item())

    #     avg_loss = np.mean(val_loss[-params.iterations:])
    #     avg_acc = np.mean(val_acc[-params.iterations:])
    #     print('Average Validation Loss: {}, Average Validation Accuracy: {}'.format(avg_loss, avg_acc))
    #     if avg_acc >= best_acc:
    #         best_acc = avg_acc
    #         best_epoch = epoch
    #         best_model_state = model.state_dict()

    # print('Best validation accuracy was {}, achieved in epoch {}'.format(best_acc, best_epoch))

    # return tr_loss, tr_acc, val_acc, val_acc, best_epoch, best_acc, best_model_state

def run():
    set_seed(params.seed)

    device = 'cuda:0' if torch.cuda.is_available() and params.use_cuda else 'cpu'
    print(device)

    tr_dataloader = get_dataloader('train')
    # train(tr_dataloader, model, loss_fn, optimizer, lr_scheduler, val_dataloader, device)
    train(tr_dataloader) # temporary, for debugging