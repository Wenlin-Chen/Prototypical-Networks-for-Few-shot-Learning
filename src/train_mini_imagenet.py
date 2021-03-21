from mini_imagenet_dataloader import MiniImageNetDataLoader
from prototypical_batch_sampler import PrototypicalBatchSampler
import torch
import os
import params_mini_imagenet as params
from model import EmbeddingNet
from prototypical_loss import PrototypicalLoss as Loss
import numpy as np


def set_seed(seed):
    torch.cuda.cudnn_enabled = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def train(dataloader_train, dataloader_test, model, loss_fn, optimizer, lr_scheduler, device):

    tr_loss = []
    tr_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0.0
    best_epoch = 0
    best_model_state = None

    num_train_batches = len(dataloader.train_filenames)//(dataloader.num_samples_per_class* params.classes_per_it_tr)
    num_val_batches = len(dataloader.val_filenames)//(dataloader.num_samples_per_class* params.classes_per_it_val)


    for epoch in range(params.epochs):
        print('======= Epoch: {} ======='.format(epoch))
        # train
        model.train()
        for i in range(num_train_batches):
            optimizer.zero_grad()

            episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
            dataloader_train.get_batch(phase='train', idx=i)

            # change from 1-hot encoding to integer
            episode_train_label = np.argmax(episode_train_label, axis=1)
            episode_test_label = np.argmax(episode_test_label, axis=1)
            
            x = torch.tensor(np.moveaxis(np.concatenate((episode_train_img,episode_test_img)), 3, 1),dtype=torch.float)
            y = torch.tensor(np.concatenate((episode_train_label,episode_test_label)),dtype=torch.int)
            x, y = x.to(device), y.to(device)
            #print(x.size(), y.size())
            x_embed = model(x)

            #print(x_embed.size())
            loss_val, acc_val = loss_fn(x_embed, y, params.shot_num)

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
            for i in range(num_val_batches):
                episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
                dataloader_test.get_batch(phase='val', idx=i)

                # change from 1-hot encoding to integer
                episode_train_label = np.argmax(episode_train_label, axis=1)
                episode_test_label = np.argmax(episode_test_label, axis=1)
                
                x = torch.from_numpy(np.moveaxis(np.concatenate((episode_train_img,episode_test_img)), 3, 1)).float()
                y = torch.from_numpy(np.concatenate((episode_train_label,episode_test_label))).float()
                x, y = x.to(device), y.to(device)
                x_embed = model(x)
                loss_val, acc_val = loss_fn(x_embed, y, params.shot_num)
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

def test(dataloader_test, model, loss_fn, device):
    test_acc = []

    num_val_batches = len(dataloader.test_filenames)//(dataloader.num_samples_per_class* params.classes_per_it_val)

    model.eval()
    with torch.no_grad():
        for epoch in range(15):
            for i in range(num_test_batches):
                episode_train_img, episode_train_label, episode_test_img, episode_test_label = \
                dataloader_test.get_batch(phase='test', idx=i)

                # change from 1-hot encoding to integer
                episode_train_label = np.argmax(episode_train_label, axis=1)
                episode_test_label = np.argmax(episode_test_label, axis=1)
                
                x = torch.from_numpy(np.moveaxis(np.concatenate((episode_train_img,episode_test_img)), 3, 1)).float()
                y = torch.from_numpy(np.concatenate((episode_train_label,episode_test_label))).int()

                x, y = x.to(device), y.to(device)
                x_embed = model(x)
                _, acc = loss_fn(x_embed, y, params.num_support_val)
                test_acc.append(acc.item())

    avg_acc = np.mean(test_acc)
    print('Test Accuracy: {}'.format(avg_acc))

    return avg_acc


def run():
    set_seed(params.seed)

    device = 'cpu'#'cuda:0' if torch.cuda.is_available() and params.use_cuda else 'cpu'
    print(device)

    dataloader_train = MiniImageNetDataLoader(
        shot_num = params.shot_num, 
        way_num = params.classes_per_it_tr,
        episode_test_sample_num = params.num_query )

    dataloader_train.generate_data_list(phase='train')
    dataloader_train.load_list(phase='train')

    dataloader_test = MiniImageNetDataLoader(
        shot_num = params.shot_num, 
        way_num = params.classes_per_it_val,
        episode_test_sample_num = params.num_query )
    dataloader_test.generate_data_list(phase='val')
    dataloader_test.generate_data_list(phase='test')
    dataloader_test.load_list(phase='val')
    dataloader_test.load_list(phase='test')


    model = EmbeddingNet(img_channels=3, hidden_channels=64, embedded_channels=64).to(device)

    loss_fn = Loss(device).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=params.lr_scheduler_gamma,
                                    step_size=params.lr_scheduler_step)


    tr_stats = train(dataloader_train, dataloader_test, model, loss_fn, optimizer, lr_scheduler, device)

    best_model_state = tr_stats[-1]
    model.load_state_dict(best_model_state)
    test_acc = test(dataloader_test, model, loss_fn, device)

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    run()