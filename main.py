from dataloader import L2RDataset
from train_test import train_step, test_step, precision
from design_model import design_model
from model import RankNet
import torch
import torch.nn as nn
import datetime
import os

seed = 5
torch.manual_seed(seed=seed)

if __name__ == '__main__':

    now = datetime.datetime.now()
    now = "{0:%Y%m%d%H%M}".format(now)

    train_file = 'Toda_201911.txt'
    dataset = 'Toda_201911'
    val_file = 'Toda_201912.txt'
    train_ds = L2RDataset(file=train_file, data_id='BOATRACE')
    val_ds = L2RDataset(file=val_file, data_id='BOATRACE')

    max_epoch = 30
    dims = [10, 20, 10, 5]
    actf1 = nn.ReLU()

    layers = design_model(actf1, dims)
    rank_type = "RankNet"
    model = RankNet(layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_ndcg_score = 0

    for epoch in range(max_epoch):
        epoch_train_loss = train_step(model, train_ds, optimizer)
        print("Epoch: {} Train Loss: {}".format(epoch, epoch_train_loss))
        epoch_train_dcg = test_step(model, train_ds)
        #epoch_p = precision(model, train_ds)
        epoch_val_dcg = test_step(model, val_ds)
        for k in [1, 2, 3]:
            print("Epoch: {} Train nDCG@{}: {}".format(epoch, k, epoch_train_dcg[k]))
        #print("Epoch: {} Train P: {}".format(epoch, epoch_p))
        if epoch_val_dcg[3] > best_val_ndcg_score:
            best_epoch = epoch
            best_loss = epoch_train_loss
            best_val_ndcg_score = epoch_val_dcg[1]

            if not os.path.exists(
                    './models/{}-{}'.format(rank_type, dataset)):
                os.makedirs(
                    './models/{}-{}'.format(rank_type, dataset))
            torch.save(model,
                       './models/{}-{}/{}'.format(rank_type, dataset, now))
        print("--" * 50)

