from dataloader import L2RDataset
from train_test import train_step, test_step, precision
from design_model import design_model
from model import RankNet
from model_lambdarank import LambdaRank
import torch
import torch.nn as nn
import datetime
import os

seed = 5
torch.manual_seed(seed=seed)

def train(trains='190801', trainl='191031', vals='20191101', vall='201911030', place='戸田'):
    for rank_type in ["RankNet", 'LambdaRank']:
        now = datetime.datetime.now()
        now = "{0:%Y%m%d%H%M}".format(now)
        #trains = '190801'
        #trainl = '191031'

        train_file = 'dataset/{}_{}-{}.txt'.format(place, trains, trainl)
        dataset = '{}_{}-{}'.format(place, trains, trainl)
        val_file = 'dataset/{}_{}-{}.txt'.format(place, vals, vall)
        train_ds = L2RDataset(file=train_file, data_id='BOATRACE')
        val_ds = L2RDataset(file=val_file, data_id='BOATRACE')

        max_epoch = 100
        dims = [10, 20, 10, 5]
        actf1 = nn.ReLU()

        layers = design_model(actf1, dims)
        if rank_type == "RankNet":
            model = RankNet(layers)
        elif rank_type == "LambdaRank":
            model = LambdaRank(layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        file = open('results/shuffle_{}_Toda{}-{}_trained_at_{}.txt'.format(rank_type, trains, trainl, now), 'w')
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
                best_val_ndcg_score = epoch_val_dcg[3]
                best_val_ndcg = epoch_val_dcg

                #if not os.path.exists(
                #        './models/{}-{}'.format(rank_type, dataset)):
                #    os.makedirs(
                #       './models/{}-{}'.format(rank_type, dataset))
                torch.save(model,
                           './models/{}-{}-{}'.format(rank_type, dataset, place))
            print("--" * 50)

        for k in [1, 2, 3]:
            file.write('Valid ndcg@{}'.format(k))
            file.write(str(best_val_ndcg[k]))
            file.write('\n')

if __name__ == '__main__':
    train()
