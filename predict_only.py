import torch
import os
from dataloader import L2RDataset
from get_data import bangumihyo
import numpy as np


def mk_ds(date, place='Toda'):
    b = bangumihyo()
    file = open('{}_20{}.txt'.format(place, str(date)), 'w')
    bangumi = b.load(str(date))
    key = 1
    while key <= len(bangumi):
        i = 0
        while i < 6:
            try:
                if bangumi[key][i]['戸田']:
                    # print(result[key][i], "qid:", str(191101) + str(key), bangumi[key][i]['戸田'])
                    file.write(str(5-i))
                    file.write(" ")
                    file.write("qid:")
                    file.write(" ")
                    file.write(str(date) + str(key))
                    file.write(" ")
                    file.write(str(bangumi[key][i]['戸田']).replace(',', '').strip('{').strip('}'))
                    # print(str(bangumi[key][i]['戸田']))
                    file.write("\n")
            except(KeyError):
                pass
            i += 1
        key += 1

    file.close()

def predict(model=None, date=None, place=None):
    ranknet_model_path = torch.load(model)
    pred_file = '{}_20200109.txt'.format(place)#, date)
    predict_dataset = L2RDataset(file=pred_file, data_id='BOATRACE')

    for qid, batch_rankings, labels in predict_dataset:
        labels, _ = torch.sort(labels, descending=True)
        #print(qid, batch_rankings)
        pred = ranknet_model_path.predict(batch_rankings)
        #print(qid, pred)
        pred_ar = pred.squeeze(1).detach()
        label_ar = labels.detach()
        _, argsort = torch.sort(pred_ar, descending=True, dim=0)
        prediction_label = label_ar[argsort]
        six_np = np.array([6, 6, 6, 6, 6, 6])
        pred_rank = six_np - prediction_label.numpy().astype(np.int64)
        print(qid, pred_rank)


if __name__ == "__main__":
    seed = 5
    torch.manual_seed(seed=seed)
    mk_ds(200109)
    predict(model='./models/RankNet-Toda_201908-10/202001122029', date=200109, place='Toda')


