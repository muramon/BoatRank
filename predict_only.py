import torch
import os
from dataloader import L2RDataset
from get_data import bangumihyo
import numpy as np


def mk_ds(date, place='戸田'):
    b = bangumihyo()
    file = open('dataset/predict_{}_20{}.txt'.format(place, str(date)), 'w')
    b.download("20{}-{}-{}".format(str(date)[0:2], str(date)[2:4], str(date)[4:6]), "20{}-{}-{}".format(str(date)[0:2], str(date)[2:4], str(date)[4:6]))
    bangumi = b.load(str(date))
    key = 1
    while key <= len(bangumi):
        i = 0
        while i < 6:
            try:
                if bangumi[key][i][place]:
                    file.write(str(1))
                    file.write(" ")
                    file.write("qid:")
                    file.write(" ")
                    file.write(str(date) + str(key))
                    file.write(" ")
                    file.write(str(bangumi[key][i][place]).replace(',', '').strip('{').strip('}'))
                    # print(str(bangumi[key][i][place]))
                    file.write("\n")
            except(KeyError):
                pass
            i += 1
        key += 1

    file.close()

def predict(model=None, date=None, place=None):
    ranknet_model_path = torch.load(model)
    pred_file = 'dataset/predict_{}_20{}.txt'.format(place, str(date))
    predict_dataset = L2RDataset(file=pred_file, data_id='BOATRACE')
    predicted = open('predicted/predicted_{}_20{}.txt'.format(place, str(date)), 'w')

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
        predicted.write(str(qid))
        predicted.write(str(pred_rank))
        predicted.write('\n')
        print(qid, pred_rank)


if __name__ == "__main__":
    seed = 5
    torch.manual_seed(seed=seed)
    place = '戸田'
    datep=210422
    mk_ds(datep, place=place)
    predict(model='./models/RankNet-平和島_180101-190531-平和島', date=datep, place=place)


