import get_data
import numpy as np
import os
from dataloader import L2RDataset
import torch
import json
import csv

class BackTest():
    def __init__(self):
        #self.race_results = get_data.RaceResults()
        #self.race_results.load(str(date))
        #self.odds = odds_json#self.race_results.odds
        self.prediction_labels = np.array([])


    def backtest(self, test_ds=None, model=None, odds_json=None, model_type='RankNet', test_file = 'dataset/Toda_191201-191231.txt', kaijo=None, three=False):
        model_name = model[-23:]
        if three:
            csv_file = "./backtest/123_triu_backtest_{}_model{}_{}.csv".format(model_type, model_name, test_file[8:-4])
        else:
            csv_file = "./nirentan_thresh_test/long_backtest_{}_model{}_{}.csv".format(model_type, model_name, test_file[8:-4])
        model = torch.load(model)
        fj = open(odds_json, 'r')
        fw = open(csv_file, "w")
        if three:
            fall = open('./backtest/123_triu_all_backtest.csv', 'a')
        else:
            fall = open('./nirentan_thresh_test/niren_long_all_backtest.csv', 'a')
        writerfall = csv.writer(fall)
        fwwrite = "レース数, 予想順位,実際の順位"
        win_total = {}
        count = {}
        for i in range(41):
            win_total[i] = 0
            count[i] = 0
            threshold = "," + str(100 * (i + 2)) + "~" + str(100 * (i + 3))
            fwwrite += threshold
        fwwrite += "\n"
        fw.write(fwwrite)
        writer = csv.writer(fw)
        # place_total = 0
        # exacta_total = 0
        # quinella_total = 0
        # wide_quinella_total = 0
        # trifecta_total = 0
        # trio_total = 0
        odds_j = json.load(fj)
        race_num = 0
        fff = open('batchrankings.txt', 'w')
        for qid, batch_rankings, labels in test_ds:
            #labels, _ = torch.sort(labels, descending=True)
            labels = torch.tensor([5.,4.,3.,2.,1.,0.])
            fff.write(str(qid))
            fff.write(str(batch_rankings))
            #fff.write('\n')
            race_num += 1
            pred = model.predict(batch_rankings)
            #fff.write(str(pred))
            fff.write('\n')
            pred_ar = pred.squeeze(1).detach()
            label_ar = labels.detach()
            _, argsort = torch.sort(pred_ar, descending=True, dim=0)
            prediction_label = label_ar[argsort]
            self.prediction_labels = np.append(self.prediction_labels, prediction_label.numpy())
            odds = odds_j[str(qid)]
            six_np = np.array([6, 6, 6, 6, 6, 6])
            pred_rank = six_np - prediction_label.numpy().astype(np.int64)

            try:
                # 2連単 閾値試し
                win = {}
                for i in range(40):
                    win[i] = 0
                    if int(odds[3][1]) >= (100 * (i + 2)) and int(odds[3][1]) < (100 * (i + 3)):
                        win[i] = -100
                        count[i] += 1
                        if pred_rank[0] == int(odds[3][0][0]) and pred_rank[1] == int(odds[3][0][2]):
                            win[i] += int(odds[3][1])
                        win_total[i] += win[i]
                    else:
                        i += 1
                if int(odds[3][1]) >= 4200:
                    win[40] = -100
                    count[i] += 1
                    double_sorted = np.sort(pred_rank[:2])
                    if pred_rank[0] == int(odds[3][0][0]) and pred_rank[1] == int(odds[3][0][2]):
                        win[40] += int(odds[3][1])
                    win_total[40] += win[40]
                else:
                    win[40] = 0
                    win_total[40] += 0

                # csv 書き込み
                row = []
                row.append(qid)
                p_3 = ""
                for p in pred_rank[:3]:
                    p_3 += str(p) + "-"
                row.append(p_3.rstrip("-"))
                row.append(odds[8][0])
                for i in range(21):
                    row += [win[i]]
                writer.writerow(row)

            except(ValueError):
                continue
        for i in range(41):
            print("単勝 ", 100 * (i + 2), "~", 100 * (i + 3), ":", win_total[i])


        totalwrite = "総レース数"
        for i in range(41):
            total = "," + str(100 * (i + 2)) + "~" + str(100 * (i + 3))
            totalwrite += total
        totalwrite += ", 2200~, \n"
        fw.write(totalwrite)
        all_row = [race_num]
        for i in range(41):
            all_row += [win_total[i]]
        writer.writerow(all_row)

        writer.writerow(["還元率"])
        count_row = [race_num]
        for i in range(41):
            try:
                count_row += [round((win_total[i]+(count[i]*100))/(count[i]*100)*100, 2)]
            except:
                count_row += [0]
        writer.writerow(count_row)

        all_rows = [kaijo]
        all_rows += all_row
        writerfall.writerow(all_rows)

if __name__ == "__main__":
    #ranknet_model_path = './models/RankNet-Toda_190801-191031/trained_at_202001122249'
    #lambdarank_model_path = './models/LambdaRank-Toda_190801-191031/trained_at_202001122252'
    ranknet_model_path = './models/RankNet-Shuffle_Toda_190801-191031/trained_at_202001131829'
    lambdarank_model_path = './models/LambdaRank-Shuffle_Toda_190801-191031/trained_at_202001131832'
    test_file = 'dataset/Toda_191201-191231.txt'
    #test_file = 'dataset/Toda_200109-200109.txt'
    #val_file = 'dataset/Toda_200101-200112.txt'
    test_dataset = L2RDataset(file=test_file, data_id='BOATRACE')
    odds_json = 'dataset/Toda_191201-191231.json'
    #odds_json = 'dataset/Toda_200109-200109.json'
    #odds_json = 'dataset/Toda_200101-200112.json'
    BackTest().backtest(test_ds=test_dataset, model=ranknet_model_path, odds_json=odds_json, model_type='RankNet', test_file = 'dataset/Toda_191201-191231.txt', Shuffle=True)
    BackTest().backtest(test_ds=test_dataset, model=lambdarank_model_path, odds_json=odds_json, model_type='LambdaRank', test_file = 'dataset/Toda_191201-191231.txt', Shuffle=True)


