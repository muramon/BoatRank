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


    def backtest(self, test_ds=None, model=None, odds_json=None, model_type='RankNet'):
        csv_file = "./backtest_{}_{}.csv".format(model_type, val_file[:-4])
        fj = open(odds_json, 'r')
        fw = open(csv_file, "w")
        fw.write("予想順位,実際の順位,単勝,複勝,2連単,2連複,3連単,3連複,拡連複,流し,ボックス\n")
        writer = csv.writer(fw)
        win_total = 0
        place_total = 0
        exacta_total = 0
        quinella_total = 0
        wide_quinella_total = 0
        trifecta_total = 0
        trio_total = 0
        odds_j = json.load(fj)
        for qid, batch_rankings, labels in test_ds:
            pred = model.predict(batch_rankings)
            pred_ar = pred.squeeze(1).detach()
            label_ar = labels.detach()
            _, argsort = torch.sort(pred_ar, descending=True, dim=0)
            prediction_label = label_ar[argsort]
            self.prediction_labels = np.append(self.prediction_labels, prediction_label.numpy())
            odds = odds_j[str(qid)]
            six_np = np.array([6, 6, 6, 6, 6, 6])
            pred_rank = six_np - prediction_label.numpy().astype(np.int64)

            # 単勝
            win = -100
            if pred_rank[0] == int(odds[0][0]):
                win += odds[0][1]
            win_total += win

            # 複勝
            place = -100
            if pred_rank[0] == int(odds[1][0]):
                place += int(odds[1][1])
            if pred_rank[0] == int(odds[2][0]):
                place += int(odds[2][1])
            place_total += place

            # 2連単
            exacta = -100
            if pred_rank[0] == int(odds[3][0][0]) and pred_rank[1] == int(odds[3][0][2]):
                exacta += int(odds[3][1])
            exacta_total += exacta

            # 2連複
            quinella = -100
            double_sorted = np.sort(pred_rank[:2])
            if double_sorted[0] == int(odds[4][0][0]) and double_sorted[1] == int(odds[4][0][2]):
                quinella += int(odds[4][1])
            quinella_total += quinella

            # 拡連複
            wide_quinella = -100
            if double_sorted[0] == int(odds[5][0][0]) and double_sorted[1] == int(odds[5][0][2]):
                wide_quinella += int(odds[5][1])
            if double_sorted[0] == int(odds[6][0][0]) and double_sorted[1] == int(odds[6][0][2]):
                wide_quinella += int(odds[6][1])
            if double_sorted[0] == int(odds[7][0][0]) and double_sorted[1] == int(odds[7][0][2]):
                wide_quinella += int(odds[7][1])
            wide_quinella_total += wide_quinella

            # 3連単
            trifecta = -100
            if pred_rank[0] == int(odds[8][0][0]) and pred_rank[1] == int(odds[8][0][2]) and pred_rank[2] == int(odds[8][0][4]):
                trifecta += int(odds[8][1])
            trifecta_total += trifecta

            # 3連複
            trio = -100
            triple_sorted = np.sort(pred_rank[:3])
            if triple_sorted[0] == int(odds[9][0][0]) and triple_sorted[1] == int(odds[9][0][2]) and triple_sorted[2] == int(odds[9][0][4]):
                trio += int(odds[9][1])
            trio_total += trio

            # csv 書き込み
            row = []
            p_3 = ""
            for p in pred_rank[:3]:
                p_3 += str(p) + "-"
            row.append(p_3.rstrip("-"))
            row.append(odds[8][0])
            row += [win, place, exacta, quinella, wide_quinella, trifecta, trio]
            writer.writerow(row)

        print("単勝:", win_total)
        print("複勝:", place_total)
        print("2連単:", exacta_total)
        print("2連複:", quinella_total)
        print("拡連複:", wide_quinella_total)
        print("3連単:", trifecta_total)
        print("3連複:", trio_total)

        all_win = win_total + place_total + exacta_total + quinella_total + wide_quinella_total + trifecta_total + trio_total
        print("収支　合計:", all_win)

        fw.write("単勝, 複勝, 2連単, 2連複, 拡連複, 3連単, 3連複, 合計\n")
        all_row = [win_total, place_total, exacta_total, quinella_total, wide_quinella_total, trifecta_total, trio_total, all_win]
        writer.writerow(all_row)

if __name__ == "__main__":
    model_path = torch.load('./models/RankNet-Toda_201911/202001092050')
    val_file = 'Toda_201912.txt'
    val_dataset = L2RDataset(file=val_file, data_id='BOATRACE')
    #race_results = get_data.RaceResults()
    #date = 191213
    #results = race_results.load(str(date))

    #odds =

    #print(odds)
    odds_json = 'Toda_201912.json'
    BackTest().backtest(test_ds=val_dataset, model=model_path, odds_json=odds_json, model_type='RankNet')


