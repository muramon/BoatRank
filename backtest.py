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
            csv_file = "./backtest/listnet/long_backtest_{}_model{}_{}.csv".format(model_type, model_name, test_file[8:-4])
        model = torch.load(model)
        fj = open(odds_json, 'r')
        fw = open(csv_file, "w")
        if three:
            fall = open('./backtest/123_triu_all_backtest.csv', 'a')
        else:
            fall = open('./backtest/long_all_backtest.csv', 'a')
        writerfall = csv.writer(fall)
        fw.write("レース数, 予想順位,実際の順位,単勝,複勝,2連単,2連複,拡連複,3連単,3連複\n")
        writer = csv.writer(fw)
        win_total = 0
        place_total = 0
        exacta_total = 0
        quinella_total = 0
        wide_quinella_total = 0
        trifecta_total = 0
        trio_total = 0
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

            # 単勝
            win = -100
            try:
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
                row.append(qid)
                p_3 = ""
                for p in pred_rank[:3]:
                    p_3 += str(p) + "-"
                row.append(p_3.rstrip("-"))
                row.append(odds[8][0])
                row += [win, place, exacta, quinella, wide_quinella, trifecta, trio]
                writer.writerow(row)
            except(ValueError):
                continue

        print("単勝:", win_total)
        print("複勝:", place_total)
        print("2連単:", exacta_total)
        print("2連複:", quinella_total)
        print("拡連複:", wide_quinella_total)
        print("3連単:", trifecta_total)
        print("3連複:", trio_total)

        all_win = win_total + place_total + exacta_total + quinella_total + wide_quinella_total + trifecta_total + trio_total
        print("収支　合計:", all_win)

        fw.write("総レース数, 単勝, 複勝, 2連単, 2連複, 拡連複, 3連単, 3連複, 合計\n")
        all_row = [race_num, win_total, place_total, exacta_total, quinella_total, wide_quinella_total, trifecta_total, trio_total, all_win]
        writer.writerow(all_row)
        all_rows = [kaijo, race_num, win_total, place_total, exacta_total, quinella_total, wide_quinella_total, trifecta_total,
                   trio_total, all_win]
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


