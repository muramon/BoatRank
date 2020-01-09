import get_data
import numpy as np
import os
from dataloader import L2RDataset
import torch

class BackTest():
    def __init__(self, date):
        self.race_results = get_data.RaceResults()
        self.race_results.load(str(date))
        self.odds = self.race_results.odds
        self.prediction_labels = np.array([])


    def backtest(self, test_ds=None, model=None):
        for batch_rankings, labels in test_ds:
            pred = model.predict(batch_rankings)
            pred_ar = pred.squeeze(1).detach()
            label_ar = labels.detach()
            _, argsort = torch.sort(pred_ar, descending=True, dim=0)
            prediction_label = label_ar[argsort]
            #print("label_ar", label_ar)
            #print("prediction_label", prediction_label)
            #print(prediction_label.numpy())
            self.prediction_labels = np.append(self.prediction_labels, prediction_label.numpy())
            #print(self.prediction_labels.shape)

            #trainX, trainY, trainOdds, testX, testY, testOdds = self.prepare_data()
            #trainX = trainX.reshape([trainX.shape[0] * trainX.shape[1], trainX.shape[2]])
            #trainY = trainY.reshape([trainY.shape[0] * trainY.shape[1], trainY.shape[2]])
            #trainOdds = trainOdds.reshape([trainOdds.shape[0] * trainOdds.shape[1], trainOdds.shape[2], trainOdds.shape[3]])
            #testX = testX.reshape([testX.shape[0] * testX.shape[1], testX.shape[2]])
            #testY = testY.reshape([testY.shape[0] * testY.shape[1], testY.shape[2]])
        testOdds = np.array(self.odds)
        #testOdds = testOdds.reshape([testOdds.shape[0] * testOdds.shape[1], testOdds.shape[2]])#, testOdds.shape[3]])
        print(testOdds.shape)
            #train_dict = {input_data: trainX, actual_labels: trainY}
            #test_dict = {input_data: testX, actual_labels: testY}
            #print("#train:", trainX.shape, "#test:", testX.shape)

        with open("./backtest.csv", "a") as f:
            f.write("予想順位,実際の順位,単勝,複勝,2連単,2連複,3連単,3連複,拡連複,流し,ボックス\n")

            win_total = 0
            place_total = 0
            exacta_total = 0
            quinella_total = 0
            wide_quinella_total = 0
            trifecta_total = 0
            trio_total = 0
            for p, a, o in zip(prediction_label, label_ar, testOdds):
                p = [str(x + 1) for x in p]
                single = str(p[0])
                double = '-'.join(p[:2])
                double_sorted = '-'.join(sorted(p[:2]))
                triple = '-'.join(p)
                triple_sorted = '-'.join(sorted(p))

                # 単勝
                win = -100
                if single == o[0][0]: win += int(o[0][1])
                win_total += win

                # 複勝
                place = -100
                if single == o[1][0]: place += int(o[1][1])
                if single == o[2][0]: place += int(o[2][1])
                place_total += place

                # 2連単
                exacta = -100
                if double == o[3][0]: exacta += int(o[3][1])
                exacta_total += exacta

                # 2連複
                quinella = int(o[4][1]) if double_sorted == o[4][0] else -100
                quinella_total += quinella

                # 拡連複
                wide_quinella = -100
                if double_sorted == o[5][0]: wide_quinella += int(o[5][1])
                if double_sorted == o[6][0]: wide_quinella += int(o[6][1])
                if double_sorted == o[7][0]: wide_quinella += int(o[7][1])
                wide_quinella_total += wide_quinella

                # 3連単
                trifecta = -100
                if triple == o[8][0]: trifecta += int(o[8][1])
                trifecta_total += trifecta

                # 3連複
                trio = -100
                if triple_sorted == o[9][0]: trio += int(o[9][1])
                trio_total += trio

                f.write("%s,%s,%d,%d,%d,%d,%d,%d,%d\n" % (
                triple, '-'.join([str(x + 1) for x in a]), win, place, exacta, quinella, wide_quinella, trifecta, trio))
            print("通算利益")
            print("単勝:", win_total)
            print("複勝:", place_total)
            print("2連単:", exacta_total)
            print("2連複:", quinella_total)
            print("拡連複:", wide_quinella_total)
            print("3連単:", trifecta_total)
            print("3連複:", trio_total)

if __name__ == "__main__":
    model_path = torch.load('./models/RankNet-Toda_201911/202001092050')
    val_file = 'Toda_201912.txt'
    val_dataset = L2RDataset(file=val_file, data_id='BOATRACE')
    race_results = get_data.RaceResults()
    date = 191201
    results = race_results.load(str(date))

    odds = race_results.odds

    print(odds)
    #BackTest(date).backtest(test_ds=val_dataset, model=model_path)


Oddsは求まったがどのように扱うか，結果と番組表は場所ごとにやるから同じ感じで