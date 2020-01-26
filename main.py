from get_data import mk_dataset
from backtest import BackTest
from train import train
from dataloader import L2RDataset

def main(place):
    date_trains = 190801
    date_trainl = 191031
    date_vals = 191101
    date_vall = 191130
    date_tests = 191201
    date_testl = 191231

    mk_dataset(date_trains, date_trainl, place)
    mk_dataset(date_vals, date_vall, place)
    mk_dataset(date_tests, date_testl, place)

    train(trains=date_trains, trainl=date_trainl, vals=date_vals, vall=date_vall, place=place)
    dataset = '{}_{}-{}'.format(place, date_trains, date_trainl)

    ranknet_model_path = './models/{}-{}-{}'.format('RankNet', dataset, place)
    lambdarank_model_path = './models/{}-{}-{}'.format('LambdaRank', dataset, place)
    test_file = 'dataset/{}_{}-{}.txt'.format(place, date_tests, date_testl)
    test_dataset = L2RDataset(file=test_file, data_id='BOATRACE')
    odds_json = 'dataset/{}_{}-{}.json'.format(place, date_tests, date_testl)
    BackTest().backtest(test_ds=test_dataset, model=ranknet_model_path, odds_json=odds_json, test_file=test_file, model_type='RankNet',
                        kaijo=place, Shuffle=False)
    BackTest().backtest(test_ds=test_dataset, model=lambdarank_model_path, odds_json=odds_json, test_file=test_file, model_type='LambdaRank',
                        kaijo=place, Shuffle=False)

if __name__ == '__main__':
    for place in ["住之江", "尼崎", "鳴門", "丸亀", "児島",
                  "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津"]:# "戸田", "大村" "桐生", "江戸川", "平和島","多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国",
        main(place=place)
        #"びわこ"まだ