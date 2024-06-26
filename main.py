from get_data import mk_dataset, mk_dataset123
from backtest import BackTest
from train import train
from dataloader import L2RDataset

def main(place, mkds=False, train_=False, three=False):
    date_trains = 190801
    date_trainl = 191031
    date_vals = 191101
    date_vall = 191130
    date_tests = 191201
    date_testl = 191231

    if mkds:
        if three:
            mk_dataset123(date_trains, date_trainl, place)
            mk_dataset123(date_vals, date_vall, place)
            mk_dataset123(date_tests, date_testl, place)
        else:
            mk_dataset(date_trains, date_trainl, place)
            mk_dataset(date_vals, date_vall, place)
            mk_dataset(date_tests, date_testl, place)

    if train_:
        if train:
            train(trains=date_trains, trainl=date_trainl, vals=date_vals, vall=date_vall, place=place, three=True)
        else:
            train(trains=date_trains, trainl=date_trainl, vals=date_vals, vall=date_vall, place=place)
    dataset = '{}_{}-{}'.format(place, date_trains, date_trainl)

    if three:
        ranknet_model_path = './models/123{}-{}-{}'.format('RankNet', dataset, place)
        lambdarank_model_path = './models/123{}-{}-{}'.format('LambdaRank', dataset, place)
        test_file = 'dataset/123{}_{}-{}.txt'.format(place, date_tests, date_testl)
    else:
        ranknet_model_path = './models/{}-{}-{}'.format('RankNet', dataset, place)
        lambdarank_model_path = './models/{}-{}-{}'.format('LambdaRank', dataset, place)
        test_file = 'dataset/{}_{}-{}.txt'.format(place, date_tests, date_testl)
    test_dataset = L2RDataset(file=test_file, data_id='BOATRACE')
    odds_json = 'dataset/{}_{}-{}.json'.format(place, date_tests, date_testl)
    BackTest().backtest(test_ds=test_dataset, model=ranknet_model_path, odds_json=odds_json, test_file=test_file, model_type='RankNet',
                        kaijo=place, three=three)
    BackTest().backtest(test_ds=test_dataset, model=lambdarank_model_path, odds_json=odds_json, test_file=test_file, model_type='LambdaRank',
                        kaijo=place, three=three)

if __name__ == '__main__':
    for place in ["多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国",
                  "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津"]:#"戸田","住之江", "尼崎", "鳴門", "丸亀", "児島", "大村", "桐生", "江戸川", "平和島",
        main(place=place, mkds=True, train_=True, three=True)
        #"びわこ"まだ