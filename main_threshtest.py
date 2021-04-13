from get_data import mk_dataset, mk_dataset123
from backtest_threshold_check_sanrentan import BackTest
from train import train
from dataloader import L2RDataset

def main(place, mkds=False, train_=False, three=False, mxepoch=20):
    date_trains = 180101
    date_trainl = 190531
    date_vals = 190601
    date_vall = 190730
    date_tests = 190801
    date_testl = 191031
    # date_trains = 180101
    # date_trainl = 180131
    # date_vals = 190601
    # date_vall = 190730
    # date_tests = 190801
    # date_testl = 191031

    if mkds:
        if three: # ４位以下を同一視
            mk_dataset123(date_trains, date_trainl, place)
            mk_dataset123(date_vals, date_vall, place)
            mk_dataset123(date_tests, date_testl, place)
        else:
            mk_dataset(date_trains, date_trainl, place)
            mk_dataset(date_vals, date_vall, place)
            mk_dataset(date_tests, date_testl, place)

    if train_:
        train(trains=date_trains, trainl=date_trainl, vals=date_vals, vall=date_vall, place=place, three=three, mxepoch=mxepoch)

    dataset = '{}_{}-{}'.format(place, date_trains, date_trainl)

    if three:
        ranknet_model_path = './models/123{}-{}-{}'.format('RankNet', dataset, place)
        lambdarank_model_path = './models/123{}-{}-{}'.format('LambdaRank', dataset, place)
        test_file = 'dataset/123{}_{}-{}.txt'.format(place, date_tests, date_testl)
    else:
        ranknet_model_path = './models/{}-{}-{}'.format('RankNet', dataset, place)
        lambdarank_model_path = './models/{}-{}-{}'.format('LambdaRank', dataset, place)
        listnet_model_path = './models/{}-{}-{}'.format('ListNet', dataset, place)
        test_file = 'dataset/{}_{}-{}.txt'.format(place, date_tests, date_testl)
    test_dataset = L2RDataset(file=test_file, data_id='BOATRACE')
    odds_json = 'dataset/{}_{}-{}.json'.format(place, date_tests, date_testl)
    BackTest().backtest(test_ds=test_dataset, model=ranknet_model_path, odds_json=odds_json, test_file=test_file, model_type='RankNet',
                        kaijo=place, three=three)
    BackTest().backtest(test_ds=test_dataset, model=lambdarank_model_path, odds_json=odds_json, test_file=test_file, model_type='LambdaRank',
                        kaijo=place, three=three)
    # BackTest().backtest(test_ds=test_dataset, model=listnet_model_path, odds_json=odds_json, test_file=test_file,
    #                     model_type='ListNet',
    #                     kaijo=place, three=three)

if __name__ == '__main__':
    for place in ["住之江", "尼崎", "鳴門", "丸亀", "児島", "大村", "桐生", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国",
                  "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津"]:#
    # place = '平和島'
        main(place=place, mkds=False, train_=False, three=False, mxepoch=20)
        #"びわこ"まだ
