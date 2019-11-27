"""How to use this script
(1) Download race result files by RaceResults.download()
(2) Manually extract text files from lzh files
(3) Move the text files to ./data directory
(4) RaceResults.load() will parse the text files
"""

# レース結果のデータ

import numpy as np
import pandas as pd
import urllib
import os
import time
import glob
import collections
import re
#import patool

class RaceResults:
    def __init__(self):
        self.baseuri = "http://www1.mbrace.or.jp/od2/K/%s/k%s.lzh" # http://www1.mbrace.or.jp/od2/K/201612/k161201.lzh
        self.results = [] # List of (Racers, 1-2-3)
        self.id2index = None

    def download(self, start, end):
        period = pd.date_range(start, end)

        for date in period:
            # Get file from the website
            dirname = date.strftime("%Y%m")
            lzhname = date.strftime("%y%m%d")
            uri = self.baseuri % (dirname, lzhname)
            savename = "./data/results/lzh/%s.lzh" % lzhname
            if not os.path.exists(savename):
                print("Send request to", uri)
                urllib.request.urlretrieve(uri, savename)
                time.sleep(3)

            # lhaコマンドで解凍

            # The following unpack part didn't work my Windows environment...
            # Unpack lzh files
            #unpackedname = "./data/results/K%s.TXT" % lzhname
            #if not os.path.exists(unpackedname):
            #print("Unpacking", savename)
            #patool.extract_archive(savename, outdir="./data/results")

    def load(self):
        collection = []
        re_dict = {}
        re_list = []
        rank_list = []
        howmany = 0
        how = {}
        for filename in glob.glob("./data/results/k180102.txt"):
            with open(filename, "r", encoding="shift_jis") as f:
                remaining = -1
                oddscount = -1
                for line in f:
                    if line.startswith("----"):
                        howmany += 1
                        remaining = 6
                        oddscount = 9
                        positions = [None] * 6
                        top3 = [None] * 3
                        odds = []
                        rank_dict = {}
                    elif remaining > 0:
                        elems = line.replace("\u3000", "").split()
                        id = int(elems[2])
                        pos = int(elems[1]) - 1
                        positions[pos] = id
                        rank_dict[id] = 6 - int(elems[1])
                        #if howmany == 63:
                            #print("63", rank_dict)
                        p = re.compile('0[0-9]+')
                        if p.findall(elems[0]):
                            gain = 6 - int(elems[0][-1])
                        else:
                            gain = 0
                        #print(rank)
                        re_dict[pos] = gain
                        if elems[0] == "01": top3[0] = pos
                        elif elems[0] == "02": top3[1] = pos
                        elif elems[0] == "03": top3[2] = pos
                        collection.append(id)
                        remaining -= 1
                    elif oddscount > 0:
                        elems = line.split()
                        if len(elems) > 0:
                            try:
                                if oddscount == 8:
                                    odds.append((elems[1], int(elems[2]))) # 複勝1
                                    odds.append((elems[3], int(elems[4]))) # 複勝2
                                elif oddscount == 4 or oddscount == 3:
                                    odds.append((elems[0], int(elems[1]))) # 拡連複2, 3
                                else:
                                    # 単勝, 2連単, 2連複, 拡連複1, 3連単, 3連複
                                    odds.append((elems[1], int(elems[2])))
                                oddscount -= 1
                            except:
                                oddscount = -1 # ignore this
                    elif remaining == 0 and oddscount == 0:
                        valid = (len(odds) == 10)
                        for check in positions + top3:
                            if check is None:
                                print("check how many")
                                valid = False
                                break
                        ###

                        gain_ln = list(re_dict.values())
                        re_list.append(gain_ln)
                        rank_ln = list(rank_dict.values())
                        rank_list.append(rank_ln)
                        how[howmany] = rank_ln

                        if valid:
                            self.results.append((positions, top3, odds))
                        remaining = -1
                        oddscount = -1

        race_count = collections.Counter(collection)
        race_count[10000] = 0
        #print(len(re_list), re_list)
        #print(len(rank_list), rank_list)

        """
        remove = []
        for k, v in race_count.items():
            if v < 10:
                remove.append(k)
                race_count[10000] += 0 # Merge with unknown racer (=No 10000)

        for k in remove:
            race_count.pop(k)
        """
        i = 0
        for k in race_count.keys():
            race_count[k] = i
            i += 1

        self.id2index = race_count
        return how

    def get_input_length(self):
        return len(self.id2index)

    def get_input(self, id):
        return self.id2index.get(id, self.id2index[10000])


class bangumihyo():
    def __init__(self):
        self.baseuri = "http://www1.mbrace.or.jp/od2/B/%s/b%s.lzh" # http://www1.mbrace.or.jp/od2/K/201612/k161201.lzh
        self.results = [] # List of (Racers, 1-2-3)
        self.id2index = None

    def download(self, start, end):
        period = pd.date_range(start, end)

        for date in period:
            # Get file from the website
            dirname = date.strftime("%Y%m")
            lzhname = date.strftime("%y%m%d")
            uri = self.baseuri % (dirname, lzhname)
            savename = "./data/bangumi/lzh/%s.lzh" % lzhname
            if not os.path.exists(savename):
                print("Send request to", uri)
                urllib.request.urlretrieve(uri, savename)
                time.sleep(3)

    def load(self):
        collection = [] # 全ての配列
        id = 0
        bdict = {}
        for filename in glob.glob("./data/bangumi/b180102.txt"):
            with open(filename, "r", encoding="shift_jis") as f:
                remaining = -1
                oddscount = -1
                for line in f:
                    #print(line)

                    if line.startswith("----"):
                        remaining = 6
                        race = [] # 1つの配列
                        oddscount = 9
                        positions = [None] * 6
                        top3 = [None] * 3
                        odds = []
                    elif line.startswith("         "):
                        pass
                    elif line.startswith("艇"):
                        id += 1
                        pass
                    elif line.startswith("番"):
                        pass
                    #elif line.startswith("----"):
                    #    pass
                    elif remaining > 0:
                        player = [] # 1playerの配列
                        pdict = {}
                        #print(line)
                        elems = line.replace("\u3000", "").split()
                        #print(elems)
                        teiban = float(elems[0])
                        pdict[1] = teiban
                        #player.append(teiban)
                        p = re.compile('[0-9]+')
                        playerinfo = p.findall(elems[1])
                        player_num = float(playerinfo[0])
                        #player_name = elems[1][4:8]
                        player_age = float(playerinfo[1])
                        pdict[2] = player_age
                        #player.append(player_age)
                        player_weight = float(playerinfo[2])
                        pdict[3] = player_weight
                        #player.append(player_weight)
                        player_level = elems[1][-2:]
                        if player_level == "A1":
                            player_level = 0.0
                        elif player_level == "A2":
                            player_level = 1.0
                        elif player_level == "B1":
                            player_level = 2.0
                        elif player_level == "B2":
                            player_level = 3.0
                        pdict[4] = player_level
                        #player.append(player_level)
                        all_winrate = float(elems[2])
                        pdict[5] = all_winrate
                        #player.append(all_winrate)
                        all_2rate = float(elems[3])
                        pdict[6] = all_2rate
                        #player.append(all_2rate)
                        local_winrate = float(elems[4])
                        pdict[7] = local_winrate
                        #player.append(local_winrate)
                        local_2rate = float(elems[5])
                        pdict[8] = local_2rate
                        #player.append(local_2rate)
                        #motor_num = float(elems[6])
                        #pdict[9] = motor_num
                        #player.append(motor_num)
                        motor_2rate = float(elems[7])
                        pdict[9] = motor_2rate
                        #player.append(motor_2rate)
                        #boat_num = float(elems[8])
                        #pdict[11] = boat_num
                        #player.append(boat_num)
                        boat_2rate = float(elems[9])
                        pdict[10] = boat_2rate
                        #player.append(boat_2rate)
                        remaining -= 1
                        race.append(pdict)
                        if remaining == 0:
                            collection.append(race)
                            bdict[id] = race

        #print(len(collection), collection)
        return bdict #collection


"""
                        print("teiban", teiban)
                        #print("player_num", player_num)
                        print("player_age", player_age)
                        print("player_weight", player_weight)
                        print("player_level", player_level)
                        print("all_winrate", all_winrate)
                        print("all_2rate", all_2rate)
                        print("local_winrate", local_winrate)
                        print("local_2rate", local_2rate)
                        print("motor_num", motor_num)
                        print("motor_2rate", motor_2rate)
                        print("boat_num", boat_num)
                        print("boat_2rate", boat_2rate)


                        #id = int(elems[2])
                        pos = int(elems[1]) - 1
                        positions[pos] = id
                        if elems[0] == "01": top3[0] = pos
                        elif elems[0] == "02": top3[1] = pos
                        elif elems[0] == "03": top3[2] = pos
                        collection.append(id)
                        remaining -= 1


                    elif oddscount > 0:
                        elems = line.split()
                        if len(elems) > 0:
                            try:
                                if oddscount == 8:
                                    odds.append((elems[1], int(elems[2]))) # 複勝1
                                    odds.append((elems[3], int(elems[4]))) # 複勝2
                                elif oddscount == 4 or oddscount == 3:
                                    odds.append((elems[0], int(elems[1]))) # 拡連複2, 3
                                else:
                                    # 単勝, 2連単, 2連複, 拡連複1, 3連単, 3連複
                                    odds.append((elems[1], int(elems[2])))
                                oddscount -= 1
                            except:
                                oddscount = -1 # ignore this
                    elif remaining == 0 and oddscount == 0:
                        valid = (len(odds) == 10)
                        for check in positions + top3:
                            if check is None:
                                valid = False
                                break
                        if valid:
                            self.results.append((positions, top3, odds))
                        remaining = -1
                        oddscount = -1

        race_count = collections.Counter(collection)
        race_count[10000] = 0

        remove = []
        for k, v in race_count.items():
            if v < 10:
                remove.append(k)
                race_count[10000] += 0 # Merge with unknown racer (=No 10000)

        for k in remove:
            race_count.pop(k)

        i = 0
        for k in race_count.keys():
            race_count[k] = i
            i += 1

        self.id2index = race_count

    def get_input_length(self):
        return len(self.id2index)

    def get_input(self, id):
        return self.id2index.get(id, self.id2index[10000])

"""


if __name__ == "__main__":
    r = RaceResults()
    #r.download("2018-01-01", "2018-2-01")
    result = r.load()
    b = bangumihyo()
    #b.download("2018-01-01", "2018-2-01")
    bangumi = b.load()

    #print(result)
    #print(bangumi)


    for key in list(result.keys()):
        i = 0
        while i < 6:
            print(result[key][i], "qid:", key, bangumi[key][i])
            i += 1

    """
    結果と番組表を合わせたDataset 作る
    """
    """
    Player情報のDatabase作って，Datasetの特徴量に追加
    """
