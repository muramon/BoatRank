"""How to use this script
(1) Download race result files by RaceResults.download()
(2) Manually extract text files from lzh files
(3) Move the text files to ./data directory
(4) RaceResults.load() will parse the text files
"""

# レース結果のデータ

import numpy as np
import pandas as pd
import urllib.request
import os
import time
import glob
import collections
import re
import json
from data.bangumi.kaito import kaito_bangumi
from data.results.kaito import kaito_result
#import patool

class RaceResults:
    def __init__(self):
        self.baseuri = "http://www1.mbrace.or.jp/od2/K/%s/k%s.lzh" # http://www1.mbrace.or.jp/od2/K/201612/k161201.lzh
        self.results = [] # List of (Racers, 1-2-3)
        self.id2index = None
        self.odds = {}

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
        kaito_result(start, end)

            # lhaコマンドで解凍

            # The following unpack part didn't work my Windows environment...
            # Unpack lzh files
            #unpackedname = "./data/results/K%s.TXT" % lzhname
            #if not os.path.exists(unpackedname):
            #print("Unpacking", savename)
            #patool.extract_archive(savename, outdir="./data/results")

    def load(self, date, odds_dict):
        collection = []
        re_dict = {}
        re_list = []
        rank_list = []
        howmany = 0
        how = {}
        #fj = open('Toda_201912.json', 'a')
        for filename in glob.glob("./data/results/k%s.txt" % date):
            with open(filename, "r", encoding="shift_jis") as f:
                remaining = -1
                oddscount = -1
                for line in f:
                    if line.startswith("   第"):
                        #print(line)
                        p = re.compile('ボートレース...')
                        place = p.search(line)
                        #race_track = place.group()[6:9].replace('\u3000', '')
                        #if not race_track in self.odds:
                        #    self.odds[race_track] = []
                    elif line.startswith("----"):
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
                            odds_key = str(date) + str(howmany) # str(date) + str(key)
                            #self.odds[odds_key] = odds
                            odds_dict[odds_key] = odds
                            #json.dump(self.odds, fj)
                            #self.results.append((positions, top3, odds))
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
        return how, odds_dict

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
        # kaito
        # kaito_result("20{}-{}-{}".format(str(dates)[0:2], str(dates)[2:4], str(dates)[4:6]), "20{}-{}-{}".format(str(datel)[0:2], str(datel)[2:4], str(datel)[4:6]))
        # kaito_bangumi("20{}-{}-{}".format(str(dates)[0:2], str(dates)[2:4], str(dates)[4:6]), "20{}-{}-{}".format(str(datel)[0:2], str(datel)[2:4], str(datel)[4:6]))
        kaito_bangumi(start, end)

    def load(self, date):
        collection = [] # 全ての配列
        id = 0
        bdict = {}
        for filename in glob.glob("./data/bangumi/b%s.txt" % date):
            with open(filename, "r", encoding="shift_jis") as f:
                remaining = -1
                oddscount = -1
                for line in f:
                    #print(line)
                    if line.startswith("   第"):
                        #print(line)
                        p = re.compile('ボートレース...')
                        place = p.search(line)
                        race_track = place.group()[6:9].replace('\u3000', '')
                    elif line.startswith("----"):
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
                        pdict[race_track] = {}
                        #print(line)
                        elems = line.replace("\u3000", "").split()
                        teiban = float(elems[0])
                        pdict[race_track][1] = teiban
                        #player.append(teiban)
                        p = re.compile('[0-9]+')
                        playerinfo = p.findall(elems[1])
                        player_num = float(playerinfo[0])
                        #player_name = elems[1][4:8]
                        player_age = float(playerinfo[1])
                        pdict[race_track][2] = player_age
                        #player.append(player_age)
                        player_weight = float(playerinfo[2])
                        pdict[race_track][3] = player_weight
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
                        pdict[race_track][4] = player_level
                        #player.append(player_level)
                        all_winrate = float(elems[2])
                        pdict[race_track][5] = all_winrate
                        #player.append(all_winrate)
                        try:
                            all_2rate = float(elems[3])
                        except(ValueError):
                            print(date, elems)
                        pdict[race_track][6] = all_2rate
                        #player.append(all_2rate)
                        try:
                            local_winrate = float(elems[4])
                        except(ValueError):
                            print(date, elems)
                        try:
                            pdict[race_track][7] = local_winrate
                        except(ValueError):
                            print(date, elems)
                        except(UnboundLocalError):
                            print("UnboundLocalError")
                        #player.append(local_winrate)
                        local_2rate = float(elems[5])
                        pdict[race_track][8] = local_2rate
                        #player.append(local_2rate)
                        #motor_num = float(elems[6])
                        #pdict[9] = motor_num
                        #player.append(motor_num)
                        try:
                            motor_2rate = float(elems[7])
                        except(ValueError):
                            pdict[race_track][9] = motor_2rate
                        #player.append(motor_2rate)
                        #boat_num = float(elems[8])
                        #pdict[11] = boat_num
                        #player.append(boat_num)
                        try:
                            boat_2rate = float(elems[9])
                        except(ValueError):
                            boat_2rate = 0
                        except(IndexError):
                            boat_2rate = 0
                        pdict[race_track][10] = boat_2rate
                        #player.append(boat_2rate)
                        remaining -= 1
                        race.append(pdict)
                        if remaining == 0:
                            collection.append(race)
                            bdict[id] = race

        #print(len(collection), collection)
        return bdict #collection


class RaceResults123:
    def __init__(self):
        self.baseuri = "http://www1.mbrace.or.jp/od2/K/%s/k%s.lzh" # http://www1.mbrace.or.jp/od2/K/201612/k161201.lzh
        self.results = [] # List of (Racers, 1-2-3)
        self.id2index = None
        self.odds = {}

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

    def load(self, date, odds_dict):
        collection = []
        re_dict = {}
        re_list = []
        rank_list = []
        howmany = 0
        how = {}
        #fj = open('Toda_201912.json', 'a')
        for filename in glob.glob("./data/results/k%s.txt" % date):
            with open(filename, "r", encoding="shift_jis") as f:
                remaining = -1
                oddscount = -1
                for line in f:
                    if line.startswith("   第"):
                        #print(line)
                        p = re.compile('ボートレース...')
                        place = p.search(line)
                        #race_track = place.group()[6:9].replace('\u3000', '')
                        #if not race_track in self.odds:
                        #    self.odds[race_track] = []
                    elif line.startswith("----"):
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
                        if int(elems[1]) <= 3:
                            rank_dict[id] = 4 - int(elems[1])
                        else:
                            rank_dict[id] = 0
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
                            odds_key = str(date) + str(howmany) # str(date) + str(key)
                            #self.odds[odds_key] = odds
                            odds_dict[odds_key] = odds
                            #json.dump(self.odds, fj)
                            #self.results.append((positions, top3, odds))
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
        return how, odds_dict

    def get_input_length(self):
        return len(self.id2index)

    def get_input(self, id):
        return self.id2index.get(id, self.id2index[10000])



"""
class boatfan():
    def __init__(self):
        self.fan = []

    def load(self):
"""


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

def mk_dataset(dates, datel, place, download=False):
    r = RaceResults()
    b = bangumihyo()
    if download:
        r.download("20{}-{}-{}".format(str(dates)[0:2], str(dates)[2:4], str(dates)[4:6]), "20{}-{}-{}".format(str(datel)[0:2], str(datel)[2:4], str(datel)[4:6]))
        b.download("20{}-{}-{}".format(str(dates)[0:2], str(dates)[2:4], str(dates)[4:6]), "20{}-{}-{}".format(str(datel)[0:2], str(datel)[2:4], str(datel)[4:6]))
    file = open('dataset/{}_{}-{}.txt'.format(place, dates, datel), 'w')
    date = dates
    fj = open('dataset/{}_{}-{}.json'.format(place, dates, datel), 'w')
    odds_dict = {}
    while date <= datel:
        date_txt = "k{}.txt".format(date)
        if date_txt in os.listdir("./data/results"):
            result, odds_dict = r.load(str(date), odds_dict)
            bangumi = b.load(str(date))
            for key in list(result.keys()):
                i = 0
                while i < 6:
                    try:
                        if bangumi[key][i][place]:
                            # print(result[key][i], "qid:", str(191101) + str(key), bangumi[key][i][place])
                            file.write(str(result[key][i]))
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
        date += 1
    # print(odds_dict)
    json.dump(odds_dict, fj, indent=4)
    file.close()


def mk_dataset123(dates, datel, place):
    r = RaceResults123()
    b = bangumihyo()
    file = open('dataset/123{}_{}-{}.txt'.format(place, dates, datel), 'w')
    date = dates
    fj = open('dataset/{}_{}-{}.json'.format(place, dates, datel), 'w')
    odds_dict = {}
    while date <= datel:
        date_txt = "k{}.txt".format(date)
        if date_txt in os.listdir("./data/results"):
            result, odds_dict = r.load(str(date), odds_dict)
            bangumi = b.load(str(date))
            for key in list(result.keys()):
                i = 0
                while i < 6:
                    try:
                        if bangumi[key][i][place]:
                            # print(result[key][i], "qid:", str(191101) + str(key), bangumi[key][i][place])
                            file.write(str(result[key][i]))
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
        date += 1
    # print(odds_dict)
    json.dump(odds_dict, fj, indent=4)
    file.close()


if __name__ == "__main__":
    dates=210414
    datel=210415
    place='平和島'
    mk_dataset(dates, datel, place, download=True)
    # print("20{}-{}-{}".format(str(dates)[0:2], str(dates)[2:4], str(dates)[4:6]), "20{}-{}-{}".format(str(datel)[0:2], str(datel)[2:4], str(datel)[4:6]))

