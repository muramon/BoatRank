from get_data import RaceResults, bangumihyo

if __name__ == "__main__":
    r = RaceResults()
    r.download("2018-02-02", "2019-07-31")

    b = bangumihyo()
    b.download("2018-02-02", "2019-07-31")