import pickle

LABEL_CSV_PATH = "../label.csv"
DATA_PATH = "../data.csv"
OUT_PATH = "../data.cache"


class LOLData:
    def __init__(self):
        self.label_encode = dict()
        self.label_decode = {"0": "暂无推荐英雄"}
        self.data_hero = list()
        self.data_flag = list()
        self.data_money_radio = list()

        self.__make_label_dict()
        self.__make_data()
        pass

    def __make_label_dict(self):
        with open(LABEL_CSV_PATH, "r") as f:
            for line in f.readlines():
                index, hero = line.rstrip("\n").split(",")
                self.label_encode[hero] = index
                self.label_decode[index] = hero

    def __make_data(self):
        with open(DATA_PATH, "r") as f:
            for line in f.readlines():
                sample = line.rstrip("\n").split(",")
                hero = [self.label_encode[i] for i in sample[:10]]
                money_own = sample[10]
                money_con = sample[11]
                flag = 1 if sample[12] == "胜利" else -1
                # 平均经济差
                money_radio = (int(money_own) - int(money_con)) / ((int(money_own) + int(money_con)) * 0.5)
                self.data_hero.append(hero)
                self.data_flag.append(flag)
                self.data_money_radio.append(money_radio)

    def __getitem__(self, index):
        return self.data_hero[index], self.data_flag[index], self.data_money_radio[index]

    def __len__(self):
        return len(self.data_flag)


if __name__ == '__main__':
    lol_data = LOLData()
    with open(OUT_PATH, "wb") as file:
        pickle.dump(lol_data, file)
    print("生成完毕")
