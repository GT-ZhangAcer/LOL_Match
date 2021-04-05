import pickle
import random

import numpy as np
from paddle.io import Dataset

from cw_config import *
from script_tools.label_csv2pkl import LOLData


class LOLDataset(Dataset):
    OWN_OTHER_PAYER_INDEX = [i for i in range(5)]

    def __init__(self,
                 is_eval: bool = False):
        super(LOLDataset, self).__init__()
        if is_eval is False:
            with open(DATA_CACHE_FILE_PATH, "rb") as f:
                self.data = pickle.load(f)
                self.data_index = [i for i in range(len(self.data)) if i % 5 != 4]
        else:
            with open(DATA_CACHE_FILE_PATH, "rb") as f:
                self.data = pickle.load(f)
                self.data_index = [i for i in range(len(self.data)) if i % 5 == 4]

    def __getitem__(self, index):
        hero, flag, money_radio = self.data[index]

        # 是否进行队伍反转
        judge = random.uniform(0., 1.)
        if judge < REVERSAL_TEAM_PROB:
            flag *= -1
            money_radio *= -1
            own_hero = hero[8]
            own_team_hero = hero[5:8] + [hero[9]]
            con_team_hero = hero[:5]
        else:
            own_hero = hero[0]
            own_team_hero = hero[1:5]
            con_team_hero = hero[5:]

        # 模拟英雄选择情况
        hero_select_none_num = random.randint(0, RANDOM_HERO_NONE_SELECT_MAX_NUM_EACH_TEAM)
        own_drop_list = random.sample(self.OWN_OTHER_PAYER_INDEX, hero_select_none_num)
        con_drop_list = random.sample(self.OWN_OTHER_PAYER_INDEX, hero_select_none_num)
        # 假装这部分人没有选择英雄
        for drop_index in own_drop_list:
            if drop_index == 1 or 4:
                drop_index = 3
            own_team_hero[drop_index] = 0
        for drop_index in con_drop_list:
            if drop_index == 3:
                drop_index = 4
            con_team_hero[drop_index] = 0

        # 是否打乱英雄选择顺序
        if RANDOM_HERO_SEQUENCE:
            random.shuffle(own_team_hero)
            random.shuffle(con_team_hero)

        own_hero = np.array(own_hero).astype("int64")
        own_team_hero = np.array(own_team_hero).astype("int64")
        con_team_hero = np.array(con_team_hero).astype("int64")
        money_radio = np.array(money_radio).astype("float32")
        flag = np.array(flag).astype("float32")

        return own_team_hero, con_team_hero, own_hero, flag, money_radio

    def __len__(self):
        return len(self.data_index)


if __name__ == '__main__':
    tmp = LOLDataset()
    for tmp_sample in tmp:
        pass
