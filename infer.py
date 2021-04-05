import pickle

import paddle
import numpy as np

from cw_config import *
from cw_net import CWNet
from reader import LOLDataset
from script_tools.label_csv2pkl import LOLData


class LOLInfer:
    def __init__(self):
        with open(DATA_CACHE_FILE_PATH, "rb") as f:
            data = pickle.load(f)
            self.label_encode = data.label_encode
            self.label_decode = data.label_decode
        # 定义输入格式
        input_field = [paddle.static.InputSpec(shape=[4], dtype="int64", name="own_team_hero"),
                       paddle.static.InputSpec(shape=[5], dtype="int64", name="cow_team_hero")]

        # 实例化模型
        self.model = paddle.Model(CWNet(is_infer=True), input_field)
        self.model.load(F_MODEL_PATH)
        self.model.prepare()

    def format_data(self, own, con):
        own = [self.label_encode[i] for i in own]
        con = [self.label_encode[i] for i in con]
        own = np.array([own]).astype("int64")
        con = np.array([con]).astype("int64")
        return own, con

    def infer(self, own_data, con_data):
        result = self.model.predict_batch(self.format_data(own_data, con_data))
        flag = f"未选择英雄时的胜率预估{float((result[2][0] + 1) * 0.5) * 100:.4f}%\n"
        money = f"未选择英雄时经济差占比估计{float((result[3][0] + 1) * 0.5) * 100:.4f}%\n"
        top1 = f"Top1英雄：{self.label_decode[str(int(result[1][0][0]))]}，推荐指数{result[0][0][result[1][0][0]] * 100:.4f}%\n"
        top2 = f"Top2英雄：{self.label_decode[str(int(result[1][0][1]))]}，推荐指数{result[0][0][result[1][0][1]] * 100:.4f}%\n"
        top3 = f"Top3英雄：{self.label_decode[str(int(result[1][0][2]))]}，推荐指数{result[0][0][result[1][0][2]] * 100:.4f}%\n"
        print(flag + money + top1 + top2 + top3)


if __name__ == '__main__':
    # 加载模拟数据A进行测试 真实对局情况如下
    # 虚空之女, 荒漠屠夫, 流浪法师, 恶魔小丑, 魔法猫咪, 亡灵战神, 龙血武姬, 铸星龙王, 荣耀行刑官, 正义巨像, 44220, 40205, 胜利
    own_A = ["荒漠屠夫", "流浪法师", "恶魔小丑", "魔法猫咪"]
    con_A = ["亡灵战神", "龙血武姬", "铸星龙王", "荣耀行刑官", "正义巨像"]
    # 加载模拟数据B进行测试 真实对局情况如下
    # 暴走萝莉, 海洋之灾, 德邦总管, 邪恶小法师, 魂锁典狱长, 齐天大圣, 法外狂徒, 时间刺客, 探险家, 虚空之眼, 45511, 35760, 胜利
    own_B = ["海洋之灾", "德邦总管", "邪恶小法师", "魂锁典狱长"]
    con_B = ["齐天大圣", "法外狂徒", "时间刺客", "探险家", "虚空之眼"]
    # 加载模拟数据C进行测试 真实对局情况如下
    # 虚空之女,荒漠屠夫,狂野女猎手,封魔剑魂,圣锤之毅,诺克萨斯之手,永猎双子,正义巨像,沙漠玫瑰,魂锁典狱长,54713,59059,失败
    own_C = ["荒漠屠夫", "狂野女猎手", "封魔剑魂", "圣锤之毅"]
    con_C = ["诺克萨斯之手", "永猎双子", "正义巨像", "沙漠玫瑰", "魂锁典狱长"]

    # 加载预测器
    infer = LOLInfer()
    # 执行预测
    infer.infer(own_A, con_A)
    infer.infer(own_B, con_B)
    infer.infer(own_C, con_C)
