import paddle

from cw_config import *


class CWNet(paddle.nn.Layer):
    """
    常务模型，输入其余9名玩家信息，输出最优英雄以及本次比赛可能的经济差
    """

    def __init__(self, is_infer: bool = False):
        super(CWNet, self).__init__()
        self.is_infer = is_infer

        # Embedding层
        self.emb_own = paddle.nn.Embedding(HERO_NUM, EMB_SIZE)
        self.emb_con = paddle.nn.Embedding(HERO_NUM, EMB_SIZE)
        # Linear层
        self.fc_own = paddle.nn.Linear(EMB_SIZE, LINEAR_SIZE)
        self.fc_con = paddle.nn.Linear(EMB_SIZE, LINEAR_SIZE)
        # GRU层
        self.gru_own = paddle.nn.GRU(LINEAR_SIZE, GRU_SIZE, direction="bidirect")
        self.gru_con = paddle.nn.GRU(LINEAR_SIZE, GRU_SIZE, direction="bidirect")
        # 合成GRU特征
        self.sub_own = paddle.nn.Linear(GRU_SIZE * 2, SUB_LINEAR_SIZE)
        self.sub_con = paddle.nn.Linear(GRU_SIZE * 2, SUB_LINEAR_SIZE)
        # 降维
        self.features_own = paddle.nn.Linear(4 * SUB_LINEAR_SIZE, FEATURES_SIZE)
        self.features_con = paddle.nn.Linear(5 * SUB_LINEAR_SIZE, FEATURES_SIZE)
        # 合成正方与反方特征
        self.sub = paddle.nn.Linear(FEATURES_SIZE * 2, FEATURES_SIZE)
        # 输出层
        self.role = paddle.nn.Linear(FEATURES_SIZE, HERO_NUM)
        self.flag = paddle.nn.Linear(FEATURES_SIZE, 1)
        self.money = paddle.nn.Linear(FEATURES_SIZE, 1)

    def forward(self, *inputs, **kwargs):
        own_input, con_input = inputs
        # Embedding层
        own = self.emb_own(own_input)
        con = self.emb_con(con_input)
        # Linear层
        own = self.fc_own(own)
        con = self.fc_con(con)
        # GRU层
        own = self.gru_own(own)[0]
        con = self.gru_con(con)[0]
        # 合成GRU特征
        own = self.sub_own(own)
        con = self.sub_con(con)
        # 降维
        own = paddle.tensor.flatten(own, start_axis=1)
        con = paddle.tensor.flatten(con, start_axis=1)
        own = self.features_own(own)
        con = self.features_con(con)
        # 合成正方与反方特征
        features = paddle.tensor.concat([own, con], axis=1)
        features = self.sub(features)
        # 输出层
        role = self.role(features)
        flag = self.flag(features)
        money = self.money(features)

        if not self.is_infer:
            return role, flag, money
        else:
            role_prob = paddle.nn.functional.softmax(role)
            role_sort = paddle.tensor.argsort(role_prob, descending=True)
            return role_prob, role_sort, flag, money


if __name__ == '__main__':
    net = CWNet()
    paddle.summary(net, input_size=[(8, 4), (8, 5)], dtypes="int64")
