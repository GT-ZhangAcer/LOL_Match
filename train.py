import paddle

from cw_config import *
from cw_net import CWNet
from reader import LOLDataset
from script_tools.label_csv2pkl import LOLData


class CWLoss(paddle.nn.Layer):
    def __init__(self):
        super(CWLoss, self).__init__()

    def forward(self, *inputs, **kwargs):
        role, flag_f, money, own_hero, flag, money_radio = inputs
        select_loss = paddle.nn.functional.cross_entropy(role, own_hero)
        money_loss = paddle.nn.functional.mse_loss(money, money_radio)
        flag_loss = paddle.nn.functional.mse_loss(flag_f, flag)
        sub_loss = select_loss + money_loss + flag_loss
        return sub_loss


# 定义输入格式
input_field = [paddle.static.InputSpec(shape=[4], dtype="int64", name="own_team_hero"),
               paddle.static.InputSpec(shape=[5], dtype="int64", name="cow_team_hero")]
label_field = [paddle.static.InputSpec(shape=[1], dtype="int64", name="own_hero"),
               paddle.static.InputSpec(shape=[1], dtype="float32", name="flag"),
               paddle.static.InputSpec(shape=[1], dtype="float32", name="money_radio")]

# 实例化模型
model = paddle.Model(CWNet(), input_field, label_field)

# 加载数据集
train_data = LOLDataset()
eval_data = LOLDataset(is_eval=True)

# 定义优化器
opt = paddle.optimizer.Adam(learning_rate=0.0001, parameters=model.parameters())
loss = CWLoss()
model.prepare(opt, loss)
# 开始训练
model.fit(train_data=train_data,
          eval_data=eval_data,
          batch_size=BATCH_SIZE,
          num_workers=NUM_WORKERS,
          epochs=EPOCH,
          log_freq=200,
          save_dir=MODEL_PATH,
          save_freq=50)
