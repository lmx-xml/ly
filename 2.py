# 导入所需的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import MeanEncoder  # 用于特征编码（如均值编码）
from datetime import datetime  # 用于处理日期时间
from sklearn.preprocessing import StandardScaler  # 用于数据标准化
from sklearn.model_selection import KFold  # 用于交叉验证
import torch  # 用于构建和训练神经网络
import torch.nn as nn  # 用于神经网络模块
from torch.autograd import Variable  # 用于自动求导

# 读取训练数据和测试数据
df = pd.read_csv(r'D:\car\used_car_train_20200313.csv', sep=' ')  # 训练数据
test = pd.read_csv(r'D:\car\used_car_testB_20200421.csv', sep=' ')  # 测试数据

# 定义一个函数，用于处理日期格式
def date_process(x):
    year = int(str(x)[:4])  # 获取年份
    month = int(str(x)[4:6])  # 获取月份
    day = int(str(x)[6:8])  # 获取日期
    if month < 1:  # 如果月份小于1，则设为1
        month = 1
    date = datetime(year, month, day)  # 转换为datetime类型
    return date

# 对训练数据的日期字段进行处理
df['regDate'] = df['regDate'].apply(date_process)  # 注册日期
df['creatDate'] = df['creatDate'].apply(date_process)  # 创建日期

# 从日期中提取出年、月、日信息
df['regDate_year'] = df['regDate'].dt.year
df['regDate_month'] = df['regDate'].dt.month
df['regDate_day'] = df['regDate'].dt.day
df['creatDate_year'] = df['creatDate'].dt.year
df['creatDate_month'] = df['creatDate'].dt.month
df['creatDate_day'] = df['creatDate'].dt.day

# 计算汽车的使用年限
df['car_age_day'] = (df['creatDate'] - df['regDate']).dt.days
df['car_age_year'] = round(df['car_age_day'] / 365, 1)  # 以年为单位

# 数据清洗
df['notRepairedDamage'] = df['notRepairedDamage'].replace('-', 0.0).astype('float64')  # 填补缺失值并转换类型
df['power'][df['power'] > 600] = 600  # 修正功率异常值
df['power'][df['power'] < 1] = 1  # 修正功率异常值
df['v_13'][df['v_13'] > 6] = 6  # 修正v_13异常值
df['v_14'][df['v_14'] > 4] = 4  # 修正v_14异常值
df['fuelType'] = df['fuelType'].fillna(0)  # 填补缺失值
df['gearbox'] = df['gearbox'].fillna(0)  # 填补缺失值
df['bodyType'] = df['bodyType'].fillna(0)  # 填补缺失值
df['model'] = df['model'].fillna(0)  # 填补缺失值

# 对测试数据做类似的处理
test['regDate'] = test['regDate'].apply(date_process)
test['creatDate'] = test['creatDate'].apply(date_process)
test['regDate_year'] = test['regDate'].dt.year
test['regDate_month'] = test['regDate'].dt.month
test['regDate_day'] = test['regDate'].dt.day
test['creatDate_year'] = test['creatDate'].dt.year
test['creatDate_month'] = test['creatDate'].dt.month
test['creatDate_day'] = test['creatDate'].dt.day
test['car_age_day'] = (test['creatDate'] - test['regDate']).dt.days
test['car_age_year'] = round(test['car_age_day'] / 365, 1)

test['notRepairedDamage'] = test['notRepairedDamage'].replace('-', 0).astype('float64')
test['power'][test['power'] > 600] = 600
test['power'][test['power'] < 1] = 1
test['v_13'][test['v_13'] > 6] = 6
test['v_14'][test['v_14'] > 4] = 4
test['fuelType'] = test['fuelType'].fillna(0)
test['gearbox'] = test['gearbox'].fillna(0)
test['bodyType'] = test['bodyType'].fillna(0)
test['model'] = test['model'].fillna(0)

# 创建新特征（通过特征间的交互作用生成新特征）
num_cols = [0, 2, 3, 6, 8, 10, 12, 14]  # 选择数值特征的列索引
for index, value in enumerate(num_cols):
    for j in num_cols[index + 1:]:
        # 生成特征之间的交互特征
        df['new' + str(value) + '*' + str(j)] = df['v_' + str(value)] * df['v_' + str(j)]  # 特征相乘
        df['new' + str(value) + '+' + str(j)] = df['v_' + str(value)] + df['v_' + str(j)]  # 特征相加
        df['new' + str(value) + '-' + str(j)] = df['v_' + str(value)] - df['v_' + str(j)]  # 特征相减
        test['new' + str(value) + '*' + str(j)] = test['v_' + str(value)] * test['v_' + str(j)]
        test['new' + str(value) + '+' + str(j)] = test['v_' + str(value)] + test['v_' + str(j)]
        test['new' + str(value) + '-' + str(j)] = test['v_' + str(value)] - test['v_' + str(j)]

# 使用“汽车年龄”与其他特征的交互作用
for i in range(15):
    df['new' + str(i) + '*year'] = df['v_' + str(i)] * df['car_age_year']
    test['new' + str(i) + '*year'] = test['v_' + str(i)] * test['car_age_year']

# 选择需要用于模型训练的特征列
num_cols1 = [3, 5, 1, 11]  # 选择另一组特征列
for index, value in enumerate(num_cols1):
    for j in num_cols1[index + 1:]:
        df['new' + str(value) + '-' + str(j)] = df['v_' + str(value)] - df['v_' + str(j)]  # 计算特征之间的差值
        test['new' + str(value) + '-' + str(j)] = test['v_' + str(value)] - test['v_' + str(j)]

# 准备训练数据和测试数据
X = df.drop(columns=['price', 'SaleID', 'seller', 'offerType', 'name', 'creatDate', 'regionCode', 'regDate'])  # 特征集
test = test.drop(columns=['SaleID', 'seller', 'offerType', 'name', 'creatDate', 'regionCode', 'regDate'])  # 测试集
Y = df['price']  # 目标变量

# 均值编码
class_list = ['model', 'brand', 'power', 'v_0', 'v_3', 'v_8', 'v_12']  # 选择用于均值编码的类别特征
MeanEnocodeFeature = class_list  # 定义均值编码的特征
ME = MeanEncoder.MeanEncoder(MeanEnocodeFeature, target_type='regression')  # 初始化均值编码器
X = ME.fit_transform(X, Y)  # 训练均值编码器并转换训练数据
test = ME.transform(test)  # 转换测试数据

# 合并训练集和测试集，进行标准化
df_concat = pd.concat([X, test], ignore_index=True)  # 合并训练和测试数据
df_concat = StandardScaler().fit_transform(df_concat)  # 对数据进行标准化
X1 = df_concat[:150000]  # 训练集数据
test1 = df_concat[150000:]  # 测试集数据

# 定义神经网络模型结构
input_size = 143  # 输入层大小
hidden_size = 320  # 隐藏层大小
num_classes = 1  # 输出层大小（回归问题）
batch_size = 2048
learning_rate = 0.05
x = torch.tensor(X1, dtype=torch.float32)
y = torch.FloatTensor(Y.to_numpy())
y = Variable(y.view(-1, 1))
test = torch.tensor(test1, dtype=torch.float32)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


net = Net(input_size, hidden_size, num_classes)
print(net)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

result = []
mean_score = 0
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=2022)
for train_index, test_index in kf.split(x):
    x_train = x[train_index]
    y_train = y[train_index]
    x_test = x[test_index]
    y_test = y[test_index]
    for i in range(2000):
        for start in range(0, len(x_train), batch_size):
            end = start + batch_size if start + batch_size < len(x_train) else len(x_train)
            xx = x_train[start:end]
            yy = y_train[start:end]
            outputs = net(xx)
            loss = criterion(outputs, yy)
            net.zero_grad()
            loss.backward()
            optimizer.step()
    y_pred = net.forward(x_test)
    loss1 = criterion(y_test, y_pred)
    mean_score += loss1.item() / n_folds
    print('验证集loss:{}'.format(loss1.item()))
    test_pred = net.forward(test)
    result.append(test_pred)
# 模型评估
print('mean 验证集Auc:{}'.format(mean_score))
cat_pre = sum(result) / n_folds
cat_pre = cat_pre.detach().numpy()
ret = pd.DataFrame(cat_pre, columns=['price'])
ret.to_csv('D:\car\predictions.csv')