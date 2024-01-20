import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import argparse
## your model path should takes the form as
Model_Path = 'model_params.json'

## for TA's test
## you need to modify the class name to your student id.
## you also need to implement the predict function, which reads the .json file,
## calls your trained model and returns predict results as an ndarray
## the evaluation function is f1_score as follows:
'''
from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_true, y_pred, average="macro")
'''
## the test
class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)  # 汇报错误
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept  # 是否加入截距项
        self.coef_ = None

    def sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

    def get_gradient(self, X, y, coef_):
        return np.dot(X.T, (self.sigmoid(np.dot(X, coef_)) - y))

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e5, decay=0.75):  # fit的意思是拟合参数，此处使用梯度下降法
        '''
        :param X:
        :param y:
        :param lr:
        :param tol:
        :param max_iter:
        :return losses:
        '''

        if self.fit_intercept:
            X_tilde = np.c_[np.ones(X.shape[0]), X]  # c_是按列连接两个矩阵，np.ones(X.shape[0])是一个全1的矩阵，X是原矩阵
        else:
            X_tilde = X
        # Initialize coefficients
        self.coef_ = np.zeros(X_tilde.shape[1])  # coef_是系数矩阵，初始化为全0矩阵

        # List to store loss values at each iteration
        losses = []
        y_pred = self.sigmoid(np.dot(X_tilde, self.coef_))  #

        for i in range(int(max_iter)):

            loss = -y * np.dot(X_tilde, self.coef_) + np.log(1 + np.exp(np.dot(X_tilde, self.coef_)))
            loss = loss.sum()
            losses.append(loss)

            if self.penalty == 'l2':
                self.coef_ = self.coef_ - lr * (self.get_gradient(X_tilde, y, self.coef_) + self.gamma * self.coef_)
            else:
                self.coef_ = self.coef_ - lr * (
                            self.get_gradient(X_tilde, y, self.coef_) + self.gamma * np.sign(self.coef_))
            y_pred = self.sigmoid(np.dot(X_tilde, self.coef_))

            print(f'    iteration:{i},    loss:{loss:.2e}')
            if i > 1 and losses[-2] - losses[-1] < 0:
                lr = lr * decay
            if lr < tol:
                break

        return losses

    def predict(self, X):  # 在已经训练好模型后进行预测，此处使用sigmoid函数
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.

        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.

        Returns:
        - probs: numpy array of shape (n_samples,), prediction probabilities.
        """
        if self.fit_intercept:
            X_tilde = np.c_[np.ones(X.shape[0]), X]

        # Compute the linear combination of inputs and weights
        linear_output = np.dot(X_tilde, self.coef_)

        return np.where(self.sigmoid(linear_output) >= 0.5, 1, 0)

    def cal_accuracy(self, y_pred_test, y_test):
        y_pred = y_pred_test
        # 返回一个百分数，并保留4位小数，需要带百分号
        return f'accuracy:{100 * np.mean(y_pred == y_test):.4f}%'

    def cal_f1_score(self, y_pred_test, y_test):
        y_pred = y_pred_test
        TP = np.sum((y_pred == 1) & (y_test == 1))
        FP = np.sum((y_pred == 1) & (y_test == 0))
        FN = np.sum((y_pred == 0) & (y_test == 1))
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        return f'f1_score:{2 * precision * recall / (precision + recall):.4f}'


# %%
class PB21000000():
    def __init__(self):
        # self.path = 'testing_dataset.xlsx'
        self.model_path = Model_Path
        with open(self.model_path, 'r') as file:
            self.model_params = json.load(file)

        # 使用加载的参数初始化模型
        self.model = LogisticRegression(
            penalty=self.model_params['penalty'],
            gamma=self.model_params['gamma'],
            fit_intercept=self.model_params['fit_intercept']
        )
        self.model.coef_ = np.array(self.model_params['coef'])
        self.MEAN = self.model_params['MEAN']
        self.STD = self.model_params['STD']
        self.Mean_4_fillna = pd.Series(
            self.model_params['Mean_4_fillna']) if 'Mean_4_fillna' in self.model_params else None
        self.Prob_4_fillna = {key: pd.Series(value) for key, value in self.model_params[
            'Prob_4_fillna'].items()} if 'Prob_4_fillna' in self.model_params else None
        self.delete_list = self.model_params['delete_list']

    def testingset_data_processing(self, data_path):
        df = pd.read_excel(data_path)
        # 处理时间戳
        df['Time Stamp'] = df['Time Stamp'][:].apply(lambda x: x[:6] + '20' + x[8:])
        for i in range(len(self.delete_list)):
            df.drop(self.delete_list[i], axis=1, inplace=True)
        df_labels = df[['WW', 'W2']].copy()
        df_labels['WW'].fillna('无', inplace=True)
        df_labels['W2'].replace('阵性', '阵雨', inplace=True)
        df_labels['W2'].replace('雷暴，有降水或无', '雷暴，有降雨或无', inplace=True)
        df_labels['W2'].fillna('无', inplace=True)
        # 字里行间看出来的label
        df_labels['WW'].astype('str')
        df_labels['W2'].astype('str')
        df['WW'] = np.array([df_labels['WW'][i].find('雨') > 0 for i in range(df.shape[0])])
        df['W2'] = np.array([df_labels['W2'][i].find('雨') > 0 for i in range(df.shape[0])])
        df['VV'].replace('低于 0.1', 0, inplace=True)  # 目的是把低于0.1替换成0，以减少onhot编码的维度
        # 取保时间戳是datetime格式
        df['Time Stamp'] = pd.to_datetime(df['Time Stamp'], format='%d.%m.%Y %H:%M')
        df.drop(['RRR'], axis=1, inplace=True)
        # 缺失值处理的第二步
        # 处理缺失值，若是float或int类型，用均值或者中位数填充；若是str，用多项分布进行随机填充
        for col in df.columns:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                mean = self.Mean_4_fillna[col]
                df[col].fillna(mean, inplace=True)
                # df[col].fillna(df[col].median(),inplace=True) #用中位数填充
            elif df[col].dtype == 'object':

                prob = self.Prob_4_fillna[col]
                # df[col]=df[col].apply(lambda x:prob.index[np.random.multinomial(1,prob).argmax()] if pd.isnull(x) else x)
                # 用众数填充
                df[col].fillna(df[col].mode()[0], inplace=True)
        # onehot编码
        df_timestamp = df['Time Stamp'].copy()
        df.drop(['Time Stamp'], axis=1, inplace=True)

        # 为onehot编码做准备
        df_onehot = pd.get_dummies(df, dtype='float64')

        # 合并数据
        df_onehot['Time Stamp'] = df_timestamp
        df_onehot = df_onehot[['Time Stamp'] + list(df_onehot.columns[:-1])]

        # 合并小时数据成为一天的数据
        # Time Stamp里面是小时的数据，这里按照天取平均
        df_onehot['Time Stamp'] = pd.to_datetime(df_onehot['Time Stamp'], format='%Y-%m-%d %H:%M:%S')
        # 对每天的数据进行处理，取平均
        for col in df_onehot.columns[1:]:
            df_onehot_X = df_onehot.groupby(df_onehot['Time Stamp'].dt.date)[col].mean()
            df_onehot[col] = df_onehot['Time Stamp'].dt.date.map(df_onehot_X)
        return (df_onehot.iloc[:, 1:] - self.MEAN) / self.STD

    def predict(self, data_path):
        df_onehot = self.testingset_data_processing(data_path)
        return self.model.predict(df_onehot)


## for local validation
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test scripts') # 这句代码是用来解析命令行参数的
    # parser.add_argument('--test_data_path', type=str,default='/home/hyliu/ML_Project/testing_dataset.xls')
    parser.add_argument('--test_data_path', type=str, default='testing_dataset.xlsx')
    args = parser.parse_args()  # parse_args()的返回值是一个命名空间，其属性即为添加的各个参数；用法为在命令行中传入参数
    test_data = pd.read_excel(args.test_data_path)
    true = test_data['RRR'].values
    bot = PB21000000()
    pred = bot.predict(args.test_data_path)  # bot的全称是best of the best
    print(f'test_set accuracy:{bot.model.cal_accuracy(pred, true)}')
    print(f'test_set macro_f1:{bot.model.cal_f1_score(pred, true)}')