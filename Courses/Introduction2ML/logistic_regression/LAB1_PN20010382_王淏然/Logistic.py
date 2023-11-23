import numpy as np
import pandas as pd

class LogisticRegression:

    def __init__(self, penalty="l2", gamma=0, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)#汇报错误
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept#是否加入截距项
        self.coef_ = None

    def sigmoid(self, x):
        return 1/(np.exp(-x)+1)

    def get_gradient(self,X_tilde, y, coef_):#求梯度

        mat = -np.array(y - self.sigmoid(np.dot(X_tilde, coef_))).reshape(-1, 1) * X_tilde
        return mat.sum(axis=0)

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7,decay=0.75):#fit的意思是拟合参数，此处使用梯度下降法
        '''
        :param X:
        :param y:
        :param lr:
        :param tol:
        :param max_iter:
        :return losses:
        '''

        if self.fit_intercept:
            X_tilde = np.c_[np.ones(X.shape[0]), X]#c_是按列连接两个矩阵，np.ones(X.shape[0])是一个全1的矩阵，X是原矩阵
        else:
            X_tilde = X
        # Initialize coefficients
        self.coef_ = np.zeros(X_tilde.shape[1])#coef_是系数矩阵，初始化为全0矩阵
        
        # List to store loss values at each iteration
        losses = []


        y_pred=self.sigmoid(np.dot(X_tilde,self.coef_))#

        for i in range(int(max_iter)):

            loss=-y*np.dot(X_tilde, self.coef_)+np.log(1+np.exp(np.dot(X_tilde,self.coef_)))

            loss=loss.sum()
            losses.append(loss)

            if self.penalty=='l2':
                self.coef_ = self.coef_ - lr * (self.get_gradient(X_tilde, y, self.coef_)+self.gamma*self.coef_)
            else:
                self.coef_ = self.coef_ - lr * (self.get_gradient(X_tilde, y, self.coef_)+self.gamma*np.sign(self.coef_))
            y_pred = self.sigmoid(np.dot(X_tilde, self.coef_))

            print(f'    iteration:{i},    loss:{loss:.2e}')
            if i>1 and losses[-2]-losses[-1]<0:
                lr=lr*decay
            if lr<tol:
                break

        return losses

    def predict(self, X):#在已经训练好模型后进行预测，此处使用sigmoid函数
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

        # TODO:                                                                        #
        # Task3: Apply the sigmoid function to compute prediction probabilities.

        return self.sigmoid(linear_output)

    def cal_accuracy(self,y_pred_test,y_test):
        y_pred=np.where(y_pred_test>=0.5,1,0)
        #返回一个百分数，并保留4位小数，需要带百分号
        return f'accuracy:{100*np.mean(y_pred==y_test):.4f}%'
