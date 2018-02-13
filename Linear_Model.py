import numpy as np 
import pandas as pd 

#随机梯度下降算法实现线性回归模型
class linear_model(object):
    def __init__(self,learning_rate=0.1,theta = None,in_put = None,output = None):
        self.learning_rate = learning_rate
        self.theta = theta
        self.in_put = in_put
        self.output = output

    def fit(self,in_put,output):
        '''
        input和ouput均为矩阵形式,公式为y = xw
        将样本依次加入计算，公式：$\theta_j := \theta_j + \alpha (y^i - h_\theta(x^i))x^i_j$
        '''
        in_put = np.matrix(in_put.values)
        output = np.matrix(output.values)
        self.in_put = in_put
        self.output = output

        if in_put.shape[0] != output.shape[0]:
            return 'In_put length should be the same with ouput length!'
        theta = np.ones([in_put.shape[1],output.shape[1]])#初始化参数theta
        for i in range(in_put.shape[0]):#遍历样本
            #按梯度下降方向进行优化
            new_theta = theta + self.learning_rate*((output[i][:]-in_put[i][:]*theta)*in_put[i][:]).T
            if self.accuracy(new_theta) < self.accuracy(theta):#用误差率来验证，指导模型优化方向，否则可能会出现绕圈的现象
                theta = new_theta
                print('Result of Sample %d:' % i,self.accuracy(theta))
            else:
                continue
        
        self.theta = theta

    def predict(self,in_put):
        '''根据参数预测output'''
        if self.theta is None:
            return 'Model not fitted!!!'
        else:
            in_put = np.matrix(in_put.values)
            y_pred = in_put * self.theta
            return y_pred

    def accuracy(self,theta):
        '''在训练过程中实时显示误差率'''
        pred = self.in_put * theta 
        error = np.abs(self.output[:][0] - pred[:][0])/self.output[:][0]
        return np.mean(error)
    
    def score(self):
        '''模型最终结果'''
        return self.accuracy(self.theta)
        

if __name__ == '__main__':
    #in_put = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9],'B':[2,3,4,5,6,7,8,9,10]})
    #output = pd.DataFrame({'C':[4,6,8,10,12,14,16,18,20]})
    data = pd.read_excel(r'E:\Modeling\HEFEI\feature_engineering\managed_data_3.xlsx')
    in_put = data.drop(['RA_NAME','AVG_PRICE','PRIMARY_SCHOOL','MIDDLE_SCHOOL'],axis=1).astype('float64')
    output = pd.DataFrame({'result':data['AVG_PRICE'].astype('float64')})#Series要转成dataframe
    clf = linear_model(learning_rate=0.00001)
    clf.fit(in_put,output)
    y_pred = clf.predict(in_put)
    print(clf.score())