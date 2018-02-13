import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#随机梯度下降算法实现线性回归模型
class linear_model(object):
    def __init__(self,learning_rate=0.1,theta = None,in_put = None,output = None):
        self.learning_rate = learning_rate
        self.theta = theta
        self.in_put = in_put
        self.output = output

    def divide(self,in_put,output):
        '''
        将数据集分为训练集和测试集，直接划分导致数据集差异较大，最好选择随机抽样
        '''
        length = in_put.shape[0]
        #依次为x_train,y_train,x_test,y_test
        return in_put[:int((2*length)/3)],output[:int((2*length)/3)],in_put[int((2*length)/3):],output[int((2*length)/3):]

    def fit(self,in_put,output):
        '''
        input和ouput均为矩阵形式,公式为y = xw
        将样本依次加入计算，公式：$\theta_j := \theta_j + \alpha (y^i - h_\theta(x^i))x^i_j$
        '''
        in_put = np.matrix(in_put.values)
        output = np.matrix(output.values)
        self.in_put = in_put
        self.output = output

        x_train,y_train,x_test,y_test = self.divide(in_put,output)

        if in_put.shape[0] != output.shape[0]:
            raise Exception('In_put length should be the same with ouput length!')
        theta = np.ones([in_put.shape[1],output.shape[1]])#初始化参数theta
        train_metrics = []#记录训练误差变化
        test_metrics = []#记录测试误差变化
        
        for i in range(x_train.shape[0]):#遍历样本
            #按梯度下降方向进行优化
            new_theta = theta + self.learning_rate*((y_train[i][:]-x_train[i][:]*theta)*x_train[i][:]).T
            if self.accuracy(new_theta,in_put,output) < self.accuracy(theta,in_put,output): #用误差率来验证，指导模型优化方向，否则可能会出现绕圈的现象
                theta = new_theta
                train_metric = self.accuracy(theta,x_train,y_train)
                train_metrics.append(train_metric)
                print('Result of Sample %d:' % i,train_metric)
                test_metric = self.accuracy(theta,x_test,y_test)
                test_metrics.append(test_metric)
                print('test:',test_metric)
            else:
                #train_metrics.append(train_metric)#如果没有更好的效果就记录下上一次的结果
                #test_metrics.append(test_metric)
                train_metrics.append(self.accuracy(new_theta,x_train,y_train))
                test_metrics.append(self.accuracy(new_theta,x_test,y_test))
                continue
        
        self.theta = theta
        return train_metrics,test_metrics

    def predict(self,in_put):
        '''根据参数预测output'''
        if self.theta is None:
            raise Exception('Model not fitted yet !!!')
        else:
            in_put = np.matrix(in_put.values)
            y_pred = in_put * self.theta
            return y_pred

    def accuracy(self,theta,in_put,output):
        '''在训练过程中实时显示误差率'''
        pred = in_put * theta 
        error = np.abs(output[:][0] - pred[:][0])/output[:][0]
        return np.mean(error)
    
    def score(self):
        '''模型最终结果'''
        return self.accuracy(self.theta,self.in_put,self.output)
        

if __name__ == '__main__':
    #in_put = pd.DataFrame({'A':[1,2,3,4,5,6,7,8,9],'B':[2,3,4,5,6,7,8,9,10]})
    #output = pd.DataFrame({'C':[4,6,8,10,12,14,16,18,20]})
    
    #训练
    data = pd.read_excel(r'E:\Modeling\HEFEI\feature_engineering\managed_data_3.xlsx')
    train = data[:]
    #test = data[1500:]
    in_put = train.drop(['RA_NAME','AVG_PRICE','PRIMARY_SCHOOL','MIDDLE_SCHOOL'],axis=1).astype('float64')
    output = pd.DataFrame({'result':train['AVG_PRICE'].astype('float64')})#Series要转成dataframe
    clf = linear_model(learning_rate=0.00001)
    train_metrics,test_metrics = clf.fit(in_put,output)
    print(clf.score())

    #绘制学习曲线
    plt.plot(train_metrics,color = 'g',label = 'train')#绿色是训练集
    plt.plot(test_metrics,color='r',label = 'test')#红色是测试集
    plt.legend()
    plt.show()
    
    #测试
    '''
    in_put_test = test.drop(['RA_NAME','AVG_PRICE','PRIMARY_SCHOOL','MIDDLE_SCHOOL'],axis=1).astype('float64')
    output_test = pd.DataFrame({'result':test['AVG_PRICE'].astype('float64')})#Series要转成dataframe
    output_pred = clf.predict(in_put_test)  
    output_test = np.matrix(output_test)  
    error = np.abs(output_test[:][0] -  output_pred[:][0])/output_test[:][0]
    print(np.mean(error))
    '''