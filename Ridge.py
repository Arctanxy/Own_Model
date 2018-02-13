import numpy as np 
import pandas as pd 

#参考地址：http://python.jobbole.com/88799/

'''
    岭回归是在最小二乘法的代价函数的基础上添加了L2范数作为惩罚项
    L2范数随着模型的复杂度增大而增大
    模型约复杂就越容易过拟合
    所以添加了L2范数之后就可以有效地降低过拟合的风险
    
    化简后Ridge回归的参数求解公式为：
        w = (x^T x + \lambda I)^(-1) x^T y
'''

'''
    Numpy基本矩阵运算，数据结构为matrix，而非ndarray
    .T ——返回自身的转置
    .H ——返回自身的共轭转置
    .I ——返回自身的逆矩阵
    输入的原始数据为DataFrame形式，在运算中使用np.matrix()将之转换为矩阵
'''
#矩阵微分算法
def ridge_regression(x,y,lam = 0.2):
    '''
    直接使用矩阵微分来计算函数
    x,y 都是矩阵
    w是模型参数
    '''
    xtx = x.T * x
    print(xtx)
    m = xtx.shape[0]
    print(m)
    I = np.matrix(np.eye(m))
    print(I)
    w = (xtx + lam * I).I * x.T *y
    return w

#随机梯度下降算法
class ridge(object):
    def __init__(self,learning_rate=0.1):
        self.learning_rate = learning_rate
    
    def fit(self,input,output):
        '''
        input和ouput均为矩阵形式,公式为y = xw
        将样本依次加入计算，公式：$\theta_j := \theta_j + \alpha (y^i - h_\theta(x^i))x^i_j$
        '''
        input = np.matrix(input.values)
        output = np.matrix(output.values)

        if input.shape[0] != output.shape[0]:
            return 'Input length should be the same with ouput length!'
        theta = np.ones([input.shape[1],output.shape[1]])#初始化参数theta
        for i in range(input.shape[0]):#遍历样本
            print(input[i][:].shape,output[i][:].shape)
            #按梯度下降方向进行优化
            theta = theta + self.learning_rate*((output[i][:]-input[i][:]*theta)*input[i][:]).T
            print(theta)
        

if __name__ == '__main__':
    input = pd.DataFrame({'A':[1,2,3,4],'B':[2,3,4,5]})
    output = pd.DataFrame({'C':[2,3,2,4]})
    clf = ridge(learning_rate=0.2)
    clf.fit(input,output)