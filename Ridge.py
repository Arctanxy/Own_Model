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

