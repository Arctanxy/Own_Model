'''
坐标下降算法（还有坐标上升算法，原理类似）：
    例如要求一个min_f(w1,w2,...wn)的问题，其中各个xi是自变量，坐标下降法的运算步骤如下：
        1. 给定一个初始值，如w = (w1,w2,...wn)
        2. for dim in range(n):
            固定除w_dim以外的其他所有w
            以w_dim为自变量求使f取最小值的w_dim
            end
        3. 循环往复，直到新的f值不再变化或者变化很小
'''

import numpy as np 
import pandas as pd 

