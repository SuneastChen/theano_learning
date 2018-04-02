# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 4 - basic usage
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import numpy as np
import theano.tensor as T
from theano import function

# basic
x = T.dscalar('x')    # 定义theano的变量
y = T.dscalar('y')
z = x+y     # define the actual function in here
f = function([x, y], z)  # the inputs are in [], and the output in the "z"

print(f(2,3))  # only give the inputs "x and y" for this function, then it will calculate the output "z"

# to pretty-print the function,查看函数的定义
from theano import pp
print(pp(z))       # z = x+y

# how about matrix
x = T.dmatrix('x')   # 定义theano的矩阵,dmatrix代表float64,fmatrix代表float32
y = T.dmatrix('y')
z = x + y    # 或z = T.dot(x, y) 
f = function([x, y], z)
print(f(np.arange(12).reshape((3,4)), 10*np.ones((3,4))))
