# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 6 - shared variables
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

state = theano.shared(np.array(0,dtype=np.float64), 'state') # inital state = 0,定义GPU与CPU共享的全局变量
# theano中的所有的变量类型要统一,不然会报错
inc = T.scalar('inc', dtype=state.dtype)    # 或用inc = T.dscalar('inc')代替
accumulator = theano.function([inc], state, updates=[(state, state+inc)])    # 相当于 state += inc
                            # ([输入], 返回值, updates)
# 直接print不太好,打印的是上一步的值
# print(accumulator(10))  # --> 0
# print(accumulator(10))  # --> 10
# print(accumulator(10))  # --> 20


print(state.get_value())   # 此种方法获取shared的值
accumulator(1)   # return previous value, 0 in here
print(state.get_value())
accumulator(10)  # return previous value, 1 in here
print(state.get_value())

# to set variable value
state.set_value(-1)    # 重新设置shared的值
accumulator(3)
print(state.get_value())


# 好像多此一举,没什么用
# shared变量不可以直接传入到function,要通过临时变量a代替
tmp_func = state * 2 + inc
a = T.scalar(dtype=state.dtype)   # a是一个空变量,用作state的临时变量
skip_shared = theano.function([inc, a], tmp_func, givens=[(state, a)])
print(skip_shared(2, 3))   # a传入了3
print(state.get_value()) # old state value
