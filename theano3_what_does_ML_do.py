# View more python tutorials on my Youtube and Youku channel!!!
from __future__ import print_function
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt


class Layer(object):   # 定义神经网络层类
    def __init__(self, inputs, in_size, out_size, activation_function=None):
        self.W = theano.shared(np.random.normal(0, 1, (in_size, out_size)))
        self.b = theano.shared(np.zeros((out_size, )) + 0.1)
        self.Wx_plus_b = T.dot(inputs, self.W) + self.b
        self.activation_function = activation_function
        if activation_function is None:
            self.outputs = self.Wx_plus_b
        else:
            self.outputs = self.activation_function(self.Wx_plus_b)


# Make up some fake data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise        # y = x^2 - 0.5

# show the fake data
plt.scatter(x_data, y_data)
plt.show()

# 定义theano的变量,确定dtype
x = T.dmatrix("x")
y = T.dmatrix("y")

# add layers
l1 = Layer(x, 1, 10, T.nnet.relu)
l2 = Layer(l1.outputs, 10, 1, None)

# compute the cost
cost = T.mean(T.square(l2.outputs - y))

# compute the gradients
gW1, gb1, gW2, gb2 = T.grad(cost, [l1.W, l1.b, l2.W, l2.b])    # 梯度下降,使cost最小,得到损失函数相对于各层W,b的导数

# apply gradient descent
learning_rate = 0.1
train = theano.function(     # 定义训练函数
    inputs=[x, y],
    outputs=[cost],
    updates=[(l1.W, l1.W - learning_rate * gW1),    # 再更新参数W,b
             (l1.b, l1.b - learning_rate * gb1),
             (l2.W, l2.W - learning_rate * gW2),
             (l2.b, l2.b - learning_rate * gb2)])

# 定义prediction 函数
predict = theano.function(inputs=[x], outputs=l2.outputs)

# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()



# 以上结构已建好,下面开始传入真实数据,训练
for i in range(1000):
    # training
    err = train(x_data, y_data)
    if i % 50 == 0:
        # to visualize the result and improvement
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = predict(x_data)
        # plot the prediction
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(.5)
# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 3 - What does machine learning do?
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
