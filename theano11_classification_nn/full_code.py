# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 11 - classification example
"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T

def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction)/len(correct_prediction)
    return accuracy


N = 400                                   # training sample size
feats = 784                               # number of input variables

# generate a dataset: D = (input_values, target_class) D为元组
D = (np.random.randn(N, feats), np.random.randint(size=N, low=0, high=2))    # X.shape(400, 784)

# 声明Theano符号变量
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weights and biases
W = theano.shared(np.random.randn(feats), name="w")   # W.shape(784,)
b = theano.shared(0.01, name="b")


# 构建Theano表达图
p_1 = T.nnet.sigmoid(T.dot(x, W) + b)   # Logistic Probability that target = 1 (activation function)
prediction = p_1 > 0.5                    # 预测阈值
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # 整批数据的Cross-entropy loss function
# or
# xent = T.nnet.binary_crossentropy(p_1, y) # this is provided by theano
cost = xent.mean() + 0.01 * (W ** 2).sum() # 加入 (l2 regularization)的损失函数
gW, gb = T.grad(cost, [W, b])              # Compute the gradient of the cost


# 定义训练与预测函数
learning_rate = 0.1
train = theano.function(
          inputs=[x, y],
          outputs=[prediction, xent.mean()],
          updates=((W, W - learning_rate * gW), (b, b - learning_rate * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Training
for i in range(500):
    pred, err = train(D[0], D[1])
    if i % 50 == 0:
        print('cost:', err)
        print("accuracy:", compute_accuracy(D[1], predict(D[0])))

print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

