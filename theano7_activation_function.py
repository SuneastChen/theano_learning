# View more python tutorials on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

# 7 - Activation function

"""
The available activation functions in theano can be found in this link:
http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html

The activation functions include but not limited to softplus, sigmoid, relu, softmax, elu, tanh...

For the hidden layer, we could use relu, tanh, softplus...
For classification problems, we could use sigmoid or softmax for the output layer.
For regression problems, we could use a linear function for the output layer.

"""

theano.tensor.nnet.nnet.sigmoid(x)
theano.tensor.nnet.nnet.softplus(x)
theano.tensor.nnet.nnet.softmax(x)
theano.tensor.nnet.relu(x, alpha=0)




import theano.tensor as T
x, y, b = T.dvectors('x', 'y', 'b')
W = T.dmatrix('W')
y = T.nnet.sigmoid(T.dot(W, x) + b)