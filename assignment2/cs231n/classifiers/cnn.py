import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, stride = 1, pad = 0):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    W1 = weight_scale * np.random.randn(num_filters,input_dim[0],filter_size,filter_size)
    b1 = np.zeros((1,num_filters))
    #convout = num_filters * ((input_dim.shape[1] - filter_size)/stride + 1) * ((input_dim.shape[2] - filter_size)/stride + 1)
    #reluout = convout/4
    #out_size1 = ((input_dim.shape[1] - filter_size)/stride + 1)/2
    #out_size2 = ((input_dim.shape[2] - filter_size)/stride + 1)/2
    out_size1 = input_dim[1]/2
    out_size2 = input_dim[2]/2
    W2 = weight_scale * np.random.randn(hidden_dim, num_filters, out_size1, out_size2)
    b2 = np.zeros((1,hidden_dim))
    W3 = weight_scale * np.random.randn(num_classes, hidden_dim, 1, 1)
    b3 = np.zeros((1,num_classes))
    self.params['W1'] = W1
    self.params['W2'] = W2
    self.params['W3'] = W3
    self.params['b1'] = b1
    self.params['b2'] = b2
    self.params['b3'] = b3
    '''
    C, H, W = input_dim
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['W2'] = weight_scale * np.random.randn(num_filters*H*W/4, hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b1'] = np.zeros((1, num_filters))
    self.params['b2'] = np.zeros((1, hidden_dim))
    self.params['b3'] = np.zeros((1, num_classes))
    '''
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
  '''
  def loss(self, X, y=None):
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    conv_aff_param = {'stride': 1, 'pad': 0}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    scores = None
    c1,cache1 = conv_forward_naive(X, W1, b1, conv_param)
    r1,cache2 = relu_forward(c1)
    p1,cache3 = max_pool_forward_naive(r1, pool_param)
    c2,cache4 = conv_forward_naive(p1, W2, b2, conv_aff_param)
    r2,cache5 = relu_forward(c2)
    c3,cache6 = conv_forward_naive(r2, W3, b3, conv_aff_param)
    scores = c3.squeeze()
    if y is None:
      return scores
    loss, grads = 0, {}
    loss,dloss = softmax_loss(scores, y)
    loss_reg = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    loss += loss_reg
    dc3 = dloss.reshape(dloss.shape[0], dloss.shape[1], 1, 1)
    dr2, dW3, db3 = conv_backward_naive(dc3, cache6)
    dc2 = relu_backward(dr2, cache5)
    dp1, dW2, db2 = conv_backward_naive(dc2, cache4)
    dr1 = max_pool_backward_naive(dp1, cache3)
    dc1 = relu_backward(dr1, cache2)
    dx, dW1, db1 = conv_backward_naive(dc1, cache1)
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    return loss, grads
  '''

  def loss(self, X, y=None):
    #this is fast
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    conv_aff_param = {'stride': 1, 'pad': 0}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    scores = None
    c1,cache1 = conv_forward_strides(X, W1, b1, conv_param)
    r1,cache2 = relu_forward(c1)
    p1,cache3 = max_pool_forward_reshape(r1, pool_param)
    c2,cache4 = conv_forward_strides(p1, W2, b2, conv_aff_param)
    r2,cache5 = relu_forward(c2)
    c3,cache6 = conv_forward_strides(r2, W3, b3, conv_aff_param)
    scores = c3.squeeze()
    if y is None:
      return scores
    loss, grads = 0, {}
    loss,dloss = softmax_loss(scores, y)
    loss_reg = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    loss += loss_reg
    dc3 = dloss.reshape(dloss.shape[0], dloss.shape[1], 1, 1)
    dr2, dW3, db3 = conv_backward_strides(dc3, cache6)
    dc2 = relu_backward(dr2, cache5)
    dp1, dW2, db2 = conv_backward_strides(dc2, cache4)
    dr1 = max_pool_backward_reshape(dp1, cache3)
    dc1 = relu_backward(dr1, cache2)
    dx, dW1, db1 = conv_backward_strides(dc1, cache1)
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    return loss, grads
  

  def loss_affine(self, X, y=None):
    #from the internet
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    a2, cache2 = affine_relu_forward(a1, W2, b2)
    scores, cache3 = affine_forward(a2, W3, b3)
    if y is None:
      return scores
    data_loss, dscores = softmax_loss(scores, y)
    da2, dW3, db3 = affine_backward(dscores, cache3)
    da1, dW2, db2 = affine_relu_backward(da2, cache2)
    dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    reg_loss = 0.5 * self.reg * sum(np.sum(W * W) for W in [W1, W2, W3])
    loss = data_loss + reg_loss
    grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
    return loss, grads
  
  '''
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    conv_aff_param = {'stride': 1, 'pad': 0}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #print W1.shape
    #print W2.shape
    #print W3.shape
    #print b1.shape
    #print b2.shape
    #print b3.shape
    c1,cache1 = conv_forward_naive(X, W1, b1, conv_param)
    r1,cache2 = relu_forward(c1)
    p1,cache3 = max_pool_forward_naive(r1, pool_param)
    c2,cache4 = conv_forward_naive(p1, W2, b2, conv_aff_param)
    r2,cache5 = relu_forward(c2)
    c3,cache6 = conv_forward_naive(r2, W3, b3, conv_aff_param)
    scores = c3.squeeze()
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dloss = softmax_loss(scores, y)
    loss_reg = 0.5 * self.reg * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    loss += loss_reg
    
    dc3 = dloss.reshape(dloss.shape[0], dloss.shape[1], 1, 1)
    dr2, dW3, db3 = conv_backward_naive(dc3, cache6)
    dc2 = relu_backward(dr2, cache5)
    dp1, dW2, db2 = conv_backward_naive(dc2, cache4)
    dr1 = max_pool_backward_naive(dp1, cache3)
    dc1 = relu_backward(dr1, cache2)
    dx, dW1, db1 = conv_backward_naive(dc1, cache1)
    
    dW1 += self.reg * W1
    dW2 += self.reg * W2
    dW3 += self.reg * W3
    
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['b1'] = db1
    grads['b2'] = db2
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  '''
