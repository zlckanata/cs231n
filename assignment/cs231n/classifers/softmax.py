import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """
  pass
  train_num = X.shape[0]
  num_class = W.shape[1]
  scores = X.dot(W) #N by C
  scores_max = np.max(scores, axis=1) #N by 1
  scores_norm = (scores.T-scores_max).T #N by C
  scores_exp = np.exp(scores_norm) #N by C
  scores_sumprob = np.sum(scores_exp, axis=1, keepdims=True)
  scores_prob = scores_exp/scores_sumprob
  scores_prob_right = scores_prob[range(train_num),y].reshape(train_num,1)
  scores_prob_multi_prepare = np.copy(scores_prob)
  scores_prob_wrong_sum_negative = scores_prob_right - 1
  scores_prob_multi_prepare[range(train_num), y] = scores_prob_wrong_sum_negative.squeeze()
  scores_prob_multi_final = -scores_prob_multi_prepare * scores_prob_right
  #using the fomulation of dloss, to do some tricky calculation
  #y_trueClass = np.zeros_like(prob)
  #y_trueClass[np.arange(num_train), y] = 1.0
  for x in xrange(train_num):# here I use one loop, two loops type can be make using extra y_trueClass and so...
  #this loop is not in full version, this is half loop with half vectorized method
    loss += -np.log(scores_prob[x,y[x]])
    dW += np.dot(np.matrix(X[x]).T,np.matrix(scores_prob_multi_final[x]))
  loss /= train_num
  loss += 0.5 * reg * np.sum(W * W)
  dW /= train_num
  dW += reg * W
  """#for this part, I mistake the derivate function, and I get complex process and wrong answer
  dW_each = np.zeros_like(W)
  num_train, dim = X.shape
  num_class = W.shape[1]
  f = X.dot(W)
  f_max = np.reshape(np.max(f, axis=1), (num_train, 1))
  prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True) # N by C
  y_trueClass = np.zeros_like(prob)
  y_trueClass[np.arange(num_train), y] = 1.0
  for i in xrange(num_train):
      for j in xrange(num_class):    
          loss += -(y_trueClass[i, j] * np.log(prob[i, j]))
          dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]
      dW += dW_each
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #from the net
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  """
  pass
  train_num = X.shape[0]
  scores = X.dot(W) #N by C
  scores_max = np.max(scores, axis=1) #N by 1
  scores_norm = (scores.T-scores_max).T #N by C
  scores_exp = np.exp(scores_norm) #N by C
  scores_sumprob = np.sum(scores_exp, axis=1, keepdims=True)
  scores_prob = scores_exp/scores_sumprob
  scores_prob_right = scores_prob[range(train_num),y].reshape(train_num,1)
  scores_prob_multi_prepare = np.copy(scores_prob)
  scores_prob_wrong_sum_negative = scores_prob_right - 1
  scores_prob_multi_prepare[range(train_num), y] = scores_prob_wrong_sum_negative.squeeze()
  scores_prob_multi_final = -scores_prob_multe_prepare * scores_prob_right
  loss += -np.sum(np.log(scores_prob[range(train),y]))/train_num + 0.5 * reg * np.sum(W * W)
  dW += np.dot(X.T,scores_prob_multi_final)/train_num + reg * W
  """#for this part, I mistake the derivate function, and I get complex process and wrong answer
  train_num = X.shape[0]
  scores = X.dot(W) #N by C
  scores_max = np.max(scores, axis=1) #N by 1
  scores_norm = (scores.T-scores_max).T #N by C
  scores_exp = np.exp(scores_norm) #N by C
  scores_sumprob = np.sum(scores_exp, axis=1, keepdims=True)
  scores_prob = scores_exp/scores_sumprob
  scores_prob_right = scores_prob[range(train_num),y]
  scores_prob_wrong_sum_negative = scores_prob_right - 1
  scores_prob_multi_prepare = np.copy(scores_prob)
  scores_prob_multi_prepare[range(train_num), y] = scores_prob_wrong_sum_negative
  scores_prob_multi_final = scores_prob_multi_prepare
  loss += -np.sum(np.log(scores_prob[range(train_num),y]))/train_num + 0.5 * reg * np.sum(W * W)
  dW += np.dot(X.T,scores_prob_multi_final)/train_num + reg * W
  #just measure a little(drop the *right)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

