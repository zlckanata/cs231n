import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

class ZlcNet(object):

  def __init__(self,conv_lay=1,pool_lay=[1],hidden_dim=[200],num_filter=[32],loss='softmax',
               input_dim=(3,32,32),num_class=10,use_batchnorm=False,use_spatial_batchnorm=False,
               filter_size=[3],weight_scale=1e-3,reg=0.0,dropout=0.0,
               dtype=np.float32, stride = 1):
    self.dropout = dropout > 0
    self.use_batchnorm = use_batchnorm
    self.use_spatial_batchnorm = use_spatial_batchnorm
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.filter_size = filter_size
    self.num_filter = num_filter
    self.conv_lay = conv_lay
    self.pool_lay = pool_lay
    self.fc_lay = len(hidden_dim)+1
    self.loss_choice = loss

    assert not(loss == 'svm' and loss == 'softmax'), 'plz choose right loss_func'
    assert conv_lay == len(num_filter), 'conv_lay and num_filter do not fit'
    assert len(pool_lay) <= 5, 'the num of elements in pool_lay should be no more than 5'
    assert len(hidden_dim) <= 2, 'the num of fc should be no more than 2(more is just waste)'
    assert conv_lay == len(filter_size), 'conv_lay and filter_size do not fit'
    assert len(hidden_dim) > 0, 'hidden_dim should be no less than one(cause one of the hidden_dim is for the loss layer)'
    if len(pool_lay):
      assert max(pool_lay) <= conv_lay, 'pool_lay should follow conv_lay or input_lay'
    
    channels = [input_dim[0]] + num_filter
    
    #initialize params in conv_lay(two ways)
    if weight_scale < 0:
      for i in range(conv_lay):
        print np.sqrt(2.0/(channels[i] * filter_size[i] * filter_size[i])) * np.abs(weight_scale)
        self.params['conv_W'+str(i+1)] = np.sqrt(2.0/(channels[i] * filter_size[i] * filter_size[i])) * np.random.randn(channels[i+1],channels[i],filter_size[i],filter_size[i]) * np.abs(weight_scale)
        self.params['conv_b'+str(i+1)] = np.zeros((1,channels[i+1]))
    else:
      for i in range(conv_lay):
        print weight_scale
        self.params['conv_W'+str(i+1)] = weight_scale * np.random.randn(channels[i+1],channels[i],filter_size[i],filter_size[i])
        self.params['conv_b'+str(i+1)] = np.zeros((1,channels[i+1]))
    
    conv_out_side_size = input_dim[2]/(2**len(pool_lay))

    fc_dims = [channels[-1]] + hidden_dim + [num_class]
    
    #initialize params in fc_lay(two ways)
    if weight_scale < 0:
      print np.sqrt(2.0/(fc_dims[0] * conv_out_side_size * conv_out_side_size)) * np.abs(weight_scale)
      self.params['fc_W'+str(1)] = np.sqrt(2.0/(fc_dims[0] * conv_out_side_size * conv_out_side_size)) * np.random.randn(fc_dims[1],fc_dims[0],conv_out_side_size,conv_out_side_size) * np.abs(weight_scale)
      self.params['fc_b'+str(1)] = np.zeros((1,fc_dims[1]))
      for i in range(len(hidden_dim)):
        print np.sqrt(2.0/fc_dims[i+1]) * np.abs(weight_scale)
        self.params['fc_W'+str(i+2)] = np.sqrt(2.0/fc_dims[i+1]) * np.random.randn(fc_dims[i+2],fc_dims[i+1],1,1) * np.abs(weight_scale)
        self.params['fc_b'+str(i+2)] = np.zeros((1,fc_dims[i+2]))
    else:
      print weight_scale
      self.params['fc_W'+str(1)] = weight_scale * np.random.randn(fc_dims[1],fc_dims[0],conv_out_side_size,conv_out_side_size)
      self.params['fc_b'+str(1)] = np.zeros((1,fc_dims[1]))
      for i in range(len(hidden_dim)):
        print weight_scale
        self.params['fc_W'+str(i+2)] = weight_scale * np.random.randn(fc_dims[i+2],fc_dims[i+1],1,1)
        self.params['fc_b'+str(i+2)] = np.zeros((1,fc_dims[i+2]))

      for k, v in self.params.iteritems():
        self.params[k] = v.astype(dtype)
    

  def loss(self, X, y=None):
    #this is fast
	
    num_filter = self.num_filter
    pool_lay = self.pool_lay
    conv_lay = self.conv_lay
    fc_lay = self.fc_lay

    conv_W = {}
    conv_b = {}
    conv_W[0] = 0
    conv_b[0] = 0
    for i in range(conv_lay):
      conv_W[i+1] = self.params['conv_W'+str(i+1)]
      conv_b[i+1] = self.params['conv_b'+str(i+1)]

    fc_W = {}
    fc_b = {}
    fc_W[0] = 0
    fc_b[0] = 0
    for i in range(fc_lay):
      fc_W[i+1] = self.params['fc_W'+str(i+1)]
      fc_b[i+1] = self.params['fc_b'+str(i+1)]

    filter_size = self.filter_size

    conv_param = {}
    for i in range(conv_lay):
      conv_param[i] = {'stride': 1, 'pad': (filter_size[i] - 1) / 2}
    conv_aff_param = {'stride': 1, 'pad': 0}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    scores = None
    cache = {}
    out = X

    for i in range(conv_lay):
      out,cache['c'+str(i+1)] = conv_forward_strides(out, conv_W[i+1], conv_b[i+1], conv_param[i])
      out,cache['cr'+str(i+1)] = relu_forward(out)
      if(i+1 in pool_lay):
        out,cache['p'+str(i+1)] = max_pool_forward_reshape(out, pool_param)
	
    for i in range(fc_lay-1):
      out,cache['a'+str(i+1)] = conv_forward_strides(out, fc_W[i+1], fc_b[i+1], conv_aff_param)
      out,cache['ar'+str(i+1)] = relu_forward(out)
  
    out,cache['a'+str(fc_lay)] = conv_forward_strides(out, fc_W[fc_lay], fc_b[fc_lay], conv_aff_param)

    scores = out.squeeze()
    
    if y is None:
      return scores
    if self.loss_choice == 'softmax':
      loss, grads = 0, {}
      loss,dloss = softmax_loss(scores, y)
      loss_reg = 0
      for i in range(conv_lay):
        loss_reg += 0.5 * self.reg * (np.sum(conv_W[i+1]**2))
      for i in range(fc_lay):
        loss_reg += 0.5 * self.reg * (np.sum(fc_b[i+1]**2))
      loss += loss_reg

    if self.loss_choice == 'svm':
      loss, grads = 0, {}
      loss,dloss = svm_loss(scores, y)
      loss_reg = 0
      for i in range(conv_lay):
        loss_reg += 0.5 * self.reg * (np.sum(conv_W[i+1]**2))
      for i in range(fc_lay):
        loss_reg += 0.5 * self.reg * (np.sum(fc_b[i+1]**2))
      loss += loss_reg

    dW_conv = {}
    db_conv = {}
    dW_fc = {}
    db_fc = {}

    #reshape the dloss to fit the conv_backprop
    dout = dloss.reshape(dloss.shape[0], dloss.shape[1], 1, 1)
    #go ahead for only one step to make the "for loop" below more convenient
    dout,dW_fc[fc_lay],db_fc[fc_lay] = conv_backward_strides(dout, cache['a'+str(fc_lay)])

    for i in range(fc_lay-1)[::-1]:
      dout = relu_backward(dout, cache['ar'+str(i+1)])
      dout,dW_fc[i+1],db_fc[i+1] = conv_backward_strides(dout, cache['a'+str(i+1)])
  
    for i in range(conv_lay)[::-1]:
      if i+1 in pool_lay:
        dout = max_pool_backward_reshape(dout, cache['p'+str(i+1)])
      dout = relu_backward(dout, cache['cr'+str(i+1)])
      dout,dW_conv[i+1],db_conv[i+1] = conv_backward_strides(dout, cache['c'+str(i+1)])

    for i in range(conv_lay):
      dW_conv[i+1] += self.reg * conv_W[i+1]
      grads['conv_W'+str(i+1)] = dW_conv[i+1]
      grads['conv_b'+str(i+1)] = db_conv[i+1]
	  
    for i in range(fc_lay):
      dW_fc[i+1] += self.reg * fc_W[i+1]
      grads['fc_W'+str(i+1)] = dW_fc[i+1]
      grads['fc_b'+str(i+1)] = db_fc[i+1]

    return loss, grads
    
