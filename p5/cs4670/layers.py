import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)

  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  # TODO-BLOCK-BEGIN
  N = x.shape[0]
  X = x.reshape((N,np.prod(x.shape[1:])))
  out = X.dot(w) # apply weights , (D,H)
  out += b[np.newaxis,:] # add bias layer
  # TODO-BLOCK-END
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  # TODO-BLOCK-BEGIN
  N = x.shape[0]
  X = x.reshape((N,np.prod(x.shape[1:])))
  db = dout.sum(axis=0)

  dw = dout.T.dot(X).T
  dx = dout.dot(w.T)
  dx=dx.reshape(x.shape)
  # TODO-BLOCK-END
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  # TODO-BLOCK-BEGIN
  out = np.maximum(x,0)
  # TODO-BLOCK-END
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  # TODO-BLOCK-BEGIN
  dx = dout * (x>0)
  # TODO-BLOCK-END
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  **NOTE**: The filters are already pre-flipped for convolution, so your code
  should implement a cross-correlation (i.e. don't flip the filter).

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  # TODO-BLOCK-BEGIN
  # Calculate output array shape
  N = x.shape[0]
  C = x.shape[1]
  H = x.shape[2]
  W = x.shape[3]
  HH = w.shape[2]
  WW = w.shape[3]
  F = w.shape[0]
  pad = conv_param['pad']
  stride = conv_param['stride']
  H_out = 1 + (H + 2 * pad - HH) / stride
  W_out = 1 + (W + 2 * pad - WW) / stride
  # Create the output array
  out = np.zeros((N,F,H_out,W_out))
  # Pad the array with zero-padding
  x_pad = np.zeros((N,C,2*pad+H,2*pad+W))
  for n in range(N):
    for c in range(C):
      x_pad[n,c,:,:] = np.pad(x[n,c,:,:],pad_width=pad,mode='constant')
  # Populate output array
  for n in range(N):
    for i_out in range(0,H_out):
      for j_out in range(0,W_out):
        for f in range(0,F):
          # origin coordinates for the input matrix, x:
          i_in = pad-1 + i_out * stride
          j_in = pad-1 + j_out * stride
          # apply the weights and bias:
          out[n,f,i_out,j_out] = np.sum(\
            x_pad[n,:,i_in:i_in+HH,j_in:j_in+WW]\
            *w[f,:,:,:]) + b[f]
  # TODO-BLOCK-END
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

                          *** EXTRA CREDIT ***

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # EXTRA CREDIT: Implement the convolutional backward pass.                  #
  #############################################################################
  # TODO-BLOCK-BEGIN
  import inspect
  frameinfo = inspect.getframeinfo(inspect.currentframe())
  print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
  # TODO-BLOCK-END
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  # TODO-BLOCK-BEGIN
  # Calculate output array shape
  N = x.shape[0]
  C = x.shape[1]
  H = x.shape[2]
  W = x.shape[3]
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  H_out = 1 + (H - HH) / stride
  W_out = 1 + (W - WW) / stride
  # Create the output array
  out = np.zeros((N,C,H_out,W_out))
  # Populate output array
  for n in range(N):
    for i_out in range(0,H_out):
      for j_out in range(0,W_out):
        for c in range(0,C):
          # origin coordinates for the input matrix, x:
          i_in = i_out * stride
          j_in = j_out * stride
          # apply the weights and bias:
          out[n,c,i_out,j_out] = np.max(x[n,c,i_in:i_in+HH,j_in:j_in+WW])
  # TODO-BLOCK-END
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  # TODO-BLOCK-BEGIN
  x,pool_param = cache
  N = x.shape[0]
  C = x.shape[1]
  H = x.shape[2]
  W = x.shape[3]
  HH = pool_param['pool_height']
  WW = pool_param['pool_width']
  stride = pool_param['stride']
  H_out = 1 + (H - HH) / stride
  W_out = 1 + (W - WW) / stride
  dx = np.zeros_like(x)
  for n in range(N):
    for i_out in range(H_out):
      for j_out in range(W_out):
        for c in range(C):
          # origin coordinates for the input matrix, x:
          i_in = i_out * stride
          j_in = j_out * stride
          max_ind = x[n,c,i_in:i_in+HH,j_in:j_in+WW].argmax()
          max_i,max_j = np.unravel_index(max_ind,(HH,WW))
          max_i += i_in
          max_j += j_in
          dx[n,c,max_i,max_j] = dout[n,c,i_out,j_out]
  # TODO-BLOCK-END
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


