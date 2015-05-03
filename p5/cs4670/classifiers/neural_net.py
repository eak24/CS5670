import numpy as np
import matplotlib.pyplot as plt

def init_two_layer_model(input_size, hidden_size, output_size):
  """
  Initialize the weights and biases for a two-layer fully connected neural
  network. The net has an input dimension of D, a hidden layer dimension of H,
  and performs classification over C classes. Weights are initialized to small
  random values and biases are initialized to zero.

  Inputs:
  - input_size: The dimension D of the input data
  - hidden_size: The number of neurons H in the hidden layer
  - ouput_size: The number of classes C

  Returns:
  A dictionary mapping parameter names to arrays of parameter values. It has
  the following keys:
  - W1: First layer weights; has shape (D, H)
  - b1: First layer biases; has shape (H,)
  - W2: Second layer weights; has shape (H, C)
  - b2: Second layer biases; has shape (C,)
  """
  # initialize a model
  model = {}
  model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)
  model['b1'] = np.zeros(hidden_size)
  model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)
  model['b2'] = np.zeros(output_size)
  return model

def two_layer_net(X, model, y=None, reg=0.0):
  """
  Compute the loss and gradients for a two layer fully connected neural network.
  The net has an input dimension of D, a hidden layer dimension of H, and
  performs classification over C classes. We use a softmax loss function and L2
  regularization for the weight matrices (no regularization on the bias).
  Recall that "softmax loss" is shorthand for softmax followed by cross-entropy
  loss.  The two layer net should use a ReLU nonlinearity after the first
  affine layer.

  The two layer net has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each
  class.

  Inputs:
  - X: Input data of shape (N, D). Each X[i] is a training sample.
    **NOTE** data is stored row-major (each sample is a row).  In course notes,
    each training sample is often shown as a column vector.
  - model: Dictionary mapping parameter names to arrays of parameter values.
    It should contain the following:
    - W1: First layer weights; has shape (D, H)
    - b1: First layer biases; has shape (H,)
    - W2: Second layer weights; has shape (H, C)
    - b2: Second layer biases; has shape (C,)
  - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
    an integer in the range 0 <= y[i] < C. This parameter is optional; if it
    is not passed then we only return scores, and if it is passed then we
    instead return the loss and gradients.
  - reg: Regularization strength.

  Returns:
  If y not is passed, return a matrix scores of shape (N, C) where scores[i, c]
  is the score for class c on input X[i].

  If y is passed, instead return a tuple of:
  - loss: Loss (data loss and regularization loss) for this batch of training
    samples.
  - grads: Dictionary mapping parameter names to gradients of those parameters
    with respect to the loss function. This should have the same keys as model.
  """

  # unpack variables from the model dictionary
  W1,b1,W2,b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, D = X.shape

  # compute the forward pass
  scores = None
  #############################################################################
  # TODO: Perform the forward pass, computing the class scores for the input. #
  # Store the result in the scores variable, which should be an array of      #
  # shape (N, C).                                                             #
  #############################################################################
  # TODO-BLOCK-BEGIN
  first_l = X.dot(W1)# first layer: apply weights , (D,H)
  first_l += b1[np.newaxis,:] #add bias to first layer
  second_l =np.maximum(first_l,0) #second layer: ReLU
  third_l = second_l.dot(W2) #third layer: apply weights, (N,C)
  third_l += b2 #add bias to the third layer
  scores = third_l#-np.log(scores)
  #print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
  # TODO-BLOCK-END
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # If the targets are not given then jump out, we're done
  if y is None:
    return scores

  # compute the loss
  loss = None
  #############################################################################
  # TODO: Finish the forward pass, and compute the loss. This should include  #
  # both the data loss and L2 regularization for W1 and W2. Store the result  #
  # in the variable loss, which should be a scalar. Use the Softmax           #
  # classifier loss.  For the softmax, Don't forget to shiftt the scores      #
  # (subtract the max) so that the exp is numerically stable.                 #
  # So that your results match ours, multiply the regularization loss by 0.5  #
  #############################################################################
  # TODO-BLOCK-BEGIN
  fourh_l = third_l-third_l.max()
  fourh_l = np.exp(fourh_l) # fourth layer: softmax. Each row is an example
  L = fourh_l / np.sum(fourh_l,axis=1,keepdims=True) 
  loss = -np.log(L)
  indexes  = np.arange(N)*scores.shape[1] + y #From each row i, take the y[i]th loss
  loss = loss.flatten()[indexes].sum()
  loss /= N
  loss+= .5*reg*((W1**2).sum()+(W2**2).sum())
  #output

  #print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
  # TODO-BLOCK-END
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  # compute the gradients
  grads = {}
  #############################################################################
  # TODO: Compute the backward pass, computing the derivatives of the weights #
  # and biases. Store the results in the grads dictionary. For example,       #
  # grads['W1'] should store the gradient on W1, and be a matrix of same size #
  #############################################################################
  # TODO-BLOCK-BEGIN
  #Compute dL/dFj,i
  dLdF = L.flatten() # = e^{scores_j,i}/Sum[e^{scores_j,i}]
  dLdF[indexes] -=1
  dLdF/=N
  dLdF =dLdF.reshape(L.shape)
  grads['b2'] = dLdF.sum(axis=0)#dL/db2 = Sum[ dL/dF_{j,i},j]

  #Fij = H(i,:) dot W2(:,j) +B(i)
  # = H(i,1)*W2(1,j)...H(i,x)*W2(x,j)+... +B(i)
  # dFij/dWxj = H(i,x)
  # dL/dWX,Y  = Sum[dL/dF(i,y) dF(i,y)/dW(x,y)] =Sum[dL/dF(i,y) H(i,x)] = dL/dF(:,y) dot H(:,x) 
  #Compute dL/dW2(x,y)
  # dL/dWxy
  grads['W2'] = dLdF.T.dot(second_l).T
  grads['W2'] += W2*reg #the gradient due to the regularization

  #C = Max[XW1 + b1,0]
  #dL/dC = dL/dF dot (W2.T)
  dLdC = dLdF.dot(W2.T)*(first_l>0)
  grads['b1'] = (dLdC).sum(axis=0)

  grads['W1'] = dLdC.T.dot(X).T
  grads['W1'] += W1*reg #the gradient due to the regularization


  #import inspect
  #frameinfo = inspect.getframeinfo(inspect.currentframe())
  #print "TODO: {}: line {}".format(frameinfo.filename, frameinfo.lineno)
  # TODO-BLOCK-END
  #############################################################################
  #                              END OF YOUR CODE                             #
  #############################################################################

  return loss, grads


