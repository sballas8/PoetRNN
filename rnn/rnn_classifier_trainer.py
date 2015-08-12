import numpy as np
import preprocessing as prep
import time
import cPickle as pickle
import os

# this came from Fei-Fei Li and Andrej Karpathy's Convolutional Neural network class http://cs231n.stanford.edu with only minor modifications
class RNNClassifierTrainer(object):
  """ The trainer class performs SGD with momentum on a cost function """
  def __init__(self):
    self.step_cache = {} # for storing velocities in momentum update

  def train(self, X, y, X_val_list, y_val_list, 
            model, loss_function, 
            reg=0.0,
            learning_rate=1e-2, momentum=0, learning_rate_decay=0.95,decay_after=1,
            update='momentum', sample_batches=True,
            num_epochs=30, batch_size=100, acc_frequency=None,
            verbose=False,**kwargs):
    """
    Optimize the parameters of a model to minimize a loss function. We use
    training data X and y to compute the loss and gradients, and periodically
    check the accuracy on the validation set.

    Inputs:
    - X: Array of training data; each X[i] is a training sample.
    - y: Vector of training labels; y[i] gives the label for X[i].
    - X_val: Array of validation data
    - y_val: Vector of validation labels
    - model: Dictionary that maps parameter names to parameter values. Each
      parameter value is a numpy array.
    - loss_function: A function that can be called in the following ways:
      scores = loss_function(X, model, reg=reg)
      loss, grads = loss_function(X, model, y, reg=reg)
    - reg: Regularization strength. This will be passed to the loss function.
    - learning_rate: Initial learning rate to use.
    - momentum: Parameter to use for momentum updates.
    - learning_rate_decay: The learning rate is multiplied by this after each
      epoch.
    - update: The update rule to use. One of 'sgd', 'momentum', or 'rmsprop'.
    - sample_batches: If True, use a minibatch of data for each parameter update
      (stochastic gradient descent); if False, use the entire training set for
      each parameter update (gradient descent).
    - num_epochs: The number of epochs to take over the training data.
    - batch_size: The number of training samples to use at each iteration.
    - acc_frequency: If set to an integer, we compute the training and
      validation set error after every acc_frequency iterations.
    - verbose: If True, print status after each epoch.

    Returns a tuple of:
    - best_model: The model that got the highest validation accuracy during
      training.
    - loss_history: List containing the value of the loss function at each
      iteration.
    - train_acc_history: List storing the training set accuracy at each epoch.
    - val_acc_history: List storing the validation set accuracy at each epoch.
    """
    #unpack the function arguments
    drop_prob1=kwargs.get('drop_prob1',0.0)
    drop_prob2=kwargs.get('drop_prob2',0.0)
    checkpoint_output_dir=kwargs.get('checkpoint_output_dir','cv')
    max_val=kwargs.get('max_val',-1)
    fappend=kwargs.get('fappend')
    iter_to_update=kwargs.get('iter_to_update',1)
    #unpack dictionary and strip it from the model
    dictionary=model['dictionary']
    model={k:v for k,v in model.iteritems() if k!='dictionary'}
    N = len(X)
    M = len(X_val_list)
    batch_val=True
    if max_val==-1:
        batch_val=False
    #batch the validation data and labels into tensors and create the mask
    X_val,val_mask,val_chars,y_val=prep.poem_batch_to_tensor(X_val_list,y_val_list)
    if sample_batches:
      iterations_per_epoch = N / batch_size # using SGD
    else:
      iterations_per_epoch = 1 # using GD
    num_iters = num_epochs * iterations_per_epoch
    epoch = 0
    best_val_loss = 100.0 # probably loss will never be this large
    best_model = {}
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    time_list=[]
    for it in xrange(num_iters):
      t0=time.time()  
      if it % iter_to_update == 0:
        print 'starting iteration ', it
        if it>0:
          avg_time=np.mean(np.array(time_list))
          time_list=[]
          print 'Average time per batch is %.3f seconds' % avg_time

      # get batch of data
      if sample_batches:
        batch_mask = np.random.choice(N, batch_size)
        X_batch_list = X[batch_mask]
        y_batch_list = y[batch_mask]
      else:
        # no SGD used, full gradient descent
        X_batch_list = X
        y_batch_list = y
      X_batch,batch_mask,batch_num_chars,y_batch=prep.poem_batch_to_tensor(X_batch_list,y_batch_list)

      # evaluate cost and gradient
      #print 'computing loss'  
      cost, grads = loss_function(X_batch, y_batch,batch_mask, model,batch_num_chars, reg,drop_prob1=drop_prob1,drop_prob2=drop_prob2)
      loss_history.append(cost)

      # perform a parameter update
      for p in model:
        # compute the parameter step
        if update == 'sgd':
          dx = -learning_rate * grads[p]
        #code for momentum update
        elif update == 'momentum':
          if not p in self.step_cache:
            #print 'adding ', p
            self.step_cache[p] = np.zeros(grads[p].shape)
          
          self.step_cache[p]*=momentum
          self.step_cache[p]+=-learning_rate*grads[p]  
          dx=self.step_cache[p]
          #code for rmsprop update
        elif update == 'rmsprop':
          decay_rate = 0.95 # you could also make this an option
          smoothing=1e-8  
          if not p in self.step_cache: 
            self.step_cache[p] = np.zeros(grads[p].shape)
         
          #####################################################################
          self.step_cache[p]=decay_rate*self.step_cache[p]+(1-decay_rate)*grads[p]**2
          dx=-learning_rate*grads[p]/np.sqrt(self.step_cache[p]+smoothing) 
        else:
          raise ValueError('Unrecognized update type "%s"' % update)

        # update the parameters
        #print np.linalg.norm(dx)
        model[p] += dx
        dt=time.time()-t0
        time_list.append(dt)
        

      # every epoch perform an evaluation on the validation set
      first_it = (it == 0)
      epoch_end = (it + 1) % iterations_per_epoch == 0
      acc_check = (acc_frequency is not None and it % acc_frequency == 0)
      if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
          # decay the learning rate once the epoch is more than decay_after
          if epoch>decay_after-1:  
            learning_rate *= learning_rate_decay
          epoch += 1
            

        # evaluate train accuracy
        if N > 100:
          train_mask = np.random.choice(N, 100)
          X_train_subset_list = X[train_mask]
          y_train_subset_list = y[train_mask]
        else:
          X_train_subset_list = X
          y_train_subset_list = y
        #batch the masked data
        X_train_mat,train_mask,train_num_chars,y_train_mat=prep.poem_batch_to_tensor(X_train_subset_list,y_train_subset_list)
        scores_train = loss_function(X_train_mat, model=model,mask=train_mask,num_chars=train_num_chars)
        y_pred_train = np.argmax(scores_train, axis=-1)
        train_acc = np.mean(y_pred_train == y_train_mat)
        train_acc_history.append(train_acc)

        # evaluate val accuracy
        # use all validation data if not batching
        if batch_val==False:
          scores_val = loss_function(X_val, model=model,mask=val_mask,num_chars=val_chars)
          val_loss,_ = loss_function(X_val, y_val, model=model,mask=val_mask,num_chars=val_chars)
          y_pred_val = np.argmax(scores_val, axis=-1)
          val_acc = np.mean(y_pred_val ==  y_val)
          val_acc_history.append(val_acc)
        #take a sample of the validation data if we are batching
        else:
          val_mask=np.random.choice(M,max_val)
          X_val_subset_list=X_val_list[val_mask]
          y_val_subset_list=y_val_list[val_mask]
          #batch the validation sample
          X_val_mat,val_mask,val_num_chars,y_val_mat=prep.poem_batch_to_tensor(X_val_subset_list,y_val_subset_list)
          scores_val=loss_function(X_val_mat,model=model,mask=val_mask,num_chars=val_num_chars)
          val_loss,_=loss_function(X_val_mat,y_val_mat,model=model,mask=val_mask,num_chars=val_num_chars)
          y_pred_val=np.argmax(scores_val,axis=-1)
          val_acc=np.mean(y_pred_val==y_val_mat)
          val_acc_history.append(val_acc)    
        
        # keep track of the best model based on validation accuracy
        if val_loss < best_val_loss and epoch>0:
          # make a copy of the model
          best_val_loss = val_loss
          best_model = {}
          #add the dictionary for when we sample
          best_model['dictionary']=dictionary
          for p in model:
            best_model[p] = model[p].copy()
          #save the model if it is best
          filename=os.path.join(checkpoint_output_dir,'checkpoint_%s_%.3f.p' % (fappend,val_loss))
          print 'Saving epoch %s model to file' % epoch
          pickle.dump(best_model,open(filename,'wb'))    

        # print progress if needed
        if verbose:
          print ('Finished epoch %d / %d: cost %f, train: %f, val %f, lr %e'
                 % (epoch, num_epochs, cost, train_acc, val_acc, learning_rate))
    
    # always save the final model
    final_model=model.copy()
    final_model['dictionary']=dictionary
    filename=os.path.join(checkpoint_output_dir,'finalcheckpoint_%s_%.3f.p' % (fappend,val_loss))
    pickle.dump(final_model,open(filename,'wb'))             
    if verbose:
      print 'finished optimization. best validation loss: %f' % (best_val_loss)
    # return the best model and the training history statistics
    return best_model, loss_history, train_acc_history, val_acc_history











