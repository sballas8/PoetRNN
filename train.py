import numpy as np
import csv
import argparse
import os
import json
import rnn.preprocessing as prep
import rnn.RNN as RNN 
from rnn.rnn_classifier_trainer import *
import cPickle as pickle








def main(params):
    #create vocab dictionary
    data_filename=os.path.join('data',params['dataset'])
    poem_dict=prep.generate_dictionary(data_filename)
    vocab_size=len(poem_dict)
    #load poems
    poems=[]
    labels=[]
    with open(data_filename,'rb') as my_file:
        reader=csv.reader(my_file)
        for row in reader:
            poems.append(prep.poem_to_mat(row[0],poem_dict))
    #generate the labels
    for poem in poems:
        labels.append(prep.generate_labels(poem,poem_dict))
    # convert to numpy arrays
    poems=np.array(poems)
    labels=np.array(labels)
    #split the data into training and validation sets
    train_size=int(np.ceil(poems.shape[0]*params['train_frac']))
    poems_train=poems[:train_size]
    poems_val=poems[train_size:]
    labels_train=labels[:train_size]
    labels_val=labels[train_size:]
    
    #build the model if not given
    if params['init_model'] is None:
        if params['num_layers']==1:
            model=RNN.LSTM.init(vocab_size,params['hidden_size'],vocab_size)
            loss_function=RNN.LSTM_cost
        elif params['num_layers']==2:
            model=RNN.LSTM.init_two_layer(vocab_size,params['hidden_size'],params['hidden_size'],vocab_size)
            loss_function=RNN.two_layer_LSTM_cost
        else:
            raise ValueError('number of layers must be either 1 or 2, not %s' % params['num_layers'])
    #load the model if model is given
    else:
        model=pickle.load(open(params['init_model'],'rb'))
        if 'WLSTM' in model.keys():
            loss_funtion=RNN.LSTM_cost
            print 'Loaded one layer LSTM model'
        elif 'WLSTM1' in model.keys():
            loss_function=RNN.two_layer_LSTM_cost
            print 'Loaded two layer LSTM model'
        else:
            raise ValueError('model given is not one or two layer LSTM model')
    #train the model
        
    trainer=RNNClassifierTrainer()
    best_model, loss_history, train_acc, val_acc=trainer.train(poems_train,labels_train,poems_val,labels_val,model,loss_function,learning_rate=params['learning_rate'], learning_rate_decay=params['decay_rate'],
                                                 update='rmsprop', sample_batches=True,batch_size=params['batch_size'],
                                                 num_epochs=params['max_epochs'], decay_after=params['decay_after'],
                                                 verbose=True,drop_prob1=params['drop_prob_layer1'],drop_prob2=params['drop_prob_layer2'],
                                                 checkpoint_output_dir=params['checkpoint_output_dir'],max_val=params['eval_max_images'],
                                                 fappend=params['fappend'],iter_to_update=params['iter_to_update'],dictionary=poem_dict)

if __name__ == "__main__":
    #get arguments from the command line
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset', dest='dataset', default='haikus.csv', help='dataset: cvs filename of the poems inside the data directory')
    parser.add_argument('-o', '--checkpoint_output_directory', dest='checkpoint_output_dir', type=str, default='cv/', help='output directory to write checkpoints to')
    parser.add_argument('--hidden_size', dest='hidden_size', type=int, default=256, help='size of hidden layer in RNNs')
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', type=float, default=1e-3, help='solver learning rate')
    parser.add_argument('-c', '--regc', dest='regc', type=float, default=1e-8, help='regularization strength')
    parser.add_argument('-m', '--max_epochs', dest='max_epochs', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--decay_rate', dest='decay_rate', type=float, default=0.999, help='decay rate for adadelta/rmsprop')
    parser.add_argument('--decay_after', dest='decay_after',type=int, default=1,help='number of epochs to begin decaying learning rate after')
    parser.add_argument('-b', '--batch_size', dest='batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--eval_max_images', dest='eval_max_images', type=int, default=-1, help='for efficiency we can use a smaller number of images to get validation error')
    parser.add_argument('--drop_prob_layer1', dest='drop_prob_layer1', type=float, default=0.5, help='what dropout to apply in first layer of RNN')
    parser.add_argument('--drop_prob_layer2', dest='drop_prob_layer2', type=float, default=0.5, help='what dropout to apply in second layer of RNN')
    parser.add_argument('-n','--num_layers', dest='num_layers', type=int, default=2, help='number of hidden layers. can be either 1 or two')
    parser.add_argument('-t','--train_frac', dest='train_frac', type=float, default=.95, help='fraction of data in the training set, should be between 0 and 1')
    parser.add_argument('--fappend', dest='fappend', type=str, default='baseline', help='append this string to checkpoint filenames')
    parser.add_argument('--iter_to_update', dest='iter_to_update', type=int, default=1, help='how often to display progress')
    parser.add_argument('-i','--init_model',dest='init_model', default=None,help='location of initial model to start training, default is random model')
    
    
    args=parser.parse_args()
    params=vars(args)
    print 'parsed parameters:'
    print json.dumps(params, indent = 2)
    main(params)
    

