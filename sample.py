import numpy as np
import argparse
import os
import json
import cPickle as pickle
import rnn.RNN as RNN


def main(params):
    #load the model
    filename=os.path.join('cv',params['model'])
    model=pickle.load(open(filename,'rb'))
    if 'WLSTM' in model.keys():
        num_layers=1
    elif 'WLSTM1' in model.keys():
        num_layers=2
    else: 
        raise ValueError('model is not one or two layer LSTM')
    #generate poems
    poem_list=[]
    for i in xrange(params['num_poems']):
        poem=RNN.LSTM_sample(model,temp=params['temp'],seed=params['seed'],num_layers=num_layers)
        poem_list.append(poem)
    
    #print poems if desired
    # 0 and 2 indicate print so this works well
    if params['output']%2==0:
        for poem in poem_list:
            print poem+'\n'
    if params['output']>0:
        filename=os.path.join('poems',params['output_filename']+'.txt')
        with open(filename,'wb') as my_file:
            for poem in poem_list:
                my_file.write(poem+'\n')
    
    
    
    


#get arguments from command line

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    
    parser.add_argument('-m','--model', dest='model', help='filename of model in cv folder to sample from', required=True)
    parser.add_argument('-t','--temp', dest='temp', type=float, default='1.0', help='temperature for sampling, default is 1')
    parser.add_argument('-s','--seed', dest='seed', type=str, default=None, help='a string to seed the model with when sampling')
    parser.add_argument('-n','--num_poems', dest='num_poems', type=int, default='10', help='number of poems to produce')
    parser.add_argument('-o','--output', dest='output', type=int, default=0, help='0:print result, 1:write to file, 2:print and write to file') 
    parser.add_argument('-f','--output_filename', dest='output_filename', default='poems', help='filename to output poems to')
    
    
    args=parser.parse_args()
    params=vars(args)
    print 'parsed parameters:'
    print json.dumps(params, indent = 2)
    main(params)