import numpy as np
import csv


#some lists that are useful later
letter_distribution=np.array([.08167,.01492,.02782,.04253,.12702,.02228,.02015,.06094,.06966,.00153,.00772,.04025,.02406,.06749,.07507,.01929,.00095,.05987,.06327,.09056,.02758,.00978,.02361,.00150,.01974,.00074])
letters=np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'])
#generate letter to number dictionary
def generate_dictionary(location):
    with open(location,'rb') as my_file:
        reader=csv.reader(my_file)
        char_to_nums={}
        char_to_nums['\t']=0 # we use this as a special character later
        for row in reader:
            for letter in row[0]:
                if letter not in char_to_nums.keys():
                    char_to_nums[letter]=len(char_to_nums)
        my_file.close()
    return char_to_nums
#generate the reverse dictionary
def reverse_dictionary(dictionary):
    rev_dict={v: k for k, v in dictionary.iteritems()}
    return rev_dict    

#convert poems to matrices using 1-k encoding
def poem_to_mat(poem,dictionary):
    poem_length=len(poem)
    vocab_length=len(dictionary)
    poem_mat=np.zeros((poem_length,vocab_length))
    vocab_list=[dictionary[s] for s in poem]
    poem_mat[xrange(poem_length),vocab_list]=1
    return poem_mat
    
#generate labels for poems
def generate_labels(poem_mat,dictionary):
    labels=np.zeros(poem_mat.shape[0])
    labels[:-1]=np.argmax(poem_mat,axis=1)[1:]
    labels[-1]=dictionary['\t'] #label last character as tab to indicate the end has been reached. 
    return labels
    
#convert a batch of poems to a 3-tensor along with a tensor of labels and a mask
def poem_batch_to_tensor(X,y=None):
    b=len(X)
    v=len(X[0][0])
    #print b,v
    len_list=np.array([len(X[i]) for i in range(b)])
    m=np.max(len_list)
    num_chars=np.sum(len_list) # useful for 
    #pad with zeros so they are all same length
    X_mat=np.array([np.vstack((np.array(X[i]),np.zeros((m-len_list[i],v)))) for i in range(b)]) 
    X_mat=np.swapaxes(X_mat,0,1)
    y_mat=np.array([np.hstack((np.array(y[i]),np.zeros(m-len_list[i]))) for i in range(b)],dtype=int).T
    #create a mask of so that later we only accumulate cost for entries that actually correspond to letters
    mask=np.ones((m,b))
    for i in range(b):
        mask[len_list[i]:,i]=0
    return X_mat, mask, num_chars,y_mat