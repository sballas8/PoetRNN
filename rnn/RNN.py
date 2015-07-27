#First try at a "vanilla" RNN
import numpy as np
from preprocessing import reverse_dictionary,letters,letter_distribution

# a batched softmax function that can mask certain values
def softmax_loss(X,y,mask=None,num_chars=None):
    m,b=X.shape[:2]
    #create 'identity mask' if no mask exists
    if mask is None:
        mask=np.ones((m,b))
        num_chars=m*b
    probs=np.exp(X-np.amax(X,axis=-1, keepdims=True))
    probs/=np.sum(probs,axis=-1,keepdims=True)
    rows=np.array(range(m)*b,dtype=int).reshape((b,m)).T
    cols=np.array(range(b)*m,dtype=int).reshape((m,b))
    # accumulate loss only from unmasked values
    loss = -np.sum(np.log(probs[rows,cols, y])*mask) / num_chars
    grad = probs.copy()
    grad[rows,cols, y] -= 1
    #compute gradient for unmasked values
    grad*=mask[:,:,np.newaxis]
    grad /= num_chars
    return loss, grad


#static methods in this class seemed to make testing easier. This class is largely taken from Andrej Karpathy's
# gist https://gist.github.com/karpathy/587454dc0146a6ae21fc
class LSTM:
    #LSTM model with optional linear output_layer on top
    @staticmethod
    def init(input_size,hidden_size,output_size=None,output_layer=True):
        model={}
        WLSTM=np.random.randn(input_size+hidden_size+1,4*hidden_size)/np.sqrt(input_size+hidden_size) #randomly initialize weights
        WLSTM[0,:]=0 #initialize biases to zero
        model['WLSTM']=WLSTM
        #create output layer if needed
        if (not output_size is None) and output_layer:
            Wout=np.random.randn(hidden_size+1,output_size)/np.sqrt(hidden_size) 
            Wout[0,:]=0   
            model['Wout']=Wout
        return model
    #two layer LSTM model with linear output layer 
    @staticmethod
    def init_two_layer(input_size,hidden1,hidden2,output_size):
        model1=LSTM.init(input_size,hidden1,output_layer=False)
        model2=LSTM.init(hidden1,hidden2,output_size)
        model={}
        model['WLSTM1']=model1['WLSTM']
        model['WLSTM2']=model2['WLSTM']
        model['Wout']=model2['Wout']
        return model
    #todo: allow for arbitrarily many layers
    
    #LSTM forward pass with optional dropout, the drop_mask feature is useful for testing, but largely unnecessary    
    @staticmethod
    def batch_forward(X,model,output_layer=True,c0=None,h0=None,drop_prob=0.0,drop_mask=None):
        #unpack the model
        WLSTM=model['WLSTM']
        if output_layer:
            Wout=model['Wout']
        # X is a (n,b,v) array where n=sequence length, b=batch size and v=vocab size
        cache={} # a cache of variables for backpropogation
        n,b,v=X.shape
        d=WLSTM.shape[1]/4 #hidden size
        #initialize c and h to zero if not given
        if c0 is None:
            c0=np.zeros((b,d))
        if h0 is None:
            h0=np.zeros((b,d))
        xphpb = WLSTM.shape[0] # x plus h plus bias
        Hin = np.ones((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
        Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
        Houtstack=np.ones((n,b,d+1)) #hidden representation with bias
        IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
        IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
        C = np.zeros((n, b, d)) # cell content
        Ct = np.zeros((n, b, d)) # tanh of cell content
        for t in xrange(n):
            if t==0:
                prevh=h0
            else:
                prevh=Hout[t-1]
            Hin[t,:,1:v+1]=X[t]
            Hin[t,:,v+1:]=prevh
            IFOG[t]=Hin[t].dot(WLSTM)
            IFOGf[t,:,:3*d]=1.0/(1.0+np.exp(-IFOG[t,:,:3*d]))
            IFOGf[t,:,3*d:]=np.tanh(IFOG[t,:,3*d:])
            if t==0:
                prevc=c0
            else:
                prevc=C[t-1]
            C[t]=prevc*IFOGf[t,:,d:2*d]+IFOGf[t,:,:d]*IFOGf[t,:,3*d:]
            Ct[t]=np.tanh(C[t])
            Hout[t]=IFOGf[t,:,2*d:3*d]*Ct[t]
        if drop_prob>0:
            if drop_mask is None:
                scale=1.0/(1.0-drop_prob)
                drop_mask=(np.random.rand(*(Hout.shape))<(1-drop_prob))*scale
            #print drop_mask
            Hout*=drop_mask
        Houtstack[:,:,1:]=Hout
        if output_layer:
            Y=Houtstack.dot(Wout)
        cache['WLSTM']=WLSTM
        
        cache['X']=X
        cache['Hin']=Hin
        cache['Houtstack']=Houtstack
        cache['IFOG']=IFOG
        cache['IFOGf']=IFOGf
        cache['C']=C
        cache['Ct']=Ct
        cache['output_layer']=output_layer
        cache['c0']=c0
        cache['h0']=h0
        cache['drop_prob']=drop_prob
        if drop_prob>0:
            cache['drop_mask']=drop_mask
        if output_layer:
            cache['Wout']=Wout
            return Y,cache
        else:
            return Hout, cache
    
   
            
    @staticmethod
    def batch_backward(dY,cache):
        # dY is (n,b,v) array of backprop error message
        #unpack cache
        output_layer=cache['output_layer']
        WLSTM=cache['WLSTM']
        if output_layer:
            Wout=cache['Wout']
            dWout=np.zeros(Wout.shape)
        X=cache['X']
        Hin=cache['Hin']
        Houtstack=cache['Houtstack']
        Hout=Houtstack[:,:,1:]
        IFOG=cache['IFOG']
        IFOGf=cache['IFOGf']
        C=cache['C']
        Ct=cache['Ct']
        c0=cache['c0']
        h0=cache['h0']
        drop_prob=cache['drop_prob']
        n,b,d = Hout.shape
        v=WLSTM.shape[0] - d - 1 #vocab size
        #input_size=WLSTM.shape[0] - d - 1
        dIFOG = np.zeros(IFOG.shape)
        dIFOGf = np.zeros(IFOGf.shape)
        dWLSTM = np.zeros(WLSTM.shape)
        dHoutstack=np.zeros(Houtstack.shape)
        dHin = np.zeros(Hin.shape)
        dC = np.zeros(C.shape)
        dX = np.zeros((n,b,v))
        # backprop first last linear layer
        if output_layer:
            dHoutstack=dY.dot(Wout.T)
            dWout=np.tensordot(np.rollaxis(Houtstack,2),dY)
            dHout=dHoutstack[:,:,1:]
        else:
            dHout=dY.copy()
            
        if drop_prob>0:
            dHout*=cache['drop_mask']
        for t in reversed(xrange(n)):
            tanhCt=Ct[t]
            dIFOGf[t,:,2*d:3*d]=dHout[t]*tanhCt
            dC[t]+=(1.0-tanhCt**2)*(dHout[t]*IFOGf[t,:,2*d:3*d])
            if t>0:
                dIFOGf[t,:,d:2*d]=C[t-1]*dC[t]
                dC[t-1]+=dC[t]*IFOGf[t,:,d:2*d]
            else:
                dIFOGf[t,:,d:2*d] = c0 * dC[t]
            dIFOGf[t,:,:d]=dC[t]*IFOGf[t,:,3*d:]
            dIFOGf[t,:,3*d:]=IFOGf[t,:,:d]*dC[t]
            dIFOG[t,:,3*d:]=(1.0-IFOGf[t,:,3*d:]**2)*dIFOGf[t,:,3*d:]
            sigs=IFOGf[t,:,:3*d]
            dIFOG[t,:,:3*d]=(sigs*(1.0-sigs))*dIFOGf[t,:,:3*d]
            
            dWLSTM+=Hin[t].T.dot(dIFOG[t])
            dHin[t]=dIFOG[t].dot(WLSTM.T)
            
            dX[t]=dHin[t,:,1:v+1]
            if t>0:
                dHout[t-1]+=dHin[t,:,v+1:]
        grads={}
        grads['X']=dX
        grads['WLSTM']=dWLSTM
        grads['Hout']=dHout
        if output_layer:
            grads['Wout']=dWout
        return grads

#loss function for single layer LSTM
def LSTM_cost(X,y=None,mask=None,model=None,num_chars=None,reg=0,**kwargs):
    drop_prob=kwargs.get('drop_prob1',0.0)
    if y is None:
        drop_prob=0.0
    scores,cache=LSTM.batch_forward(X,model,drop_prob=drop_prob)
    if y is None:
        return scores
    unreg_loss,unreg_loss_grad=softmax_loss(scores,y,mask,num_chars)
    loss=unreg_loss+.5*reg*(np.sum(model['WLSTM'][1:,:]**2)+np.sum(model['Wout'][1:,:]**2))
    grad=LSTM.batch_backward(unreg_loss_grad,cache)
    grad['WLSTM'][1:,:]+=reg*model['WLSTM'][1:,:]
    grad['Wout'][1:,:]+=reg*model['Wout'][1:,:]
    return loss,grad
#loss function for two layer LSTM
def two_layer_LSTM_cost(X,y=None,mask=None,model=None,num_chars=None,reg=0,**kwargs):
    #unpack the model
    WLSTM1=model['WLSTM1']
    WLSTM2=model['WLSTM2']
    Wout=model['Wout']
    drop_prob1=kwargs.get('drop_prob1',0.0)
    drop_prob2=kwargs.get('drop_prob2',0.0)
    if y is None:
        drop_prob1=0.0
        drop_prob2=0.0
    #forward layer1
    scores1,cache1=LSTM.batch_forward(X,{'WLSTM':WLSTM1},output_layer=False,drop_prob=drop_prob1)
    #forward layer2
    scores2,cache2=LSTM.batch_forward(scores1,{'WLSTM':WLSTM2,'Wout':Wout},output_layer=True,drop_prob=drop_prob2)
    # if no labels just return the scores
    if y is None:
        return scores2
    # compute softmax probabilities
    unreg_loss,unreg_grad=softmax_loss(scores2,y,mask,num_chars)
    #regularize loss
    loss=unreg_loss+.5*reg*(np.sum(WLSTM1[1:,:]**2)+np.sum(WLSTM2[1:,:]**2)+np.sum(Wout[1:,:]**2))
    #backpropogate layer2
    grad2=LSTM.batch_backward(unreg_grad,cache2)
    #backpropogate layer1
    grad1=LSTM.batch_backward(grad2['X'],cache1)
    #regularize gradients and pack them in a dictionary
    grad2['WLSTM'][1:,:]+=reg*WLSTM2[1:,:]
    grad2['Wout'][1:,:]+=reg*Wout[1:,:]
    grad1['WLSTM'][1:,:]+=reg*WLSTM1[1:,:]
    grads={'WLSTM1':grad1['WLSTM'],'WLSTM2':grad2['WLSTM'],'Wout':grad2['Wout'],'X':grad1['X']}
    return loss,grads



#this function is aweful, but it works.
#takes a model and builds a poem by sampling from it. 
def LSTM_sample(model,temp=1.0,seed=None,max_length=1000,num_layers=2):
    #unpack the model
    dictionary=model['dictionary']
    #create models for each layer
    if num_layers==1:
        d1=model['WLSTM'].shape[1]/4
        d2=1
    v=model['Wout'].shape[1] #vocab size
    if num_layers==2:
        model1={}
        model2={}
        model1['WLSTM']=model['WLSTM1']
        model2['WLSTM']=model['WLSTM2']
        model2['Wout']=model['Wout']
        d1=model['WLSTM1'].shape[1]/4 # size of hidden layer 1
        d2=model['WLSTM2'].shape[1]/4 # size of hidden layer 2
    #d1=model['WLSTM1'].shape[1]/4 # size of hidden layer 1
    #d2=model['WLSTM2'].shape[1]/4 # size of hidden layer 2
    rev_dict=reverse_dictionary(dictionary) #create the inverse dicitonary
    s='' # the output string
    if seed is None: #initialize first character to random letter it not specified
        seed=''
        while not seed.isalpha():
            seed=np.random.choice(letters,p=letter_distribution)
    s+=seed
    current_character=seed[-1]
    x=np.zeros((1,1,v))
    x[:,:,dictionary[seed[0]]]=1
    i=len(seed)
    h1=np.zeros((1,d1))
    h2=np.zeros((1,d2))
    c1=np.zeros((1,d1))
    c2=np.zeros((1,d2))
    # I hate that this seems so hacky. I should make this more modular 
    if num_layers==2:
        #prime the model using the seed
        for j in xrange(1,len(seed)):
            layer1_scores,cache1=LSTM.batch_forward(x,model1,output_layer=False,h0=h1,c0=c1)
            Y,cache2=LSTM.batch_forward(layer1_scores,model2,h0=h2,c0=c2)
            c1=cache1['C'][0]
            c2=cache2['C'][0]
            h1=cache1['Houtstack'][0,:,1:]
            h2=cache2['Houtstack'][0,:,1:]
            x=np.zeros((1,1,v))
            x[:,:,dictionary[seed[j]]]=1
        #sample using the model
        while current_character!='\t' and i<max_length:
            layer1_scores,cache1=LSTM.batch_forward(x,model1,output_layer=False,h0=h1,c0=c1)
            Y,cache2=LSTM.batch_forward(layer1_scores,model2,h0=h2,c0=c2)
            c1=cache1['C'][0]
            c2=cache2['C'][0]
            h1=cache1['Houtstack'][0,:,1:]
            h2=cache2['Houtstack'][0,:,1:]
            Y/=temp
            probs=np.exp(Y-np.amax(Y))
            probs/=np.sum(probs)
            #print v,probs.shape
            smpl=np.random.choice(v,p=probs[0,0,:])
            current_character=rev_dict[smpl]
            s+=current_character
            x=np.zeros((1,1,v))
            x[:,:,smpl]=1
            i+=1
        return s
    if num_layers==1:
        #prime model using the seed
        for j in xrange(1,len(seed)):
            Y, cache=LSTM.batch_forward(x,model,h0=h1,c0=c1)
            c1=cache['C'][0]
            h1=cache['Houtstack'][0,:,1:]
            x=np.zeros((1,1,v))
            x[:,:,dictionary[seed[j]]]=1
        #sample from the model
        while current_character!='\t' and i<max_length:
            Y,cache=LSTM.batch_forward(x,model,h0=h1,c0=c1)
            c1=cache['C'][0]
            h1=cache['Houtstack'][0,:,1:]
            Y/=temp
            probs=np.exp(Y-np.amax(Y))
            probs/=np.sum(probs)
            #print v,probs.shape
            smpl=np.random.choice(v,p=probs[0,0,:])
            current_character=rev_dict[smpl]
            s+=current_character
            x=np.zeros((1,1,v))
            x[:,:,smpl]=1
            i+=1
        return s
            

            

        
        
    
        
    
        
        
        
        
    
    
    
    