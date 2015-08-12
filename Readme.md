# PoetRNN

This project is a *Python* implementation of a LSTM recurrent neural network designed to learn and produce short verse poetry. It stemmed from me reading Andrej Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/), thinking it was too good to be true, and wanting to try it for myself. Since I am a bit of python novice I decided I would learn more if I wrote things from scratch, and so here we are. Despite being written with poetry in mind it is likely flexible enough to learn and produce other types of data (In fact, if you think of something cool and creative to do with it I would love to hear about it). If you are interested in a more detailed description as well as seeing some of the outputs of the model check out my [blog post](http://sballas8.github.io/2015/08/11/Poet-RNN.html) 

One key difference between this implementation and Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) is that the input is not treated as a long text file. This is important for learning poetry because there is clear structure to the data (i.e. each poem is a distinct piece of data) that is ignored if you just concatenate it all to a single text file. It also makes sampling nicer because you can tell the sampler that you want 14 poems instead of having to decide in advance that you want to sample 10000 characters and probably having the sample stop in the middle of a poem. 


##Usage

###Data

The input data should be stored in the `data` file and should consist of a `.csv` file in which each row has a single column consisting of a string containing a poem (or whatever data you want to learn from). I have included two files `haikus.csv` and `limericks.csv` which contain approximately 8000 and 90000 haikus and limericks, respectively. 

Feel free to play around with your own data. It should work provided it is formatted as above. 


###The Model

The model for training is an LSTM recurrent neural network of either 1 or 2 hidden layers. The forward and backward passes are batched and the implementation is a (very slight) modification of [Andrej Karpathy's gist](https://gist.github.com/karpathy/587454dc0146a6ae21fc)

###Training

If you have some data contained in a file `data.csv` and you want to get started quickly, you can train a model using the `train.py` file with its default settings. Just run 
```bash
$ python train.py -d 'data.csv'
```

If you want to explore the settings for training just type 
```bash
$ python train.py -h
```

In each training step the trainer will perform an rmsprop update using a mini-batch of poems (the default size of which is 100 and it can be adjusted). Periodically, the trainer will tell you how long (on average) the batches are taking. After each epoch, the model will tell you the current loss, the training accuracy, and the validation accuracy. It will also write the current model (serialized using pickle) to a file in the `cv` folder if it has a higher validation accuracy than the previous best model. These models will be used to sample after training.

###Sampling  

Once you have a model trained you can use `sample.py` to sample from it. If you have a model stored in `cv/model.p` then you can sample from it using the default parameters with the command
```bash
$ python sample.py -m 'cv/model.p'
```
This will sample 10 poems using the model you created. 

There are a few parameters you can play around with. The first, and simplest parameter, is the `-n` argument which specifies the number of poems you want `sample.py` to produce. 

Second, you can seed your model with some text by specifying a string using the `-s` argument. This will feed the characters of your string forward through the model and use them as the initial characters of each poem. If no seed is provided `sample.py` will just pick a random letter to start your poem with and sample the rest. 

The third, and probably most fun, parameter to play with is the 'temperature' given by the `-t` argument. If you are unfamiliar with temperature, here is an extremely unsatisfying explanation. You should think of temperature as a positive real number which if close to 0 makes your model more conservative and if it is large will make your model more creative. 

There are a few other arguments which you can use, which control whether the output of the sampler is written to a file or not. You can learn more about this and other arguments using the 
```bash
$ python sample.py -h
```
command.

Hopefully you can find some good data to play with using this model. Good luck!!

