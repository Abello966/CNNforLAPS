##Constructing Lasagne network, compiling train and test functions
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano import shared
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import GlorotUniform
from Model.model import LasagneModel
from image_processing import *

class MixedCNN(LasagneModel):

    #available parts for training
    parts = ["shallow", "deep", "all"]

    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.test_loss = list()
        self.train_loss = list()
        model = self.compile_mixed_cnn()
        self.network, self.train_fns, self.test_fn = model

    def train(self, Xtrain, Xfeatstrain, ytrain, Xtest, Xfeatstest, ytest, saveoutput):
        #config[order] defines order of training
        if "order" in self.config.keys():
            for part in self.config['order']:
                print("Training {}".format(part))
                self.train_part(Xtrain, Xfeatstrain, ytrain, Xtest, Xfeatstest, ytest, part)

        #if not train once all parts together
        else:
            print("Training once all parts")
            self.train_part(Xtrain, Xfeatstrain, ytrain, Xtest, Xfeatstest, ytest, "all")
    
        if saveoutput:
            self.save_results(self.test_loss, self.train_loss)

    def score(self, Xtest, Xtestfeats, ytest):
        test_loss, test_acc = self.test_fn(Xtest, Xtestfeats, ytest)
        return test_acc
            
    def train_part(self, Xtrain, Xfeatstrain, ytrain, Xtest, Xfeatstest, ytest, part):
        if part not in self.parts:
            print("Reminder - Available parts: {}".format(self.parts))
            raise KeyError("part {} not recognized by MixedCNN".format(part))
        
        #each part may have a config of their own
        if part in self.config.keys():
            conf = self.config[part]
        else:
            conf = self.config
     
        res = MixedCNN.train_mixed_cnn(Xtrain, Xfeatstrain, ytrain, Xtest, Xfeatstest,
                                       ytest, self.train_fns[part], self.test_fn, conf)
        self.train_loss += res[0]
        self.test_loss += res[1]
 
    def build_mixed_cnn(self, input_var=None, input_feats=None):
        #retrieve info
        imsize = self.config['imsize']
        nclasses = self.config['nclasses']
        nfilters = self.config['nfilters']
        nfeats = self.config['nfeats']

        # deep model
        deep = InputLayer(shape=(None, 1, imsize, imsize), input_var=input_var) 
        inputsize = imsize
        while inputsize > 8:
            deep = Conv2DLayer(deep, num_filters=nfilters, filter_size=(5,5), nonlinearity=rectify)
            deep = MaxPool2DLayer(deep, pool_size=(2,2))
            inputsize = inputsize / 2
        
        deep = DropoutLayer(deep, p=.5)
        deep = DenseLayer(deep, num_units=256, nonlinearity=rectify)
 
        # 'shallow' model
        shallow = InputLayer(shape=(None, nfeats), input_var=input_feats)
        network = ConcatLayer([deep, shallow])

        network = DropoutLayer(network, p=.5)
        network = DenseLayer(network, num_units=nclasses, nonlinearity=softmax)
        return network

    def compile_mixed_cnn(self):
        #retrieve info
        nfeats = self.config['nfeats']
    
        input_var = T.tensor4('inputs')
        input_feats = T.dmatrix('feats')
        target_var = T.ivector('targets')
        lrate = T.fscalar('alpha')

        network = self.build_mixed_cnn(input_var=input_var, input_feats=input_feats)
    
        #loss function
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        #parameter updates
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, learning_rate=lrate)

        #compile train functions
        train_all_fn = theano.function([input_var, input_feats, target_var, lrate], loss, updates=updates)
        
        #train only deep part
        #impt is the last dense weight matrix
        impt = params[-2] 
        mask = np.ones(impt.get_value().shape, dtype='float32')
        mask[-1 * nfeats::] *= 0
 
        #element wise multiplication cancels updates to shallow part
        updates[impt] = updates[impt] * mask
        train_deep_fn = theano.function([input_var, input_feats, target_var, lrate], loss, updates=updates)

        #train only shallow part
        updates = lasagne.updates.adagrad(loss, params[-2::], learning_rate=lrate)
        mask = np.zeros(impt.get_value().shape, dtype='float32')
        mask[-1 * nfeats::] += 1

        #elemwise multiplication cancels updates to deep part
        updates[impt] = updates[impt] * mask
        train_shallow_fn = theano.function([input_var, input_feats, target_var, lrate], loss, updates=updates)
        train_fns = {'all': train_all_fn, "deep": train_deep_fn, "shallow": train_shallow_fn}

        #compile test function
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var))
        test_fn = theano.function([input_var, input_feats, target_var], [test_loss, test_acc])
        return network, train_fns, test_fn
    
    def train_mixed_cnn(Xtrain, Xfeatstrain, ytrain, Xtest, Xfeatstest, ytest, train_fn, test_fn, config):
        #retrieve info
        batchsize = config['batchsize']
        epochs = config['epochs']
        stopping = config['stopping']
        lrate = config['learning_rate']
        
        print("Begin training")
        train_loss = list()
        test_loss = list()
        last_loss = float('inf')
        for epoch in range(epochs):
            print("epoch {}".format(epoch))
            epoch_loss = 0
            batches = 0
            for batch in minibatch_iter(ytrain, batchsize):
                Xbatch = Xtrain[batch]
                Xfeatsbatch = Xfeatstrain[batch]
                ybatch = ytrain[batch]
                Xaugm = next(augmentation_iter(Xbatch))
                epoch_loss += train_fn(Xaugm, Xfeatsbatch, ybatch, lrate)
                batches += 1
     
            epoch_loss = epoch_loss / batches
            train_loss.append(epoch_loss)
      
            val_loss, val_acc = test_fn(Xtest, Xfeatstest, ytest)
            test_loss.append(val_loss)
    
            if (epoch_loss > stopping * last_loss):
                break
            last_loss = epoch_loss
        return val_acc
