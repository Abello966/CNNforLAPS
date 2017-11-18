from Model.model import LasagneModel
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, MaxPool2DLayer, DropoutLayer
from lasagne.nonlinearities import softmax, rectify
from lasagne.regularization import l2, regularize_network_params
import numpy as np
from Util.image_processing import minibatch_iter, minibatch_iter_balanced, augmentation_iter

class CNN(LasagneModel):
    
    def __init__(self, config, name):
        self.config = config
        self.name = name
        model = self.compile_cnn()
        self.network, self.train_fn, self.test_fn = model

    def train(self, Xtrain, ytrain, Xtest, ytest, saveoutput):
        #Retrieve info
        train_fn = self.train_fn
        test_fn = self.test_fn
        epochs = self.config['epochs']
        batchsize = self.config['batchsize']
        stopping = self.config['stopping']
        lrate = self.config['learning_rate']

        print("Begin training")
        train_loss = list()
        test_loss = list()
        last_loss = float('inf')
        for epoch in range(epochs):
                epoch_loss = 0
                batches = 0
                for batch in minibatch_iter(ytrain, batchsize):
                    Xbatch = Xtrain[batch]
                    ybatch = ytrain[batch]
                    Xaugm = next(augmentation_iter(Xbatch))
                    epoch_loss += train_fn(Xaugm, ybatch, lrate)
                    batches += 1
        
                epoch_loss = epoch_loss / batches
                train_loss.append(epoch_loss)
  
                val_loss, val_acc = test_fn(Xtest, ytest)
                test_loss.append(val_loss)

                if (epoch_loss > stopping * last_loss):
                    break
                last_loss = epoch_loss
        self.train_loss = train_loss
        self.test_loss = test_loss
        if saveoutput:
            self.save_results(train_loss, test_loss)
        return val_acc

    def score(self, Xtest, ytest):
        test_loss, test_acc = self.test_fn(Xtest, ytest)
        return test_acc

    def build_cnn(self, input_var=None):
        #Retrieve info
        imsize = self.config['imsize']
        nclasses = self.config['nclasses']
        nfilters = self.config['nfilters']

        network = InputLayer(shape=(None, 1, imsize, imsize), input_var=input_var) 
        inputsize = imsize
        while inputsize > 8:
            network = Conv2DLayer(network, num_filters=nfilters, filter_size=(5,5), nonlinearity=rectify)
            network = MaxPool2DLayer(network, pool_size=(2,2))
            inputsize = inputsize / 2
    
        #normal LP
        network = DropoutLayer(network, p=.5)
        network = DenseLayer(network, num_units=256, nonlinearity=rectify)
        network = DropoutLayer(network, p=.5)
        network = DenseLayer(network, num_units=nclasses, nonlinearity=softmax)
        return network

    def compile_cnn(self):
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        lrate = T.fscalar('alpha')

        network = self.build_cnn(input_var=input_var)

        #loss function
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        #parameter updates
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, learning_rate=lrate)

        #compile train function
        train_fn = theano.function([input_var, target_var, lrate], loss, updates=updates)

        #compile test function
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var))
        test_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        return network, train_fn, test_fn

        
