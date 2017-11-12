import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer
from lasagne.nonlinearities import softmax
from lasagne.regularization import l2, regularize_network_params
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from Model.model import LasagneModel
from image_processing import minibatch_iter, minibatch_iter_balanced

class LogReg(LasagneModel):

    def __init__(self, config, name):
        self.config = config
        self.name = name
        self.augmented = False
        model = self.build_and_compile_logreg()
        self.train_fn, self.test_fn = model

    def train(self, Xtrain, ytrain, Xtest, ytest, saveoutput=False):
        #retrieve info
        epochs = self.config['epochs']
        batchsize = self.config['batchsize']
        stopping = self.config['stopping']

        #variables
        train_loss = list()
        test_loss = list()
        last_loss = float('inf')
        for epoch in range(epochs):
            now = time.time()
            epoch_loss = 0
            batches = 0
            for batch in minibatch_iter_balanced(ytrain, batchsize):
                Xbatch = Xtrain[batch]
                ybatch = ytrain[batch]
                epoch_loss += self.train_fn(Xbatch, ybatch)
                batches += 1

            epoch_loss = epoch_loss / batches
            train_loss.append(epoch_loss)
            
            val_loss, val_acc = self.test_fn(Xtest, ytest)
            test_loss.append(val_loss)

            if (val_loss > stopping * last_loss):
                break
            last_loss = val_loss
            if self.config['verbose']:
                elapsed = time.time() - now
                print("epoch {}: {}s".format(epoch, elapsed), 
                     file=sys.__stdout__)
        
        if saveoutput:
            self.save_results(train_loss, test_loss)

        return val_acc
        

    def score(self, Xtest, ytest):
        test_loss, test_acc = self.test_fn(Xtest, ytest)
        return test_acc
        

    ## auxiliary functions

    def build_and_compile_logreg(self):
        #retrieve info
        nfeats = self.config['nfeats']
        nclasses = self.config['nclasses']
        learning_rate = self.config['learning_rate']
    
        # declare model 
        input_var = T.dmatrix('inputs')
        target_var = T.ivector('targets')
        network = InputLayer(shape=(None, nfeats), input_var=input_var)
        network = DenseLayer(network, num_units=nclasses, nonlinearity=softmax)

        # loss function 
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        # parameter updates
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.adagrad(loss, params, learning_rate=learning_rate)

        #compile train function
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        #compile test function
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
        test_loss = test_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var))
        test_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        return train_fn, test_fn

 

