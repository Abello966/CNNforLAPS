##Constructing Lasagne network, compiling train and test functions
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lasagne.nonlinearities import *
from lasagne.init import GlorotUniform

from image_processing import *

def build_cnn(imsize, nfilters, nclasses, input_var=None):
    network = InputLayer(shape=(None, 1, imsize, imsize), input_var=input_var)
    
    inputsize = imsize
    while inputsize > 8:
        network = Conv2DLayer(network, num_filters=nfilters, filter_size=(5,5), nonlinearity=rectify)
        network = MaxPool2DLayer(network, pool_size=(2,2))
        inputsize = inputsize / 2
    
    #normal MLP
    network = DropoutLayer(network, p=.5)
    network = DenseLayer(network, num_units=256, nonlinearity=rectify)
    network = DropoutLayer(network, p=.5)
    network = DenseLayer(network, num_units=nclasses, nonlinearity=softmax)
    return network

def compile_cnn(imsize, nfilters, nclasses, learning_rate):
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = build_cnn(input_var=input_var, imsize=imsize, nfilters=nfilters, nclasses=nclasses)
    
    #loss function
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    #parameter updates
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
    return network, train_fn, test_fn


    
def train_cnn(Xtrain, ytrain, Xtest, ytest, train_fn, test_fn, epochs, batchsize, stopping):
    print("Begin training")
    train_loss = list()
    test_loss = list()
    last_loss = float('inf')
    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0
        for batch in minibatch_iter(Xtrain, ytrain, batchsize):
            Xbatch, ybatch = batch
            Xaugm = next(augmentation_iter(Xbatch))
            epoch_loss += train_fn(Xaugm, ybatch)
            batches += 1
        
        epoch_loss = epoch_loss / batches
        train_loss.append(epoch_loss)
  
        val_loss, val_acc = test_fn(Xtest, ytest)
        test_loss.append(val_loss)

        if (epoch_loss > stopping * last_loss):
            break
        last_loss = epoch_loss
    return train_loss, test_loss, val_acc

def save_results(train_loss, test_loss, imsize, nfilters):
    print("Saving results")
    plt.plot(np.arange(len(train_loss)), train_loss, label="train loss")
    plt.plot(test_loss, label="test loss")
    plt.yscale('log')
    plt.title("Imagem: {:d} Filtros: {:d}".format(imsize, nfilters))
    plt.legend()
    plt.savefig("plot_size{:d}_filters{:d}.png".format(imsize, nfilters))
    np.savez("data_size{:d}_filters{:d}".format(imsize, nfilters), train_loss=np.array(train_loss), test_loss=np.array(test_loss))

