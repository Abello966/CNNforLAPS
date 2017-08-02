#to-do: read from cmd line argument
DATAPATH = "LAPSClean.npz"

#Libraries
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import model_selection

#Other files
from model import *

## Define constants
verbose = True
epochs = 100
learning_rate = 0.001
batchsize = 500
stopping = 1.25
nfolds = 10

## Define variables of experiment
pos_imsize = [32]
pos_nfilters = [16]

## Load Images
files = np.load(DATAPATH)
X = files['arr_0']
y = files['arr_1']

#Some preparing 
RESULTSPATH = DATAPATH.split(".")[0]
if not os.path.exists(RESULTSPATH):
    os.makedirs(RESULTSPATH)

try:
    os.chdir(RESULTSPATH)
except:
    print("Cant go to results directory")
    sys.exit(1)

try:
    sys.stdout = open("results.txt", "w")
except:
    print("Cant open results text-file")
    sys.exit(1)

##Dataset Selection - only classes above 100 examples
density, cls = np.histogram(y, bins=max(y) + 1)
Xstack = np.vstack((X,y)).transpose()
Xfilter = np.array(list(filter(lambda x: density[x[1]] > 100, Xstack)))
yfilter = Xfilter[:, 1]
Xfilter = Xfilter[:, 0]
nclasses = len(set(yfilter))

#Separate train, test 
Xtrain, ytrain, Xtest, ytest = stratified_split(Xfilter, yfilter, nfolds, random_state=0)

#Separate train, val
Xtrain, ytrain, Xval, yval = stratified_split(Xtrain, ytrain, nfolds, random_state=0)

ytrain = ytrain.astype("uint8")
ytest = ytest.astype("uint8")
yval = yval.astype("uint8")


#Do experiments
winning_cnn = None
winning_cnn_acc = 0
results = {}
for imsize in pos_imsize:
    #Reshapes dataset
    Xtr_resh, ytr_resh = reshape_dataset(Xtrain, ytrain, imsize)
    Xval_resh, yval_resh = reshape_dataset(Xval, yval, imsize)

    for nfilters in pos_nfilters:
        print("------------" * 5)
        print("Experiment: {:d}x{:d} images and {:d} filters".format(imsize, imsize, nfilters))
        print("Compiling CNN")
        network, train_fn, test_fn = compile_cnn(imsize, nclasses, nfilters, learning_rate)
        train_loss, val_loss, acc = train_cnn(Xtr_resh, ytr_resh, Xval_resh, yval_resh, train_fn, test_fn, epochs, batchsize, stopping)
        
        print("Training ended")
        print("Epochs: {:d}".format(len(train_loss)))
        print("Validation acc: " + str(acc))
        
        save_results(train_loss, val_loss, imsize, nfilters)
        
        results[(imsize, nfilters)] = acc
        if (acc > winning_cnn_acc):
            winning_cnn = (imsize, nfilters)
            winning_cnn_acc = acc
            
        print("------------" * 5)

        #Let GC know it can free this now
        del train_fn
        del test_fn
        del network
    del Xtr_resh
    del Xval_resh
    del ytr_resh
    del yval_resh


#Print Results
print("Results: ")
print(results)

#Get best network
imsize, nfilters = winning_cnn
print("Best network: {:d} image size and {:d} filters".format(imsize, nfilters))
print("Re-training")

#Separate train, test 
Xtrain, ytrain, Xtest, ytest = stratified_split(Xfilter, yfilter, nfolds, random_state=0)
ytrain = ytrain.astype("uint8")
ytest = ytest.astype("uint8")

Xtr_resh, ytr_resh = reshape_dataset(Xtrain, ytrain, imsize)
Xtest_resh, ytest_resh = reshape_dataset(Xtest, ytest, imsize)
network, train_fn, test_fn = compile_cnn(imsize, nclasses, nfilters, learning_rate)
train_loss, test_loss, acc = train_cnn(Xtr_resh, ytr_resh, Xtest_resh, ytest_resh, train_fn, test_fn, epochs, batchsize, stopping)

print("Final test accuracy:" + str(acc))

















