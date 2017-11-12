#Libraries
import os, sys
import numpy as np

#Other files
from Model.CNN import CNN
from Util.image_processing import stratified_split, reshape_dataset

#constants
MODELS = ['cnn', 'logreg', 'mixedcnn']

## Read cmd line 
import argparse

parser = argparse.ArgumentParser(description=
                                "Perform training of models on dataset")
parser.add_argument('--images', dest='images', metavar='in', type=str, nargs=1,
                    help="input grayscale numpy images")
parser.add_argument('--feats', dest='feats', metavar='ft', type=str, nargs=1,
                    help="numpy matrix of features")
parser.add_argument('output', metavar='out', type=str, nargs=1,
                    help="directory for writing output")
parser.add_argument('model', choices=MODELS)
parser.add_argument('--name', dest='name', metavar='name', type=str, nargs=1,
                    help="name of the model")

args = parser.parse_args()
if (args.model == 'cnn' or args.model == 'mixedcnn'):
    if (args.images is None):
        print("{} requires images".format(args.model))
        parser.print_usage()
        sys.exit(1)

if (args.model == 'logreg' or args.model == 'mixedcnn'):
    if (args.feats is None):
        print("{} requires features".format(args.model))
        parser.print_usage()
        sys.exit(1)

## Define constants - TODO: read config by file
config = {}
config['verbose'] = True
config['epochs'] = 100
config['learning_rate'] = 0.001
config['batchsize'] = 500
config['stopping'] = 1.25
config['test_size'] = 0.1
config['stratified'] = False

## Define variables of experiment
pos_imsize = [32]
pos_nfilters = [16]

## Load Data
images = np.load(args.images[0])
X = images['arr_0']
y = images['arr_1']

config['nclasses'] = len(set(y))

#Some preparing 
RESULTSPATH = args.output[0]
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

#Separate train, test 
train_idx, test_idx = stratified_split(X.shape[0], config['test_size'],
                       random_state=0, stratified=config['stratified'])

Xtrain = X[train_idx]
ytrain = y[train_idx]
Xtest = X[test_idx]
ytest = y[test_idx]

#Separate train, val
train_idx, val_idx = stratified_split(Xtrain.shape[0], config['test_size'],
                      random_state=0, stratified=config['stratified'])

Xval = Xtrain[val_idx]
yval = ytrain[val_idx]
Xtrain = Xtrain[train_idx]
ytrain = ytrain[train_idx]

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
        name = "{:d}x{:d} images and {:d} filters".format(imsize, imsize, nfilters)
        print("------------" * 5)
        print(name)
        config['imsize'] = imsize
        config['nfilters'] = nfilters
        network = CNN(config, name)
        acc = network.train(Xtr_resh, ytr_resh, Xval_resh, yval_resh, True)
        print("Training ended")
        print("Epochs: {:d}".format(len(network.train_loss)))
        print("Validation acc: {}".format(acc))
        
        results[(imsize, nfilters)] = acc
        if (acc > winning_cnn_acc):
            winning_cnn = (imsize, nfilters)
            winning_cnn_acc = acc
            
        print("------------" * 5)

        #Let GC know it can free this now
        del network
    del Xtr_resh
    del Xval_resh
    del ytr_resh
    del yval_resh


#Print Results
print("Results: ")
print(results)

#Get best network
"""
imsize, nfilters = winning_cnn
print("Best network: {:d} image size and {:d} filters".format(imsize, nfilters))
print("Re-training")

#Separate train, test 
Xtrain, ytrain, Xtest, ytest = stratified_split(Xfilter, yfilter, test_size, random_state=0, stratified=stratified)
ytrain = ytrain.astype("uint8")
ytest = ytest.astype("uint8")

Xtr_resh, ytr_resh = reshape_dataset(Xtrain, ytrain, imsize)
Xtest_resh, ytest_resh = reshape_dataset(Xtest, ytest, imsize)
network, train_fn, test_fn = compile_cnn(imsize, nclasses, nfilters, learning_rate)
train_loss, test_loss, acc = train_cnn(Xtr_resh, ytr_resh, Xtest_resh, ytest_resh, train_fn, test_fn, epochs, batchsize, stopping)

print("Final test accuracy:" + str(acc))
"""
















