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
pos_imsize = [32, 64, 96, 128]
pos_nfilters = [4, 8, 16, 32]

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
        if (nfilters == 32 and imsize == 128):
            continue #not enough memory
        name = "{:d}x{:d} images and {:d} filters".format(imsize, imsize, nfilters)
        print("------------" * 5)
        print(name)
        config['imsize'] = imsize
        config['nfilters'] = nfilters
        acc = list()
        for i in range(5):
            network = CNN(config, name)
            acc.append(network.train(Xtr_resh, ytr_resh, Xval_resh, yval_resh, True))
            del network

        print("Training ended")
        print("Validation acc: {} +- {}".format(np.mean(acc), np.var(acc)))
        
        results[(imsize, nfilters)] = np.mean(acc)
        if (np.mean(acc) > winning_cnn_acc):
            winning_cnn = (imsize, nfilters)
            winning_cnn_acc = np.mean(acc)
            
        print("------------" * 5)

    del Xtr_resh
    del Xval_resh
    del ytr_resh
    del yval_resh


#Print Results
print("Results: ")
print(results)

#Get best network
imsize, nfilters = winning_cnn
name = "{:d}x{:d} images and {:d} filters".format(imsize, imsize, nfilters)
print("Best network: {:d} image size and {:d} filters".format(imsize, nfilters))
print("Re-training")
config['imsize'] = imsize
config['nfilters'] = nfilters

#Separate train, test 
train_idx, test_idx = stratified_split(X.shape[0], config['test_size'],
                       random_state=0, stratified=config['stratified'])

Xtrain = X[train_idx]
ytrain = y[train_idx]
Xtest = X[test_idx]
ytest = y[test_idx]

Xtr_resh, ytr_resh = reshape_dataset(Xtrain, ytrain, imsize)
Xtest_resh, ytest_resh = reshape_dataset(Xtest, ytest, imsize)

acc = list()
for i in range(5):
    network = CNN(config, name)
    acc.append(network.train(Xtr_resh, ytr_resh, Xtest_resh, ytest_resh, True))
    

print("Final test mean acc and var:{} +- {}".format(np.mean(acc), np.var(acc)))
