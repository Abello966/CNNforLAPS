#Libraries
import os, sys
import numpy as np
from sklearn import model_selection

#Other files
from Util.image_processing import stratified_split, reshape_dataset
from Model.MixedCNN import MixedCNN

##Constants
MODELS = ['cnn', 'logreg', 'mixedcnn']


## Read cmd line 
import argparse

parser = argparse.ArgumentParser(description=
                                "Perform training of models on dataset")
parser.add_argument('--images', dest='input', metavar='in', type=str, nargs=1,
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
    if (args.input is None):
        print("{} requires images".format(args.model))
        parser.print_usage()
        sys.exit(1)

if (args.model == 'logreg' or args.model == 'mixedcnn'):
    if (args.feats is None):
        print("{} requires features".format(args.model))
        parser.print_usage()
        sys.exit(1)

if args.name is not None:
    name = args.name[0]
else:
    name = args.output[0]

## Define config - TO-DO: read config by file
config = {}
config['test_size'] = 0.1
config['stratified'] = False
config['order'] = ['all']

#config for all training
config_all = {}
config_all['verbose'] = True
config_all['epochs'] = 1000
config_all['learning_rate'] = 0.001
config_all['batchsize'] = 500
config_all['stopping'] = float('inf')
config['all'] = config_all


# config for shallow training
config_sh = {}
config_sh['verbose'] = True
config_sh['epochs'] = 5000
config_sh['learning_rate'] = 0.01
config_sh['batchsize'] = 3000
config_sh['stopping'] = float('inf')
config['shallow'] = config_sh

# config for deep learning
config_dp = {}
config_dp['verbose'] = True
config_dp['epochs'] = 100
config_dp['learning_rate'] = 0.001
config_dp['batchsize'] = 500
config_dp['stopping'] = float('inf')
config['deep'] = config_dp

## Define variables of experiment
pos_imsize = [32]
pos_nfilters = [32]

## Load data
files = np.load(args.input[0])
extra = np.load(args.feats[0])
X = files['arr_0']
y = files['arr_1'].astype("int32")
Xfeats = extra['arr_0']

## update config
config['nclasses'] = len(set(y))
config['nfeats'] = Xfeats.shape[1]

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
train_idx, test_idx = stratified_split(X.shape[0], config['test_size'], random_state=0, stratified=config['stratified'])
Xtrain = X[train_idx]
Xtrainfeats = Xfeats[train_idx]
ytrain = y[train_idx]

Xtest = X[test_idx]
Xtestfeats = Xfeats[test_idx]
ytest = y[test_idx]

#Separate train, val
train_idx, val_idx = stratified_split(Xtrain.shape[0], config['test_size'], random_state=0, stratified=config['stratified'])
Xval = Xtrain[val_idx]
Xvalfeats = Xtrainfeats[val_idx]
yval = ytrain[val_idx]

Xtrain = Xtrain[train_idx]
Xtrainfeats = Xtrainfeats[train_idx]
ytrain = ytrain[train_idx]

#Do experiments
winning_cnn = (pos_imsize[0], pos_nfilters[0])
winning_cnn_acc = 0
results = {}
for imsize in pos_imsize:
    #Reshapes dataset
    Xtr_resh, ytr_resh = reshape_dataset(Xtrain, ytrain, imsize)
    Xval_resh, yval_resh = reshape_dataset(Xval, yval, imsize)
    config['imsize'] = imsize

    for nfilters in pos_nfilters:
        print("------------" * 5)
        name = "{:d}x{:d}_{:d}filters_{}_feats".format(imsize, imsize, nfilters, name)
        print(name)
        print("Compiling CNN")
        config['nfilters'] = nfilters
        acc = 0
        for i in range(0):
            net = MixedCNN(config, name)
            net.train(Xtr_resh, Xtrainfeats, ytrain, Xval_resh, Xvalfeats, yval, True)
            acc = net.score(Xval_resh, Xvalfeats, yval_resh)
            del net

        print("Training ended")
        print("Validation acc: {}".format(acc))
        print("------------" * 5)

        results[(imsize, nfilters)] = acc
        if winning_cnn_acc > acc:
            winning_cnn = (imsize, nfilters)
            winning_cnn_acc = acc
            
        #Let GC know it can free this now

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
train_idx, test_idx = stratified_split(X.shape[0], config['test_size'], random_state=0, stratified=config['stratified'])
Xtrain = X[train_idx]
Xtrainfeats = Xfeats[train_idx]
ytrain = y[train_idx]
Xtest = X[test_idx]
Xtestfeats = Xfeats[test_idx]
ytest = y[test_idx]

ytrain = ytrain.astype("uint8")
ytest = ytest.astype("uint8")

Xtr_resh, ytr_resh = reshape_dataset(Xtrain, ytrain, imsize)
Xtest_resh, ytest_resh = reshape_dataset(Xtest, ytest, imsize)

config['imsize'] = imsize
config['nfilters'] = nfilters

network = MixedCNN(config, "Mixed - All")
acc = network.train(Xtr_resh, Xtrainfeats, ytr_resh, Xtest_resh, Xtestfeats, ytest_resh, False)
print("Final test accuracy:" + str(acc))
