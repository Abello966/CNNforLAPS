#Libraries
import os, sys
import numpy as np

#Other files
from Model.LogReg import LogReg
from image_processing import stratified_split

#Constants
MODELS = ['cnn', 'logreg', 'mixedcnn']

## Read cmd line 
import argparse

parser = argparse.ArgumentParser(description=
                                "Perform training of models on dataset")
parser.add_argument('--input', dest='input', metavar='in', type=str, nargs=1,
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

## Define constants - TO-DO: read config by file
config = {}
config['verbose'] = False
config['epochs'] = 10000
config['learning_rate'] = 0.01
#config['batchsize'] = 1000
config['stopping'] = 1.2
config['test_size'] = 0.1
config['stratified'] = False

## Define variables of experiment
#pos_imsize = [32]
#pos_nfilters = [16]

## Load Data
feats = np.load(args.feats[0])
X = feats['arr_0']
y = feats['arr_1'].astype("int32")

## Count number of classes and feats
config['batchsize'] = X.shape[0]
config['nclasses'] = len(set(y))
config['nfeats'] = X.shape[1]
    
#Some preparing 
RESULTSPATH = args.output[0]
if not os.path.exists(RESULTSPATH):
    os.makedirs(RESULTSPATH)
try:
    os.chdir(RESULTSPATH)
except:
    print("Cant go into results directory")
    sys.exit(1)
try:
    sys.stdout = open("results.txt", "w")
except:
    print("Cant open results text-file")
    sys.exit(1)

#Separate train, test 
train_idx, test_idx = stratified_split(X.shape[0], config['test_size'], 
                                       random_state=0,
                                       stratified=config['stratified'])

Xtrain = X[train_idx]
Xtest = X[test_idx]
ytrain = y[train_idx]
ytest = y[test_idx]

#Separate train, val
#not necessary as we re not optimizing hyperparameters
"""
dataset = stratified_split(Xtrain, ytrain, config['test_size'],
                           random_state=0, stratified=config['stratified'])
Xtrain, ytrain, Xval, yval = dataset

ytrain = ytrain.astype("uint8")
ytest = ytest.astype("uint8")
yval = yval.astype("uint8")
"""

model = LogReg(config, name)
model.train(Xtrain, ytrain, Xtest, ytest, saveoutput=True)
print("Final Results: ")
print(model.score(Xtest, ytest))

