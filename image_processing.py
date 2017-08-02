import numpy as np
import random
from scipy import misc
from sklearn import model_selection

#Image resizing
#Gives a Tuple corresponding to a symmetric padding
def symm_pad(num, size):
    return (int(np.floor((size - num) / 2)), int(np.ceil((size - num) / 2)))

def reshape_image(image, target):
    maxind = np.argmax(image.shape)
    #Shrink maintaning ratio if necessary
    if image.shape[maxind] > target:
        image = misc.imresize(image, float(target) / image.shape[maxind])
    image = np.pad(image, (symm_pad(image.shape[0], target), symm_pad(image.shape[1], target)), "constant")
    return image

def reshape_dataset(X, y, imsize):
    #Apply reshaping
    Xreshaped = np.array(list(map(lambda x: reshape_image(x, imsize), X)))
    #Convert to Theano-Lasagne shape
    Xreshaped = Xreshaped.reshape((Xreshaped.shape[0], 1, imsize, imsize))
    return Xreshaped, y

##Data Augmentation and Minibatch
augmentations = {
    "identity": lambda x: x,
    "flipvert": lambda x: np.flip(x, 2),
    "fliphor" : lambda x: np.flip(x, 3),
    "rot90"   : lambda x: np.rot90(x, k=1, axes=(2,3)),
    "rot180"  : lambda x: np.rot90(x, k=2, axes=(2,3)),
    "rot270"  : lambda x: np.rot90(x, k=3, axes=(2,3))
}

augmentations_state = {
    "identity": True,
    "flipvert": True,
    "fliphor" : True,
    "rot90"   : True,
    "rot180"  : True,
    "rot270"  : True
}

def augmentation_iter(images):
    while True:
        choice = random.choice([augm for augm in augmentations_state.keys() if augmentations_state[augm]])
        yield augmentations[choice](images)
        

#Now enforcing class balance!
def minibatch_iter(images, targets, batchsize):
    indices = np.arange(len(images))
    classes = set(targets)
    class_indices_list = list()
    class_size = int(batchsize / len(classes))

    for targ_class in classes:
        class_indices = [x for x in indices if targets[x] == targ_class]
        np.random.shuffle(class_indices)
        class_indices_list.append(class_indices)

    #number of examples processed for each class 
    numproc = 0
    smallest_class = min([len(x) for x in class_indices_list])
    while numproc < smallest_class:
        excerpt = list()
        i = 0
        for class_indices in class_indices_list:
            if len(class_indices) < class_size:
                excerpt = excerpt + class_indices
            else:
                excerpt = excerpt + class_indices[0:class_size]
        yield images[excerpt], targets[excerpt]
        numproc += batchsize / len(classes)   

def stratified_split(images, targets, test_size, random_state=0, stratified=False):
    if stratified:
        nfolds = min(int(test_size ** -1), 2)
        skf = model_selection.StratifiedKFold(n_splits=nfolds, random_state=random_state)
        gen = skf.split(images, targets)
        train, test = next(gen)
        del skf
        return images[train], targets[train], images[test], targets[test]
    else:
        Xtrain, Xtest, ytrain, ytest = model_selection.train_test_split(images, targets, test_size=test_size)
        return Xtrain, ytrain, Xtest, ytest

    



