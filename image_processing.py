import numpy as np
import random
from scipy import misc

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
        

def minibatch_iter(images, targets, batchsize):
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    for index in range(0, len(images) - batchsize + 1, batchsize):
        excerpt = indices[index:index + batchsize]
        yield images[excerpt], targets[excerpt]


