##Constructing Lasagne network, compiling train and test functions
import numpy as np
import matplotlib.pyplot as plt

class LasagneModel:

    def __init__():
        raise(Exception("Abstract class instantiated"))

    def train(self, Xtrain, ytrain, saveoutput):
        raise NotImplementedError

    def score(self, Xtest, Xtestfeats, ytest):
        raise NotImplementedError

    #Universal functions
    def save_results(self, train_loss, test_loss):
        print("Saving results")
        plt.plot(np.arange(len(train_loss)), train_loss, label="train loss")
        plt.plot(test_loss, label="test loss")
        plt.yscale('log')
        plt.ylabel('epochs')
        plt.title(self.name)
        plt.legend()
        plt.savefig("plot_{}.png".format(self.name))
        np.savez("data_{}".format(self.name),
                 train_loss=np.array(train_loss),
                 test_loss=np.array(test_loss))
        plt.cla()
        plt.clf()
        plt.close()
