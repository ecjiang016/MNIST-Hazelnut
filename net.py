import numpy
import os
import wandb
import Hazelnut
from Hazelnut.modules import *
from Hazelnut.modules.ActivationFunction import Sigmoid

class Softmax:
    def __init__(self) -> None:
        self.np = numpy

    def Forward(self, inp):
        exps = self.np.exp(inp - self.np.max(inp, axis=0)[None, :]) #Add numerical stability
        return exps/self.np.sum(exps, axis=0)[None, :]

    def Forward_training(self, inp):
        exps = self.np.exp(inp - self.np.max(inp, axis=0)[None, :]) #Add numerical stability
        return exps/self.np.sum(exps, axis=0)[None, :]

    def Backward(self, inp):
        return inp

    def Build(self, _):
        pass

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass

class CrossEntropy:
    def __init__(self) -> None:
        #CPU/GPU (NumPy/CuPy)
        self.np = numpy

    def Forward(self, out, correct_out):
        return -self.np.sum(correct_out * self.np.log(out + 1e-10)) / out.shape[1]

    def Backward(self, out, correct_out):
        return out - correct_out

class Net(Hazelnut.presets.YellowFin.NN):
    def __init__(self):
        super().__init__()

        self.add(Conv(4, 3, mode='Same'))
        self.add(ActivationFunction.ReLU())
        #Residual Layers
        for _ in range(1):
            #self.add(SkipConn('Start'))
            self.add(Conv(4, 3, mode='Same'))
            #self.add(BatchNorm())
            self.add(ActivationFunction.ReLU())
            self.add(Conv(4, 3, mode='Same'))
            #self.add(BatchNorm())
            #self.add(SkipConn('End'))
            self.add(ActivationFunction.ReLU())

        #self.add(Conv(4, 3, mode='Same'))
        #self.add(ActivationFunction.ReLU())

        self.add(Conv(1, 3, mode='Same'))
        #self.add(BatchNorm())
        self.add(ActivationFunction.ReLU())

        self.add(Flatten())

        #MLP stuff
        self.add(Linear(784))
        self.add(ActivationFunction.ReLU())
        #self.add(Linear(100))
        #self.add(ActivationFunction.ReLU())
        self.add(Linear(100))
        self.add(ActivationFunction.ReLU())
        #self.add(Linear(50))
        #self.add(ActivationFunction.ReLU())
        self.add(Linear(10))
        self.add(ActivationFunction.Sigmoid())
        self.loss = LossFunction.MSE()

        #self.add(Softmax())
        #self.loss = CrossEntropy()

        self.mode = 'gpu'

        self.build((1, 28, 28))

    def save(self, name, run):
        super().save(name)
        artifact = wandb.Artifact(name, type="PolicyNet")
        artifact.add_file(name)
        run.log_artifact(artifact)
        os.remove(name)
    
    def load(self, name, run, version=':latest'):
        artifact = run.use_artifact(name+version)
        artifact_dir = artifact.download(name)
    
        os.remove(artifact_dir+"\\"+name)
        os.rmdir(artifact_dir)

        super().load(artifact_dir+"\\"+name)