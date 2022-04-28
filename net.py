import numpy
from copy import deepcopy
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

class SE_layer:
    def __init__(self, SE_channels):
        self.SE_channels = SE_channels

        self.optimizer = None
        self.np = numpy

    def Forward(self, inp):

        GA_out = self.GlobalAverage.Forward(inp)
        #Shape N, C
        FC_out = self.FC2.Forward(self.ReLU.Forward(self.FC1.Forward(GA_out.T)))
        #Shape 2*C, N
        reshaped_out = FC_out.T.reshape(-1, 2, self.C) #Shape N, 2, C
        #Both shape N, C
        split_W = self.Sigmoid.Forward(reshaped_out[:, 0, :])
        split_B = reshaped_out[:, 1, :]

        return inp * split_W[:, :, None, None] + split_B[:, :, None, None]

    def Forward_training(self, inp):
        self.inp_cache = inp.copy()

        GA_out = self.GlobalAverage.Forward_training(inp)
        #Shape N, C
        FC_out = self.FC2.Forward_training(self.ReLU.Forward_training(self.FC1.Forward_training(GA_out.T)))
        #Shape 2*C, N
        reshaped_out = FC_out.T.reshape(-1, 2, self.C) #Shape N, 2, C
        #Both shape N, C
        split_W = self.Sigmoid.Forward_training(reshaped_out[:, 0, :])
        split_B = reshaped_out[:, 1, :]

        return inp * split_W[:, :, None, None] + split_B[:, :, None, None]

    def Backward(self, inp):
        split_W_grad = self.Sigmoid.Backward((self.inp_cache * inp).sum(axis=(2, 3)) / self.HW)
        split_B_grad = inp.sum(axis=(2, 3)) / self.HW #Shape N, C
        combined_grad = self.np.stack((split_W_grad, split_B_grad), axis=1) #Shape N, 2, C
        out = self.GlobalAverage.Backward(
            self.FC1.Backward(
                self.ReLU.Backward(
                    self.FC2.Backward(
                        combined_grad.reshape(-1, 2*self.C).T #Shape C, N
                    )
                )
            ).T #Shape N, C
        )

        FC1_weight_grad, FC1_bias_grad = self.FC1.gradient
        FC2_weight_grad, FC2_bias_grad = self.FC2.gradient
        self.gradient = (FC1_weight_grad, FC1_bias_grad, FC2_weight_grad, FC2_bias_grad)

        return out

    def Update(self):
        FC1_weight_grad, FC1_bias_grad, FC2_weight_grad, FC2_bias_grad = self.gradient
        self.FC1.gradient = (FC1_weight_grad, FC1_bias_grad)
        self.FC2.gradient = (FC2_weight_grad, FC2_bias_grad)
        self.FC1.Update()
        self.FC2.Update()

    def Build(self, shape):
        _, C, H, W = shape
        self.C = C
        self.HW = H * W

        self.GlobalAverage = Hazelnut.modules.Pooling.GlobalPooling()
        self.FC1 = Hazelnut.modules.Linear(self.SE_channels)
        self.ReLU = Hazelnut.modules.ActivationFunction.ReLU()
        self.FC2 = Hazelnut.modules.Linear(self.C * 2)
        self.Sigmoid = Hazelnut.modules.ActivationFunction.Sigmoid()

        #Building all the layers
        #ReLU and Sigmoid don't need to be built
        build_inp = self.np.ones((1, C, H, W))
        self.optimizer.np = None #Remove np to allow deepcopying
        for module in [self.GlobalAverage, self.FC1, self.FC2]:
            try:
                module.optimizer = deepcopy(self.optimizer)
                module.optimizer.np = self.np #Setting the np here to allow deepcopying of the module

            except AttributeError: #Module doesn't need an optimizer
                pass

            try:
                module.np = self.np
            except AttributeError: 
                pass

            module.Build(build_inp.shape)
            build_inp = module.Forward(build_inp)

            if module == self.GlobalAverage:
                build_inp = build_inp.T

        assert build_inp.shape == (2*C, 1)

        

class Net(Hazelnut.NN):
#class Net(Hazelnut.NN):
    def __init__(self):
        super().__init__()
        #super().__init__()
        Skip = Hazelnut.modules._skip_connection.SkipConnClass()

        self.add(Conv(8, 3, mode='Same'))
        self.add(ActivationFunction.ReLU())
        #Residual Layers
        for _ in range(5):
            self.add(Skip)
            self.add(Conv(8, 3, mode='Same'))
            #self.add(BatchNorm())
            self.add(ActivationFunction.ReLU())
            self.add(Conv(8, 3, mode='Same'))
            self.add(SE_layer(4))
            #self.add(BatchNorm())
            self.add(Skip)
            self.add(ActivationFunction.ReLU())

        self.add(Conv(8, 3, mode='Same'))
        self.add(ActivationFunction.ReLU())

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
        #self.add(ActivationFunction.Sigmoid())
        #self.loss = LossFunction.MSE()

        #self.optimizer = Hazelnut.modules.Optimizers.Momentum(learning_rate=0.01)
        #self.optimizer = Hazelnut.modules.Optimizers.SGDM(learning_rate=0.02, momentum_weight=0.48)
        self.optimizer = Hazelnut.modules.Optimizers.RProp()

        self.add(Softmax())
        self.loss = CrossEntropy()

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