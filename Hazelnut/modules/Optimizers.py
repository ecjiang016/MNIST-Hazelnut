class SGD:
    def __init__(self, learning_rate=0.0001):
        self.learning_rate = learning_rate
    
    def use(self, gradient):
        return gradient * self.learning_rate

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass

class SGDM:
    """
    Stochastic Gradient Descent + Momentum
    Momentum in the form:
    learning_rate * gradient - (momentum_weight * (gradient - previous_gradient))
    """
    def __init__(self, learning_rate=1e-5, momentum_weight=1e-5) -> None:
        self.learning_rate = learning_rate
        self.momentum_weight = momentum_weight

        self.update_cache = 0

    def use(self, gradient):
        out = self.learning_rate * gradient + (self.momentum_weight * self.update_cache)
        self.update_cache = out.copy()
        return out
        
    def Save(self):
        return {'args':(), 'var':(self.learning_rate, self.momentum_weight, self.gradient_cache)}

    def Load(self, var):
        self.learning_rate, self.momentum_weight, self.gradient_cache = var

class Momentum:
    def __init__(self, learning_rate=0.0001, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta

        self.momentum_vector = 0

    def use(self, gradient):
        self.momentum_vector = (self.momentum_vector * self.beta) + (1 - self.beta) * gradient
        return self.momentum_vector * self.learning_rate

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        pass

class RProp:
    def __init__(self, step_size=0.001, scale=(0.5, 1.2), clip=(0.000001, 50)):
        """
        - step_size: starting step size > 0
        - scale: The scaling sizes for the step sizes. A tuple with (decrease scale, increase scale)
        - clip: The min and max step size. A tuple with (min size, max size)
        """
        self.steps = step_size
        self.scale_small, self.scale_large = scale
        self.clip_min, self.clip_max =  clip
        self.signs = 0

        #CPU/GPU (NumPy/CuPy)
        #self.np is set by the main neural_net program to allow deepcopying of this class
        self.np = None

    def use(self, gradient):
        new_signs = self.np.sign(gradient)
        sign_change = self.signs * new_signs
        self.steps = self.np.clip((((sign_change < 0) * self.scale_small) + ((sign_change > 0) * self.scale_large) + ((sign_change == 0) * 1)) * self.steps, self.clip_min, self.clip_max)
        self.signs = new_signs
        return self.steps * new_signs

    def Save(self):
        return {'args':(), 'var':(self.steps, self.scale_large, self.scale_large, self.clip_min, self.clip_max, self.signs)}

    def Load(self, var):
        self.steps, self.scale_large, self.scale_large, self.clip_min, self.clip_max, self.signs = var

class RMSProp:
    """Root mean squared propagation"""
    def __init__(self, learning_rate=1e-6, beta=0.9) -> None:
        """
        Args:
        - learning_rate: Learning rate
        - beta: The weight of the accumulated squared gradients over the new one

        Hidden:
        - epsilon: Small number stopping division by zero. Automatically set to 1e-10
        """
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = 1e-10

        #Initial values
        self.grad_average = 0

        #CPU/GPU (NumPy/CuPy)
        #self.np is set by the main neural_net program to allow deepcopying of this class
        self.np = None

    def use(self, gradient):
        self.grad_average = self.beta * self.grad_average + ((1 - self.beta) * self.np.square(gradient))
        
        return gradient * self.learning_rate / self.np.sqrt(self.grad_average + self.epsilon)

    def Save(self):
        return {'args':(), 'var':(self.learning_rate, self.beta, self.epsilon, self.grad_average)}

    def Load(self, var):
        self.learning_rate, self.beta, self.epsilon, self.grad_average = var

class Adam:
    """Adaptive Moment Optimization"""
    def __init__(self, learning_rate=1e-6, beta1=0.9, beta2=0.99) -> None:
        """
        Args:
        - learning_rate: Learning rate
        - beta1: The weight of the accumulated gradients over the new one
        - beta2: The weight of the accumulated squared gradients over the new one

        Hidden:
        - epsilon: Small number stopping division by zero. Automatically set to 1e-10
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-10

        #Initial values
        self.grad_average = 0
        self.square_grad_average = 0

        #CPU/GPU (NumPy/CuPy)
        #self.np is set by the main neural_net program to allow deepcopying of this class
        self.np = None

    def use(self, gradient):
        self.grad_average = self.beta1 * self.grad_average + ((1 - self.beta1) * gradient)
        self.square_grad_average = self.beta2 * self.square_grad_average + ((1 - self.beta2) * self.np.square(gradient))
        
        return gradient * self.grad_average * self.learning_rate / self.np.sqrt(self.square_grad_average + self.epsilon)

    def Save(self):
        return {'args':(), 'var':(self.learning_rate, self.beta1, self.beta2, self.epsilon, self.grad_average, self.square_grad_average)}

    def Load(self, var):
        self.learning_rate, self.beta1, self.beta2, self.epsilon, self.grad_average, self.square_grad_average = var

class YellowFin(SGDM):
    def __init__(self, beta=0.9, early_stop=True, w=20):
        super().__init__()
        #YellowFin variables
        self.beta = beta
        self.w = w #Sliding window
        assert early_stop == True or early_stop == False
        self.early_stop = early_stop

        self.epsilon = 1e-7

        #Iteration tracking
        self.t = 1

        #Curvature range variables
        self.h = []
        self.h_max = 0
        self.h_min = 0

        #Gradient variance variables
        self.g_2 = 0
        self.g_ = 0

        #Distance to optimum variables
        self.g_norm = 0
        self.h_ = 0
        self.D = 0

    def use(self, gradient):
        #Calculate learning rate and momentum weight

        #Calculate curvature range
        self.h.append(float(self.np.sum(self.np.square(gradient))))
        if len(self.h) >= self.w:
            self.h.pop(0)

        h_max_t = max(self.h)
        h_min_t = min(self.h)

        self.h_max = (self.beta * self.h_max) + ((1 - self.beta) * h_max_t)
        self.h_min = (self.beta * self.h_min) + ((1 - self.beta) * h_min_t)

        #Gradient variance
        self.g_2 = (self.beta * self.g_2) + ((1 - self.beta) * self.np.square(gradient))
        self.g_ = (self.beta * self.g_) + ((1 - self.beta) * gradient)
        C = float(self.np.sum(self.g_2 - self.np.square(self.g_)))

        #Distance to optimum
        self.g_norm = (self.beta * self.g_norm) + ((1 - self.beta) * float(self.np.sum(self.np.absolute(self.g_norm))))
        self.h_ = (self.beta * self.h_) + ((1 - self.beta) * self.h[-1])
        self.D = (self.beta * self.D) + ((1 - self.beta) * (self.g_norm/self.h_))


        #SingleStep
        
        #Get cubic root
        #From https://github.com/JianGoForIt/YellowFin/blob/master/tuner_utils/yellowfin.py
        p = (self.D + self.epsilon)**2 * (self.h_min + self.epsilon)**2 / 2 / (C + self.epsilon)
        w3 = (-self.np.sqrt(p**2 + 4.0 / 27.0 * p**3) - p) / 2.0
        w = self.np.sign(w3) * self.np.power(self.np.absolute(w3), 1.0/3.0)
        y = w - p / 3.0 / (w + self.epsilon)
        x = y + 1

        root = x ** 2
        dr = self.np.maximum((self.h_max + self.epsilon) / (self.h_min + self.epsilon), 1.0 + self.epsilon)
        self.momentum_weight = self.np.maximum(root, ((self.np.sqrt(dr) - 1) / (self.np.sqrt(dr) + 1))**2)
        self.learning_rate = self.np.square(1 - self.np.sqrt(self.momentum_weight)) / self.h_min

        if self.early_stop:
            self.learning_rate = min(self.learning_rate, self.learning_rate * self.t / (10 * self.w))
            self.t += 1

        return super().use(gradient)

    def Save(self):
        return {'args':(), 'var':(self.beta, self.epsilon, self.h, self.h_max, self.h_min, self.g_2, self.g, self.g_norm, self.h_, self.D)}

    def Load(self, var):
        self.beta, self.epsilon, self.h, self.h_max, self.h_min, self.g_2, self.g, self.g_norm, self.h_, self.D = var