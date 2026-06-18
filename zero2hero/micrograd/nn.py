from engine import Value
import random

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        out = act.relu() if self.nonlin else act
        return out

    def parameters(self):
        return self.w + [self.b]
    
    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        # This is Python syntax for setting nonlin to True for all layers except the last one.
        # For final output layer in regression/binary score models, you often want raw output without activation
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# 计算过程推演
# If layer 1 has 4 neurons, then layer 1 produces 4 numbers:

# h = [h1, h2, h3, h4]

# So layer 2 receives a 4-dimensional input vector.

# If layer 2 has, say, 3 neurons, then each of those 3 neurons receives all 4 values:

# layer 2 neuron 1: [h1, h2, h3, h4] · w1 + b1
# layer 2 neuron 2: [h1, h2, h3, h4] · w2 + b2
# layer 2 neuron 3: [h1, h2, h3, h4] · w3 + b3

# Each layer 2 neuron has its own 4 weights plus bias:

# neuron 1: w1 = [w11, w12, w13, w14], b1
# neuron 2: w2 = [w21, w22, w23, w24], b2
# neuron 3: w3 = [w31, w32, w33, w34], b3

# So the shape transition is:

# input dim = 3
# layer 1 has 4 neurons -> output dim = 4
# layer 2 has 3 neurons -> output dim = 3
# layer 3 has 1 neuron  -> output dim = 1

# In micrograd, this is why code like this works:

# n = MLP(3, [4, 4, 1])

# Meaning:

# original input dimension = 3
# first hidden layer: 4 neurons, each takes 3 inputs
# second hidden layer: 4 neurons, each takes 4 inputs
# output layer: 1 neuron, takes 4 inputs

# So the full connection pattern is:

# x = [x1, x2, x3]

# layer 1:
# 4 neurons, each connected to x1, x2, x3
# output: [h1, h2, h3, h4]

# layer 2:
# 4 neurons, each connected to h1, h2, h3, h4
# output: [g1, g2, g3, g4]

# output layer:
# 1 neuron, connected to g1, g2, g3, g4
# output: y

# That is why it is called fully connected / dense: every neuron in the next layer receives every output from the previous
# layer.