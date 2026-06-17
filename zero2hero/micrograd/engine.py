class Value:
    def __init__(self, value, _children=(), _op=''):
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.value}, grad={self.grad})"
    
    def __neg__(self):
      return self * -1

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, (self, other), '*')

        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __rtruediv__(self, other):
        return other * self**-1
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.value**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.value**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.value < 0 else self.value, (self,), 'ReLU')

        def _backward():
            self.grad += (out.value > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        visited = set()
        topo = []

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
           
    # 这是一个有严重 Bug 的直观写法
    def naive_backward(self):
        self._backward() # 触发自己的反向传播，将梯度传给子节点
        for child in self._prev:
            child.naive_backward() # 递归调用子节点
    # Backward must run in reverse topological order.
    # A Value can feed multiple downstream ops, so its grad may receive contributions
    # from several branches. We must first accumulate all contributions to v.grad,
    # then call v._backward() exactly once. Naive recursion either propagates too early
    # or revisits shared nodes and double-counts gradients.
    # see why_naive_recursive_fail.png
    # https://aistudio.google.com/prompts/14XzQDX82xw5qUBW6K4YvXnv8-vI_iNin