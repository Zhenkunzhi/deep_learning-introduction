# python 入门

## python 解释器

## python 脚本文件

## numpy  &  matplotlib  外部库

```python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('C:/Users/86156/Desktop/1.jpg') 
plt.imshow(img)
plt.show()

x = np.arange(0,6,0.1)  #以0.1为步长单位，生成0-6的数据
y1 = np.sin(x)
y2 =np.cos(x)
plt.plot(x,y1,label = "sin")
plt.plot(x,y2,linestyle = "--",label = "cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()

a = np.array([[1.0,2.0,3.0],[2.0,3.0,4.0]])

print(a)

print("hello world!")

class man:
    def __init__ (self,name,age,height):
        self.name = name
        self.age = age
        self.height = height
        print("initialized")
    def hello(self):
        print("my name is " + self.name + "!")

    def goodbye(self):
        print(self.height)

m = man("josh",12,180)
m.hello()
m.goodbye()

```





**感知机（perceptron): 作为神经网络（深度学习）的起源算法，学习感知机的构造也是学习通往 神经网络和深度学习的一种重要的思想**





**神经网络：自动的从数据中学习到合适的权重参数**

**激活函数（activation function)**

**朴素感知机是指单层网络，激活函数使用了阶跃函数模型，多层感知机是指神经网络，使用了sigmoid函数等平滑的激活函数的多层网络**





**阶跃函数：**

```python
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)    #   x>0为符号运算  即x>0为true, x<0 为false 
                                            #   np.array(object,dtype = none)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)  # 指定图中绘制的y轴的范围
plt.show()

```

**sigmoid 函数**

**NumPy的广播功能使得标量和数组能够进行运算**

```python
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()

```

**sigmoid函数的平滑性对神经网络的学习具有重要的意义**

**使用线性函数的话加深神经网络的层数就没有意义**

**ReLU函数：rectified linear unit**

```python
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.show()

```

**神经网络可以用在分类问题和回归问题上，分类问题用softmax函数**

```python
import numpy as np
import matplotlib.pylab as plt

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
```

**计算机在处理“数”时，数值必须在4字节或8字节的有限数据宽度内，在进行指数运算的时候容易出现nan（not a number)的情况，因此在使用softmax函数的时候，需要减去最大值**

```python
import numpy as np
import matplotlib.pylab as plt

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y
```



$$
y_k = \frac {exp(a_k)}{\sum_{i=1}^{n}exp(a_i)}
= \frac {Cexp(a_k)}{C\sum_{i=1}^{n}exp(a_i)} = \frac {exp(a_k + logC)}{\sum_{i=1}^{n}exp(a_i+logC)} = \frac {exp(a_k + C^`)}{\sum_{i=1}^{n}exp(a_i+ C^`)}
$$


**softmax函数性质：输出的值为0.0~1.0；输出值的总和为1.0**

**将softmax函数输出解释为概率**

**前向传播：forward propagation,使用神经网络解决问题时，首先使用训练数据(学习数据）进行权重参数的学习，进行推理时，使用刚学习到的参数，对输入的数据进行分类**

os.getcwd()： 获取当前工作目录，即当前python脚本工作的目录路径。

os.chdir("dirname") ：改变当前脚本工作目录；相当于shell下cd。

os.curdir 返回当前目录: ('.')。

os.pardir 获取当前目录的父目录字符串名：('..')。

os.makedirs('dirname1/dirname2') ：可生成多层递归目录。

os.removedirs('dirname1')： 若目录为空，则删除，并递归到上一级目录，如若也为空，则删除，依此类推。

**把数据限定到某个范围内的处理称为：正规化（normalization), 对神经网络的输入数据进行，某种既定的转换称为预处理（pre-processing)**



**很多预处理都会考虑到数据的整体分布，利用数据的整体的均值或标准差，移动数据，使数据整体以0为中心或者进行正规化，把数据的延展控制在一定的范围内**



**将数据整体的分布形状均匀化的方法，即  数据白化（whitening)**



**打包的形式输入数据成为 批（batch)    大多数处理数值计算的库都进行了能够高效处理大型数组运算的最优化**

**range(start, end, step):    start ~ end -1 步长为 step**



**神经网络中的激活函数使用平滑变化的sigmoid函数或者ReLU函数**



**机器学习的问题大体上可以分为回归问题和分类问题**



**输出层激活函数，回归问题中为恒等函数，分类问题中一般用softmax函数**



**分类问题中输出层的神经元的数量设置为要分类的类别数**



**“学习” ：从训练 数据中自动获取最优权重参数的过程   学习的目的就是以损失函数为基准找出能够使它的值达到最小的权重系数（梯度法）**



**深度学习又被称为端到端的机器学习(end to end machine learning) 是指从原始数据（输入）中获得目标结果（输出）**

**数据分为训练数据和测试数据两部分，训练数据进行学习，寻找最优的参数，使用测试数据评价训练得到的模型的实际能力，为了正确评价模型的泛化能力，将数据分为这两种，训练数据又称为监督数据**

**泛化能力是指处理未被观察的数据的能力，获得泛化能力是机器学习的最终目标**



**损失函数（loss function): 损失函数可以是任意函数，但是一般用均方误差和交叉熵误差等**

**损失函数：表示神经网络性能“恶劣程度”的指标，即当前的神经网络对监督数据在多大程度上不拟合，在多大程度上不一致**



**均方误差（mean squared error)**
$$
E = \frac 1 2 \sum_{k}(y_k - t_k)^2
$$


yk表示神经网络的输出，tk表示监督数据，k表示数据的维数



**交叉熵误差：（cross entropy error)**
$$
- \sum_{k}t_k log y_k
$$
**mini-batch学习**

**计算损失函数时必须将所有的训练数据作为对象。即以交叉熵为例：**
$$
-\frac 1N\sum_{n}\sum_{k}t_{nk}logy_{nk}
$$
**当数据量达到几百万或者几千万之多，计算损失函数是不现实的，神经网络的学习也是从训练数据中选出一部分作为全部数据的“近似”，并称之为mini-batch, 小批量 **





**参数的导数，即梯度**

****

**在进行神经网络的学习时，不能将识别精度作为指标，因为以识别精度作为指标时，则参数的导数在绝大数的地方都为零，识别精度对参数的微小变化基本没有什么反应 ，即便有反应，它的值也是不连续的突变的，阶跃函数作为损失函数，参数的微小变化也会被阶跃函数抹杀，而sigmoid函数的导数是连续的而且在任何地方都不为零。 **





**数值微分（numerical differentiation):**

```python
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
```

**1.其中h的值不能太小，不然会产生舍入误差（rounding error)省略小数精细部分的数值**

```python
>>> np.float32(1e-50)
0.0
```

**2.计算函数f 在 （x + h) 和 ( x - h ) 之间的差分，中心差分**

函数：
$$
f(x_0,x_1) = {x_0}^2 + {x_1}^2
$$
图像：

![1656744575797](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\1656744575797.png)

在x_0 = 3, x_1 = 4 时的偏导数：
$$
\frac {\partial f}{\partial {x_0}} =  \frac{f(x_0 + h,4) - f(x_0 -h,4)}{2h}
$$

$$
\frac {\partial f}{\partial {x_1}} =  \frac{f(3,x_1 + h) - f(3,x_1 - h)}{2h}
$$

```python
# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_tmp(x):
	return x**2 + 4.0**2

numerical_diff(function_tmp,3.0)
```

**梯度：**所有的下降方向中，梯度的下降的方向最多
$$
(\frac{\partial f}{\partial x_0},\frac{\partial f}{\partial x_1})
$$




```python
def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)  #生成形状和x相同但是元素都为零的数组
    
    for idx in range(x.size):    #遍历每个数组中 的元素计算该点的中点差分
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值    还原为求偏导的坐标值
        
    return grad

```

**梯度下降法（gradient descent method)**
$$
x_0 = x_0 - \eta\frac{\partial f}{\partial x_0}
$$

$$
x_1 = x_1 - \eta\frac{\partial f}{\partial x_1}
$$

其中
$$
\eta
$$
表示更新量，在神经网络的学习中，称为**“学习率”（learning rate)**

**神经网络学习也需要计算梯度，损失函数对权重参数的梯度**
$$
W = \left(\begin{matrix}w_{11} & w_{12} & w_{13}\\w_{21}&w_{22}&w_{23}\\
\end{matrix}\right)
$$

$$
\frac{\partial L}{\partial W} =
\left(
\begin{matrix}
\frac{\partial L}{\partial w_{11}} & \frac{\partial L}{\partial w_{12}}&\frac{\partial L}{\partial w_{13}}\\
\frac{\partial L}{\partial w_{21}} & \frac{\partial L}{\partial w_{22}} & \frac{\partial L}{\partial w_{23}}
\end{matrix}
\right)
$$



**过拟合：训练数据能够很好地被识别，但是不在训练数据和测试数据的无法被识别**

**epoch：表示学习中所有训练数据均被使用过一次的更新次数**

**神经网络学习分为训练数据集合和测试数据集，训练数据集用于学习，测试数据集用于评价学习模型的泛化能力，神经网络的学习以损失函数为指标，更新权重，减少损失函数的值，通过数值微分（时间复杂度比较高）计算权重参数的梯度**

#### 误差反向传播法

**计算图（computational graph)**

**正向传播（forward propagation)**

**反向传播（backward propatation)**



#### 激活函数层的实现

**实现激活函数层ReLU层和Sigmoid层**

##### ReLU层（Rectified Linear Unit)

$$
y = 
\begin{cases}
x\     \left(x\lt0\right)\\
0\     \left(x\le0\right)
\end{cases}
$$

求导数
$$
\frac{\partial y}{\partial x}= 
\begin{cases}
1\     \left(x\lt0\right)\\
0\     \left(x\le0\right)
\end{cases}
$$

```python
class Relu:
	def __init__(self)
    	self.mask = None
    
    def forward(self , x):
        self.mask = (x <= 0)   #将x<=0的值保存为True 
        out = x.copy()         #out[0/Flase]  就复制x的值
        out[self.mask] = 0     #
        
        
        return out   #输出的是out[0] self.mask = False (x>0)的情况
    
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
```

![1657162675310](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\1657162675310.png)





Sigmoid层（Sigmoid函数）：
$$
y = \frac{1}{1 + exp(-x)}
$$
Sigmoid层的计算图：
$$
\frac{\partial L}{\partial y}y^2exp(-x) \ \ \ \ \ \ \ \ \ \ <---\ \ \ \ \ \ \ \ \ \ \ \ \ \ \frac{\partial L}{\partial y}
$$

$$
\frac{\partial L}{\partial y}y^2exp(-x) = \frac{\partial L}{\partial y}y(1-y)
$$

```python
class sigmoid:
    def __init__(self):
        self.out = None
    def forward(self,x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    
    def backward(self,dout):
		dx = dout * (1.0 - self.out)*self.out
        
        return dx
```





##### Affine/Softmax层实现

神经网络的正向传播中进行矩阵乘积运算在集合领域被称为“仿射变换”。因此在这里将仿射变换的处理实现为“Affine层”


$$
X_{2,} \cdot W_{2*3} + B_{3,} = Y_{3,}
$$

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y}\cdot W^T \\
\frac{\partial L}{\partial Y} = X^T\cdot \frac{\partial L}{\partial Y}
$$

$$
X = (x_0,x_1,x_2,\cdot\cdot\cdot\ ,x_n)\\
\frac{\partial L}{\partial X} = 
\left(
\frac{\partial L}{\partial x_0},\frac{\partial L}{\partial x_1},\frac{\partial L}{\partial x2},\cdot\cdot\cdot\ ,\frac{\partial L}{\partial x_n}
\right)
$$

![1657179571671](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\1657179571671.png)

```python
class affine:
    def __init__(self):
        self.w = w
        self.b = b
        self.x = None
        self.dw = None
        self.db = None
        
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.w) + self.b
        
        return out
    def backward(self, dout):
        dx = np.dot(dout,self.w.T)
        self.dw = np.dot(self.w.T, dout)
        self.db = np.sum(dout,axis = 0)
        
        return dx
    
```









##### Softmax - with - Loss层

输出层 softmax 层

![1657183087296](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\1657183087296.png)



神经网络中进行的处理有***推理***（inference) 和 **学习**两个阶段。神经网络的推理通常不使用softmax层（用于正规化处理）。神经网络中未被正规化的输出结果有时也被称为“得分”，神经网络的学习阶段需要softmax层。

![1657183003862](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\1657183003862.png)

softmax函数记为softmax层，交叉熵误差记为cross entropy error层，假设进行三类分类，从前面的层接收三个输入得分，softmax 将输入（ a1, a2 , a3 )正规化，输出 ( y1 , y2 , y3 ) 。cross entropy error层接收 softmax层的输出( y1 , y2 , y3 ) 和监督标签 ( t1 , t2 , t3 ) , 从这些数据中输出损失 L 。

![1657184954547](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\1657184954547.png)

Softmax层反向传播的结果：
$$
（y_1 - t_1,y_2 - t_2, y_3 - t_3)
$$
y1 , y2 , y3 是Softmax 层的输出，t1 , t2 , t3 是监督数据，因此反向传输的结果是softmax层的输出和监督标签的差分。神经网络的反向传输会将这个差分表示的误差传递给前面的层。

神经网络的学习目的就是通过调整权重参数，是神经网络的输出（softmax层的输出）接近监督标签。

使用交叉熵函数最为softmax函数的损失函数后，反向传播得到的结果，是交叉熵函数的作为损失函数的原因。

```python
class softmaxwithloss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
        
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y,self.t)
        
        return self.loss
    
    def backward(self,dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        
        return dx
```

```python
# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W =W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None # softmax的输出
        self.t = None # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None # Conv层的情况下为4维，全连接层的情况下为2维  

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var  
        
        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
                        
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx


class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

```



通过将神经网络的元素以层的方式实现，可以轻松的构建神经网络，使用层进行模块化的实现具有很大的优势。

求梯度的两种方法：一、基于数值微分的方法，二、解析性的求解数学式的方法（通过使用误差方向传播法）。可以通过数值微分确认误差反向传播法实现的是否正确。

**梯度确认（gradient check)**





##### 与学习相关的技巧

寻找最优权重参数的最优化方法，权重参数的初始值、Dropout等正则化方法

**最优化（optimization)**:寻找最优参数     问题：参数空间复杂       **随机梯度下降法（stochastic gradient descent)  SGD**
$$
W \ \ \ \ \ <--\ \ \ \ \ W - \eta\frac{\partial L}{\partial W}
$$


**Momentum法（动量）**
$$
v\ \ <--\ \ \alpha v\ -\ \eta \frac{\partial L}{\partial W}\\
W\ \ \ <--\ \ \ W \ + \ v
$$
w表示需要更新的权重参数 
$$
\eta
$$
表示学习率

```python
class Momentum:

    """Momentum SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]
```



**AdaGrad:**

在神经网络的学习中，学习率很重要。学习率过小，学习的时间就会增加，学习率过大，就会导致学习发散而不能正确进行学习。在关于学习率的相关技巧中有一个叫做**学习率衰减（learning rate decay)**，随着学习的进行，使学习率逐渐降低，一开始“多学”，到后面“少学”。**AdaGrad**会为每个参数适当地调整学习率。
$$
h\ \ \leftarrow \ \ h + \frac{\partial L}{\partial W}\ast \frac{\partial L}{\partial W}
\\
W \ \ \leftarrow\ \ W  - \eta \frac {1}{\sqrt h} \frac{\partial L}{\partial W}
$$
变量 h 表示以前所有梯度值的平方和（式中*号表示矩阵各个元素的乘积，在参数更新时，通过乘以1/√h可以调整学习的尺度。AdaGrad记录过去所有梯度的平方和，学习越深入，更新的幅度就会越小，如果进行无止境的学习，学习的增量就会变为零，**RMSProp方法**并不是将过去所有的梯度一视同仁的加起来，而是逐渐的遗忘过去的梯度，在做加法运算时，将新梯度的信息更多的反映出来。从专业上来讲，称为“指数平移平均”。呈指数函数式地减少过去梯度的尺度。

```python
class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            
            
```

最后一行加上1e-7这个微小值，防止self.h[key]中有0的情况，0作为除数，在很多深度学习的框架中，这个微小值也可以设定参数。



##### Adam法

融合了momentum和adagrad的方法，此外进行超参数的“偏置矫正”。

```python
class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)
            
```

adam会设置三个超参数，一个是学习率：α，另外两个是一次momentum系数β1（0.9）和二次momentum系数β2（0.999）。

**基于MINIST数据集的更新方法的比较**

![1657262804257](C:\Users\86156\AppData\Roaming\Typora\typora-user-images\1657262804257.png)





##### 权重的初始值

在神经网络的学习中，权重的初始值非常重要，设定什么样的权重初始值，关系神经网络的学习能够成功。

抑制过拟合、提高泛化能力的技巧——**权值衰减（weight decay)**

如果想减小权重的值，需要将初始值设为较小的值才是正途，但是不能将权重设为〇，因为在误差反向传播法中，所有的权重就会进行相同的更新，这就会使得神经网络拥有许多不同的权重被更新为相同的值，失去了意义，为了瓦解权重的对称结构，必须随机生成初始值。

