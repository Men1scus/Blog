# CS231n Notes



> [Stanford University CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/index.html)
>
> [CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)
>
> [Lecture Collection | Convolutional Neural Networks for Visual Recognition (Spring 2017) - YouTube](https://youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&feature=shared)



## Lecture 1: Introduction



## Lecture 2: Image Classification with Linear Classifiers



### Image Classification

把 图片用矩阵来表示，可以将 RGB 3个数值 压缩到同一维度，在`data_utils.py`中的 `X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")`  操作，1024个R接续 1024个G 接续 1024个B，一个3072长的行向量来表示一张图片。多张行向量拼接成矩阵，表示多个图片，有多少行就是有多少张图片。

分类器就是用图片这种参数，处理后返回标签。

机器学习：用图片和标签训练模型，返回模型。用模型和测试照片可以返回标签。

### K-Nearest Neighbor

#### Nearest Neighbor

训练只是记住所有数据和标签，预测时进行计算。

L1距离：两图片的矩阵每个像素的差的绝对值，再对差值矩阵的所有像素求和。

L2距离：对差值矩阵的每个像素先平方再求和再开根号。

`  distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)` 运用Numpy的Broadcast特性，每次循环，将测试集的一行拿出来，扩充成和训练集一样的矩阵，进行相减取绝对值，再对差值矩阵中所有元素求和。就是测试集中第 i 张图片与训练集中每一张图片的 L1 距离所构成的N x 1 的一维数组。argmin(distances) 对一维数组中的每个元素求和。

#### K-Nearest Neighbor

找前 K 个最临近的图片。

作业题中预测分2次，1次和0次循环。

2次循环 第i个测试样本和第 j 个训练样本之间的L2 距离 放在dists[i]\[j] 上

1次循环 整个训练矩阵直接减第 i 个测试样本，Broadcast 机制，将第 i 个样本扩充为每行都相同的矩阵再做矩阵减法 放在dists[i] 上。

0次循环，观察 L2 距离的公式，完全平方拆开，三部分依次进行计算。中间的部分用矩阵乘法来代替，矩阵乘法本身也可以看作一种求和方式，可消掉求和号。两个平方项分别从两个方向Broadcast。



### Linear Classifier

​					$f(x_i,W,b) = Wx_i + b$

用代表10个类别的共10行的权重矩阵 $W$ 左点乘单张图片的展平形式，得到10 x 1的矩阵，还可加上10 x 1 的常数矩阵 $b$ 进行修正，得到这张图片在10个类别的分数，希望分数最高的那个类就是图片真实的类别。

h

#### Geometric Viewpoint

没看懂。

#### Loss Function

##### SVM

$L_i =\sum\limits_{j \neq y_i} max(0,s_j-s_{y_i}+ \Delta)$

#### Softmax

$L_i=-log(\frac {e^{s_{y_i}}}{\sum_\limits je^{s_j}})$

## Lecture 3: Loss Functions and Optimization



## Lecture4: Backpropagation and Neural Networks

> [lecture_4 (stanford.edu)](http://cs231n.stanford.edu/slides/2023/lecture_4.pdf)

### Backpropagation

#### Computational Graph

$q = x + y$ 对 $x,y$ 的偏导分别是二者的系数

$f = qz$  $q,z$两个变量互为偏导

####  链式法则

${\part f \over \part q} = z$	`dfdq = z `

${\part f \over \part x} = {\part f \over \part q} {\part q \over \part x}$ 	`dfdx = dfdq * dqdx `

![image-20240118234754222](https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401182347332.png)





边是变量值，结点是运算/函数。

最左端各个变量经过每个结点的运算逐步到最右端，每个结点都是运算，都是函数，可求出的偏导，可以理解成每个运算左侧变量对于右侧变量影响的大小，最终想要得知输入变量对于结果变量影响的大小，也就是结果变量对于左侧每个输入变量的偏导。

运算值经过运算结点从左向右传递，偏导值从输出节点以 1.00 开始向左传递，每次将结点左侧的变量值以 $x$ 代入该结点值的偏导，与右侧偏导相乘，得到结点左侧的偏导。逐层累乘传播偏导，最终传回输入处。

Sigmoid Gate 由四个连续的结点相连，实现激活函数 $ \sigma (x) = {1 \over {1 + e^{-x}}}$

这个门的梯度可以进行简化  ${d\sigma(x) \over dx} = (1 - \sigma(x)) \sigma(x) $	用一次计算简化 4 个结点

$x$ 仍然代入门左侧的变量值，但经 $\sigma(x)$ ,实际上是将 $\sigma(x)$ 这个整体代入门右侧的变量值。



#### Patterns

add gate  分配梯度 "unchanged and equally"

max gate 选取梯度 “router”，分配给 "max during the forward pass" 其中较大的变量会继承所有的梯度 

mul gate  交换梯度



#### Vectorized Operations

$q$ 是二维向量，我们想知道 $q$ 中每个元素对于 $f$ 的影响 

$\nabla_xf = 2W^T \cdot q$

$\nabla$ 是向量微分算子

W 和 x 都是矩阵，矩阵乘法

不用背表达式，根据 dimensions 来尝试构造表达式

$dW $与 $W$ 的 size 相同

> **then the only way of achieving this is with**

推荐资料



 **Cache forward pass variables**

缓存变量

**Gradients add up at forks**

反向传播时使用 `+=` 来累加梯度而不是覆盖



###  Neural Networks

<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401182345333.png" alt="image-20240118234508263" style="zoom:33%;" />

通过非线性函数  堆叠多个线性层

e.g.近似解释（不准确）

$X$  向左的马

$W_1$ 一匹向右的马，一匹向左的马 

h 是在  W1 里面每个模板的得分数值

$W_2$ 模板总的加权和，普遍得分高的那匹马



## Lecture 5: Convolutional Neural Networks

传递到每个结点，先进行激活再往后传

$max(0,-)$ 逐元素应用 非线性函数 
