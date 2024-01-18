# CS231n Notes 



> 本笔记中的代码来源网络。因本人水平有限，语言存在诸多不严谨之处，仅供参考，若有错误可与我联系。



## Lecture 1: Introduction



## Lecture 2: Image Classification with Linear Classifiers



### Image Classification

把图片用矩阵来表示，可以将 RGB 3个数值 压缩到同一维度，在`data_utils.py`中的 `X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")`  操作，1024个R接续 1024个G 接续 1024个B，一个3072长的行向量来表示一张图片。多张行向量拼接成矩阵，表示多个图片，有多少行就是有多少张图片。

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



#### Geometric Viewpoint

没看懂。

#### Loss Function

##### SVM

$L_i =\sum\limits_{j \neq y_i} max(0,s_j-s_{y_i}+ \Delta)$

#### Softmax

$L_i=-log(\frac {e^{s_{y_i}}}{\sum_\limits je^{s_j}})$

## Lecture 3: Regularization and Optimization
