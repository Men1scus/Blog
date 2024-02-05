# CS231n Assignment1



## K-Nearest-Neighbor

### Frobenius norm

Frobenius 范数

用来衡量矩阵的大小
$$
\Vert A \Vert = \sqrt{\sum\limits_{i,j} A^2_{i,j}}
$$


逐元素平方和，再开根号

```python
np.linalg.norm(，ord = 'fro')
```


lin alg —— linear algebra

## Support Vector Machine

$$
L_i = \sum\limits_{j\neq y_i} max(0,s_j - s_{y_i} + 1)
$$



### Naive

```python
loss = 0.0
……
margin = score[j] - scores[y[i]] + 1
if margin > 0:
	loss += margin
```

### Optimization



$L$ 的梯度是包含所有偏导的**向量**

$W$ 中的每一项对 $L$ 影响的大小所构成的矩阵
$$
L_i = \sum\limits_{j\neq y_i} max(0,s_j - s_{y_i} + 1)
$$

$$
if \ s_{y_i}  \geq s_j + 1 :
$$

$$
L_i = \sum\limits_{j \neq y_i }( s_j - s_{y_i} + 1) = \sum\limits_{j \neq y_i } (X_iW_j - X_iW_{y_j} + 1)
$$

<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401222000951.png" alt="image-20240122200000817"  />


$$
L_i(W) = \sum\limits_{j \neq y_i } X_iW_j\ + ……  = \ \sum\limits_{j \neq y_i }\sum\limits_{k=1}^{3072} x_{i,k} w_{k,j}+……
$$

$$
{\partial L_i(W) \over \partial w_{t,j} }
= \sum\limits_{j \neq y_i } \ {\partial\ \over \partial w_{t,j} } \sum\limits_{k=1}^{3072}  {x_{i,k} w_{k,j}} 
= \sum\limits_{j \neq y_i }{\partial\ \over \partial w_{t,j}}(\ x_{i,1}w_{1,j} + x_{i,2}w_{2,j} + …… + x_{i,t}w_{t,j} + …… +x_{i,3072}w_{3072,j}\ )
$$


$$
=\sum\limits_{j \neq y_i }x_{i,t}
$$
<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401222124092.png" alt="image-20240122212403960"  />

*From this we can easily see that* 
$$
\nabla_{W_j} L_i(W) 
= \nabla_{W_j}\sum\limits_{j \neq y_i } X_iW_j 
= \sum\limits_{j \neq y_i } \nabla_{W_j} X_iW_j 
= \sum\limits_{j \neq y_i } X_i^T
$$
*Mutatis mutandis*
$$
\nabla_{W_{y_i}} L_i(W) = \nabla_{W_{y_i}}\sum\limits_{j \neq y_i }(-X_iW_{y_i}) = \sum\limits_{j \neq y_i }-X_i^T
$$


$$
\nabla_W L(W)
$$

$$
\nabla_{W_j} L_i(W) = \sum\limits_{j \neq y_i } X_i^T 
$$

```python
 dW[:,j] += X[i].T
```


$$
\nabla_{W_{y_i}} L_i(W) = \sum\limits_{j \neq y_i }(-X_i^T)
$$



```python
dW[:,y[i]] += -X[i].T 
```



### Evaluates the numerical gradient

  

```python
loss, grad = svm_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e1)[0]
# 只接收一个参数 w 。本应该返回 (loss, dW) 的元组，[0] 只返回 loss 

grad_numerical = grad_check_sparse(f, W, grad)
```

#### **centered difference formula**

差分法计算数值梯度
$$
{\partial L(W) \over \partial w_{i,j}} \approx {L(w_{i,j}+h) - L(w_{i,j}-h) \over 2h}
$$


```python
oldval = x[ix]

x[ix] = oldval + h  # increment by h
fxph = f(x)  # evaluate f(x + h)
# f(x) 也就是 f(W) 也就是求 W 关于X_dev 的loss
x[ix] = oldval - h  # increment by h
fxmh = f(x)  # evaluate f(x - h)

x[ix] = oldval  # reset

grad_numerical = (fxph - fxmh) / (2 * h)
```

#### Relative Error

```python
rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
```



#### Regularization



$$
L(W) = {1 \over N} \sum\limits^N_{i=1} L_i + \lambda R(W)
$$


$$
R(W) = \sum\limits_k \sum\limits_l W^2_{k,l}
$$



```python
loss /= num_train
loss += reg * np.sum(W*W) # element-wise multiplication
```

$$
\nabla_W\ \lambda R(W) = 2 * \lambda \ W
$$



```python
dW /= num_train
dW += 2 * reg * W
```

**Why regularization ?**

**Prevent the model from doing too well on training data**

```python
loss /= num_train
loss += reg * np.sum(W*W) 

dW /= num_train
dW += 2 * reg * W
```

<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401231920079.png" alt="image-20240123192012964" style="zoom:67%;" />

```python
loss /= num_train
# loss += reg * np.sum(W*W) 
# 无实际意义
dW /= num_train
dW += 2 * reg * W
```

<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401231914041.png" alt="image-20240123191422954" style="zoom: 67%;" />

```python
loss /= num_train
loss += reg * np.sum(W*W) 

dW /= num_train
# dW += 2 * reg * W
# 无实际意义
```

<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401231917481.png" alt="image-20240123191725384" style="zoom:67%;" />

```python
loss /= num_train
# loss += reg * np.sum(W*W) 

dW /= num_train
# dW += 2 * reg * W
```

<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401231915965.png" alt="image-20240123191555866" style="zoom:67%;" />

### Vectorized

$$
L_i = \sum\limits_{j\neq y_i} max(0,s_j - s_{y_i} + 1)
$$

$$
L(W) = {1 \over N} \sum\limits^N_{i=1} L_i + \lambda R(W)
$$

margins 矩阵中第 $i$ 行 第 $j$ 列的数值为： 
$$
max(0,s_{i,j} - s_{i,y_i} + 1)
$$

```python
scores -= correct_class_score
scores += 1 
margin = np.maximum(scores, np.zeros((num_train, num_classes)))
```

margins 矩阵中第 $i$ 行 第 $y_i$​​ 列均为0

```python
margin[np.arange(num_train), y] = 0
```

loss 正则化之前本质上是 margins 矩阵中所有位置的和（两个 $\sum$​ 遍历整个矩阵）

```python
loss = np.sum(margin)
```

$$
\nabla_{W_j} L_i(W) = \sum\limits_{j \neq y_i }X_i^T
$$

$$
\nabla_{W_{y_j}} L_i(W) = \sum\limits_{j \neq y_i } -X_i^T
$$



权重矩阵的 10 列中,第 `y[i]`列对于分数矩阵中第 $i$ 行的影响是 $-X_i^T$ 

其余列对于分数矩阵中第 $i$ 行的损失值的影响都是 $X_i^T$​

`dW` 中的（i，j）反映了权重矩阵 $W$ 的第 j 类这一列对于 第 i 张图片损失值的影响

目前已经有大小为 [3073 x 500] 的 $X^T$​ ,可以通过 [500 x 10] 的`margins`矩阵转变为 [3073 x 10] 的`dW`

 margin 的第一行可以把 $X^T$ 的第一列 $X_1^T$ 中的第一个元素 $x_{1,1}$​ 作用到 `dW` 的第一行中除第 $y_1$ 个元素

只要不配对，错误梯度会增加，正确的梯度会减小，训练的时候会减去学习率乘梯度，正确的权重减小得慢或者变大



### Stochastic Gradient Descent



## Softmax

$$
L_i = -\ln({e^{s_{y_i}} \over \sum\limits_j e^{s_j}})
$$

$$
L = {1 \over N} \sum\limits^N_{i=1} L_i \ + R(W)
$$



### Data preprocessing

1. 删除旧值，加载数据
2. 划分各个集合的大小
3. 通过 `list(range())` 生成下标的列表 mask
4.  mask 切割集合
5. 将图片拉伸成长条
6. 所有训练集减去均值
7. 右侧水平拼接一列 1，用来接收偏置项
8. 打印出各个集合的 shape

### Gradient

#### Numeric instability

Dividing large numbers can be numerically unstable.


$$
\log \ C = -\ max(S)
$$

$$
{e^{s_{y_i}} \over \sum\limits_j e^{s_j}} 
= {Ce^{s_{y_i}} \over C\sum\limits_j e^{s_j}}
= {e^{s_{y_i}+\log\ C} \over \sum\limits_j e^{s_j + \log \ C}}
$$


```python
max_value = max(scores)
	for j in range(num_classes):
		scores[j] -= max_value
```


$$
L_i = -\ln({e^{s_{y_i}} \over \sum\limits_j e^{s_j}}) = -s_{y_i} + ln ({\sum\limits_j e^{s_j}})
$$

$$
= - X_iW_{y_i} + ln ({\sum\limits_{j=0}^9 e^{X_iW_j}})
$$

$$
e^{X_iW_j} = e^{\sum\limits^{3072}_{k=0}x_{i,k}\ w_{k,j}} = \prod_{k=0}^{3072} e^{x_{i,k}\ w_{k,j}}
$$

$$
=e^{x_{i,1}w_{1,j}} \cdot e^{x_{i,2}w_{2,j}} \cdot  …… \cdot e^{x_{i,t}w_{t,j}} \cdot …… \cdot e^{x_{i,3073}w_{3073,j}}
$$







$$
if\ j = y_i:
$$

$$
\text{want} \ \nabla_{W_{y_i}}\ L_i(W)
$$

$$
e^{X_iW_{y_i}} = e^{\sum\limits^{3072}_{k=0}x_{i,k}\ w_{k,y_i}} = \prod_{k=0}^{3072} e^{x_{i,k}\ w_{k,y_i}}
$$

$$
=e^{x_{i,1}w_{1,y_i}} \cdot e^{x_{i,2}w_{2,y_i}} \cdot  …… \cdot e^{x_{i,t}w_{t,y_i}} \cdot …… \cdot e^{x_{i,3073}w_{3073,y_i}}
$$

$$
{\partial L_i(W) \over \partial w_{t,y_i} } 

= -{\partial \ {{X_iW_{y_i}}}\over \partial w_{t,y_i} }
+ 
{1 \over {\sum\limits_{j=0}^9 e^{X_iW_j}}} 
\cdot 
{\partial {\sum\limits_{j=0}^9 e^{X_iW_j}}\over \partial w_{t,y_i} } 

=-{\partial \ {{X_iW_{y_i}}}\over \partial w_{t,y_i} }
+
{\partial \ {e^{X_iW_{y_i}}}\over \partial w_{t,y_i} }
\cdot 
{1 \over {\sum\limits_{j=0}^9 e^{X_iW_j}}} 
$$



*Mutatis mutandis*
$$
= -x_{i,t} + {{x_{i,t} \cdot e^{X_iW_{y_i}} } \over {\sum\limits_{j=0}^9 e^{X_iW_j}}}
$$

$$
= x_{i,t}
\cdot
(-1 
+
{e^{X_iW_{y_i}} 
\over 
{\sum\limits_{j=0}^9 e^{X_iW_j}}})
$$

$$
\nabla_{W_{y_i}}\ L_i(W) 
= 
X_{i}^T
\cdot
( - 1 
+
{e^{X_iW_{y_i}} 
\over 
{\sum\limits_{j=0}^9 e^{X_iW_j}}} )
$$

```python
for j in range(num_classes):
    if j == y[i]:
        dW[:, y[i]] += X[i].T * (-np.exp(scores[y[i]]) + softmax_value)
        
```


$$
if\ j \neq y_i:
$$

$$
\text{want} \ \nabla_{W_j}\ L_i(W)
$$

想得到 $W$ 的第 $j$ 列对损失函数第 $i$ 行的影响，可以先计算 $W$ 第 $j$ 列中第 $t$ 个对 $L_i$ 的影响，再把这些影响攀成
$$
{\partial L_i(W) \over \partial w_{t,j} } 

=  {
1 
\over 
{\sum\limits_{j=0}^9 e^{X_iW_j}}} 
\cdot 
{\partial {\sum\limits_{j=0}^9 e^{X_iW_j}}
\over 
\partial w_{t,j} } 

=  {1 \over {\sum\limits_{j=0}^9 e^{X_iW_j}}} \cdot {\partial \ {e^{X_iW_j}}\over \partial w_{t,j} }
$$

$$
={1 \over {\sum\limits_{j=0}^9 e^{X_iW_j}}} 
\cdot 
{\partial (\ e^{x_{i,1}w_{1,j}} \cdot e^{x_{i,2}w_{2,j}} \cdot  …… \cdot e^{x_{i,t}w_{t,j}} \cdot …… \cdot e^{x_{i,3073}w_{3073,j}})
\over 
\partial w_{t,j} }
$$

$$
= {1 \over {\sum\limits_{j=0}^9 e^{X_iW_j}}} 
\cdot 

{\ x_{i,t} \cdot e^{x_{i,1}w_{1,j}} \cdot e^{x_{i,2}w_{2,j}} \cdot  …… \cdot e^{x_{i,t}w_{t,j}} \cdot …… \cdot e^{x_{i,3073}w_{3073,j}}}
$$

$$
= \ x_{i,t} \ \cdot {{e^{X_iW_j}} \over {\sum\limits_{j=0}^9 e^{X_iW_j}}}
$$

$$
\because
{\partial L_i(W) \over \partial w_{t,j} } = x_{i,t} \ \cdot {{e^{X_iW_j}} \over {\sum\limits_{j=0}^9 e^{X_iW_j}}}
$$

$$
\therefore \nabla_{W_j}\ L_i(W) = X_i^T \ \cdot\ {{e^{X_iW_j}} \over {\sum\limits_{j=0}^9 e^{X_iW_j}}}
$$

```python
 else:
                dW[:, j] +=  X[i].T * (scores[j] / denominator)
```



### Vectorized

$X$ 和 $W$ 不再逐行逐列的相乘，而是整体相乘得到 分数矩阵 $S$

沿着 `axis = 1` 求取最大值和作为分母的总和

将损失值 $L_i$ 排成一个 [num_train, 1] 的数组，求和即可得到 $L_i$ 

margin 的每个点是
$$
e^{s_{i,j}} \over \sum \limits_{j = 0}^9e^{s_{i,j}}
$$

```python
margin = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
```

由于 $\nabla_{W_{y_i}}\ L_i(W) 
= 
X_{i}^T
\cdot
( - 1 
+
{e^{X_iW_{y_i}} 
\over 
{\sum\limits_{j=0}^9 e^{X_iW_j}}} ) $ 其中正确的分类还要 $-1 $

```
margin[np.arange(num_train), list(y)] -= 1;
```

最后用 $X^T$ 左乘



## Two Layer Net

仿射

```python
def affine_forward(x, w, b):
    out = None
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w ,b)
    return out, cache
```

后向传播

```python
def affine_backward(dout, cache):
    x, w, b = cache

    dx, dw, db = None, None, None
   
    dy = dout
    dw = x.reshape(x.shape[0], -1).T.dot(dy)
    dx = dy.dot(w.T).reshape(x.shape)
    db = np.sum(dout, axis=0)

    return dx, dw, db
```

只返回第一个返回值`out`

```python
db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
```



```python
def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    print(it)
    while not it.finished:
        ix = it.multi_index # 依次得到每个元素的多维坐标
   
        oldval = x[ix]
        x[ix] = oldval + h	# 该元素增加一点
        pos = f(x).copy() # 返回只有 x 中的 ix 位置的元素 +h 后的分数矩阵
        x[ix] = oldval - h 
        neg = f(x).copy() # 返回只有 x 中的 ix 位置的元素 -h 后的分数矩阵
        x[ix] = oldval
        # 考虑上游梯度 df
        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad
```

