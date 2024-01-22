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

## SVM

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
\begin{aligned}
L_i = \sum\limits_{j\neq y_i} max(0,s_j - s_{y_i} + 1)
\\\\
if \ s_{y_i}  \geq s_j + 1 :
\\\\
L_i = \sum\limits_{j \neq y_i } s_j - s_{y_i} + 1 = \sum\limits_{j \neq y_i } X_iW_j - X_iW_{y_j} + 1
\\ 
\end{aligned}
$$


<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401222000951.png" alt="image-20240122200000817"  />
$$
if\ j \neq y_i
\\\\
L_i(W) =  X_iW_j\ + ……  = \ \sum\limits_{k=1}^{3072} x_{i,k} w_{k,j}+……
\\\\
{\partial L_i(W) \over \partial w_{t,j} } =  \ {\partial\ \over \partial w_{t,j} } \sum\limits_{k=1}^{3072}  {x_{i,k} w_{k,j}} = {\partial\ \over \partial w_{t,j}}(\ x_{i,1}w_{1,j} + x_{i,2}w_{2,j} + …… + x_{i,t}w_{t,j} + …… +x_{i,3072}w_{3072,j}\ )
\\\\
=x_{i,t}
$$
<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202401222124092.png" alt="image-20240122212403960"  />

*From this we can easily see that* 
$$
\nabla_{W_j} L_i(W) = \nabla_{W_j}X_iW_j = X_i^T
$$
*Mutatis mutandis*
$$
\nabla_{W_{y_j}} L_i(W) = \nabla_{W_{y_j}}(-X_iW_{y_i}) = -X_i^T
$$

$$
\nabla_W L(W)
\newline
\nabla_{W_j} L_i(W) = X_i^T (j \neq y_i)
\newline
\nabla_{W_{y_j}} L_i(W) = -X_i^T
$$






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
\newline
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

### Vectorized

