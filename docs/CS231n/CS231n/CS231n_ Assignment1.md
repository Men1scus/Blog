# CS231n Assignment1

## K-Nearest-Neighbor

### Frobenius norm

Frobenius 范数

用来衡量矩阵的大小
$$
\Vert A \Vert = \sqrt{\sum\limits_{i,j} A^2_{i,j}}
$$


逐元素平方和，再开根号

```
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
# margin 值为正时才累加，否则不用加，因为最小值一定不小于0
margin = score[j] - scores[y[i]] + 1
if margin > 0:
	loss += margin
```

#### Regularization

$$
L(W) = {1\over N} \sum\limits^N_{i=1} L_i + \lambda R(W)	\\
R(W) = \sum\limits_k \sum\limits_l W^2_{k,l}
$$



```python
loss /= num_train
loss += reg * np.sum(W*W) # element-wise multiplication
```



### Vectorized

