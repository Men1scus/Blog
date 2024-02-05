# CS231n_ Assignment2

## Fully Connected Nets





## Batch Normalization

>[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (arxiv.org)](https://arxiv.org/abs/1502.03167)

批量归一化相当于在每层进行仿射变换后用 Normalization 和 $y = \gamma \hat{h} \ + \beta$ 替代激活函数

### Forward Pass

在一小步的时间内，根据动量衰减均值和方差

```python
running_mean = momentum * running_mean + (1 - momentum) * sample_mean
running_var = momentum * running_var + (1 - momentum) * sample_var
```

可以理解为少了 momentum 个 sample_mean，但补回来 momentum 个 running_mean

running_mean 起初是0，所以没有补充，running_mean < sample_mean, 之后每次补充的都会比拿走的少，所以逐渐衰减。

### Backward Pass

#### Naive implementation

> [Understanding the backward pass through Batch Normalization Layer](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)
>
> 强烈推荐，分析得极其透彻

借助计算图将梯度逐层分解，而不是套用论文中繁杂的公式

![img](https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202402042348593.png)

发现评论区中有人把网站中的计算图打印下来，一步一步书写草稿过程，忽然有些感动，更多的是惭愧，学习过程中既没有高效也没有踏实，目的甚至都不那么纯粹，亵渎了这门学科。

![image-20240205005606551](https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202402050056868.png)

### Alternative Backward Pass

#### 论文中未化简推导的结果

Naive implementation

<img src="https://cdn.jsdelivr.net/gh/Men1scus/FigureBed@main/img/202402042123491.png" alt="image-20240204212346423" style="zoom: 50%;" />

```python
dx_norm = dout * gamma
dvar = np.sum(dx_norm * (x - mean), axis=0) * (-1 / 2) * np.power(var + eps, (-3 / 2))
dmean = np.sum(dx_norm, axis=0) * (-1 / np.sqrt(var + eps)) + dvar * np.sum(-2 * (x - mean), axis=0) / N
dx = dx_norm * (1 / np.sqrt(var + eps)) + dvar * 2 * (x - mean) / N + dmean / N
dgamma = np.sum(dout * x_norm, axis=0)
dbeta = np.sum(dout, axis=0)
```



#### 化简公式

> [Clement Thorey | What does the gradient flowing through batch normalization looks like ? ](https://cthorey.github.io./blog/2016/backpropagation/)

<img src="https://raw.githubusercontent.com/cs231n/cs231n.github.io/master/assets/a2/batchnorm_graph.png" style="zoom: 33%;" >

将计算图整合成一个模块

“反向传播” 是误差的反向传播
$$
\frac{d \mathcal{L}}{dx_{i,j}} = 
\sum\limits_{k,l} 
\frac {d \mathcal{L}} {dy_{k,l}} 
\frac {dy_{k,l}} {d\hat{x}_{k,l}}
\frac {d\hat{x}_{k,l}} {dx_{i,j}}
$$
注意下标的巧妙之处
$$
\frac {dy_{k,l}} {d\hat{x}_{k,l}} = \gamma_l
$$
$l$ 在此处是下标，代表第 $l$ 列的参数

>Indeed, the gradient of $\hat{h}$ with respect to the $j$ input of the $i$ batch, which is precisely what the left hand term means, is non-zero only for terms in the $j$ dimension

$$
\frac {d\hat{x}_{k,l}} {dx_{i,j}} = 1_{(k=i \ and\ l=j)}\ - \frac{1}{N} 1_{(l=j)}
$$

或者可以理解为 $l=j$​ 是必要条件，只对相同列产生影响

（此处省略推导过程，我的推导没有完全按照他的来）

“For this implementation you should work out the derivatives for the batch normalizaton backward pass on paper and simplify as much as possible”
$$
\frac{\part out}{\part x} = 
\frac{\part out}{\part y} \cdot 
\frac{\part y}{\part \hat x} \cdot 
(\frac{\part \hat x}{\part x} + 
\frac{\part \hat x}{\part \mu} \cdot \frac{\part \mu}{\part x} + 
\frac{\part \hat x}{\part \sigma} \cdot \frac{\part \sigma}{\part v} \cdot \frac{\part v}{\part x})
$$
仍需反复揣度理解 $\frac{1}{N} \sum\limits_{k=1}^N$ 的含义，这里或许写错了，比 Clement 的博客上少了许多步骤
$$
= \mathrm{d}out \cdot \gamma \ \cdot \ 
(\frac{1}{\sigma} + 
\frac{-1}{\sigma} \cdot \frac{1}{N} \sum\limits_{k=1}^N \cdot 1_k+ 
\frac{-(x - \mu)}{\sigma^2} \cdot 
\frac{1}{2} (v + \epsilon)^{-\frac{1}{2}}\ \cdot
\frac{1}{N}\sum\limits_{k=1}^N \cdot 2(x_k - \mu))
$$
“伪公式” 不严谨，借用了 Broadcast 的思想，在此处我的 $1_k$ 指的是输入中第 $k$ 行的 numpy 数组
$$
= \frac{dout \cdot \gamma}{N \cdot \sigma} \cdot 
(N - 
\sum\limits_{k=1}^N \cdot 1_k - 
\frac{x - \mu}{\sigma^2} \cdot \sum\limits_{k=1}^N (x_k - \mu)
$$

$$
\text{let} \ \mathrm{d}tmp = \frac{dout \cdot \gamma}{N \cdot \sigma}
$$

```python
dtmp = dout * gamma / (N * sigma)
```


$$
= \mathrm{d}tmp \cdot 
(N - 
\sum\limits_{k=1}^N \cdot 1_k - 
\frac{x - \mu}{\sigma^2} \cdot \sum\limits_{k=1}^N (x_k - \mu)
$$

$$
= \mathrm{d}tmp \cdot N - 
\sum\limits_{k=1}^N  \mathrm{d}tmp \cdot 1_k - 
\frac{x - \mu}{\sigma^2} \cdot \sum\limits_{k=1}^N \mathrm{d}tmp \cdot (x_k - \mu)
$$

$$
= \mathrm{d}tmp \cdot N - 
\sum\limits_{k=1}^N  \mathrm{d}tmp \cdot 1_k - 
\frac{x - \mu}{\sigma} \cdot \sum\limits_{k=1}^N \mathrm{d}tmp \cdot \frac{(x_k - \mu)}{\sigma}
$$

$$
= \mathrm{d}tmp \cdot N - 
\sum\limits_{k=1}^N  \mathrm{d}tmp \cdot 1_k - 
\hat x \cdot \sum\limits_{k=1}^N \mathrm{d}tmp \cdot \hat x_k
$$

```python
dx = dtmp * N - np.sum(dtmp, axis=0) - x_norm * np.sum(dtmp * x_norm, axis=0)
```

