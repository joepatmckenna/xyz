---
title: brushing up on attention
date: '2023-07-15'
---

<script>
   import ScaledDotProductAttention from "$lib/assets/scaled_dot_product_attention.png"
</script>

## Attention

The goal of the attention mechanism in machine learning is to produce outputs by paying attention to -- weighing -- the inputs differently for each output. Given sets of input queries and key-value pairs, the attention mechanism outputs for each query a weighted sum of values where the weights represent the compatibility between the query and respective key-value pairs. A primary feature of attention that distinguishes it from a standard feedforward mechanism is that the weighing of input values is computed from other inputs, queries and keys, instead of simply being read from memory.

## Scaled Dot-Product Attention

The seminal paper [Attention is All You Need](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf) presented a version of attention called _Scaled Dot-Product Attention_:

$$
\begin{align*}
&\text{Attention}(Q, K, V) = PV \\
&\text{ with }P=\text{softmax}(W) \\
&\text{ and }W = \frac{QK^T}{\sqrt{d_k}}
\end{align*}
$$

In this formula, $Q$, $K$, and $V$ are matrices whose rows are queries, keys, and values respectively. The matrices have shapes

| matrix | shape            |
| ------ | ---------------- |
| $Q$    | $n_q \times d_k$ |
| $K$    | $n_k \times d_k$ |
| $V$    | $n_k \times d_v$ |

The variables appearing in the shape column above have the following meanings

| variable | meaning                 |
| -------- | ----------------------- |
| $n_q$    | number of queries       |
| $n_k$    | number of keys          |
| $d_k$    | dimension of each key   |
| $d_v$    | dimension of each value |

Notice that queries and keys have the same dimension $d_k$. This is to define the compatibility between a query and a key-value pair in terms of the dot-product between the query and key. The compatibility for each query and key combination is recorded in the matrix $P$ often referred to as the _attention weights_. If $p_{ij}$ is relatively large, the attention mechanism is explaining that output $i$ (row $i$ of $PV$) is _attending to_ input value $j$ (row $j$ of $V$).

The softmax in the formula above is applied to each row of $W$ separately, so the $i^{th}$ output is a [convex combination](https://en.wikipedia.org/wiki/Convex_combination) of the input values:

$$
\sum_{j=1}^{n_v}p_{ij}v_j\text{ with }p_{ij}=\frac{e_{ij}}{\sum_{\mu}e_{i\mu}}\text{ and }e_{ij}=\exp\left(\frac{q_ik_j}{\sqrt{d_k}}\right).
$$

In other words, each output is a weighted average of input values with non-negative weights summing to 1. This has interesting implications, such as each output is in the convex hull of the input values. This contributes to the stability of the mechanism, as the convex hull of the outputs has less volume than that of the input values.

### Implementation

Below is a simple Python implementation, using unbatched (2D) NumPy arrays, that mirrors the figure from the Attention is All You Need paper.

<img src={ScaledDotProductAttention} alt="Scaled Dot-Product Attention" width="200"/>

```python
import scipy
import numpy as np

def scaled_dot_product_attention(
  Q: np.ndarray,
  K: np.ndarray,
  V: np.ndarray,
  mask: np.ndarray | None = None,
) -> np.ndarray:
  """Scaled Dot-Product Attention.

  Args:
    Q: queries               (n_q, d_k)
    K: keys                  (n_k, d_k)
    V: values                (n_k, d_v)
    mask: true means ignore  (n_q, n_k)

  Returns:
    convex combinations of values weighted
      by scaled query-key dot-products
  """
  W = np.matmul(Q, K.T)            # MatMul
  W /= np.sqrt(K.shape[1])         # Scale
  if mask is not None:             # (opt.)
    W[np.where(mask)] = -np.inf    # Mask
  P = scipy.special.softmax(W, 1)  # SoftMax
  return np.matmul(P, V)           # MatMul
```
