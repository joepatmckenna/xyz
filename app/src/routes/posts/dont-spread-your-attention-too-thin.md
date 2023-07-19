---
title: don't spread your attention too thin
date: '2023-07-18'
---

<script>
   import MaeVsWeightDensity from "$lib/assets/MAE_vs_attention_weights_density.png"
</script>

## Computational cost of Scaled Dot-Product Attention

The computational cost of _Scaled Dot-Product Attention_:

$$
\begin{align*}
&\text{Attention}(Q, K, V) = PV \\
&\text{ with }P=\text{softmax}(W) \\
&\text{ and }W = \frac{QK^T}{\sqrt{d_k}}
\end{align*}
$$

is dominated by two matrix products:

| product | shapes                           | cost                    |
| ------- | -------------------------------- | ----------------------- |
| $QK^T$  | $(n_q\times d_k)(d_k\times n_k)$ | $\mathcal O(n_qn_kd_k)$ |
| $PV$    | $(n_q\times n_k)(n_k\times d_v)$ | $\mathcal O(n_qn_kd_v)$ |

Therefore, the total cost is $O(n_qn_kd_k + n_qn_kd_v)$. The $n_q$ and $n_k$ factors restrict the maximum input/output sequence lengths that can be processed on a given platform.

#### Background: computational cost of matrix multiplication

The cost of multiplying real $n\times k$ and $k\times m$ matrices $A$ and $B$ using the most basic algorithm is $\mathcal O(nmk)$ because for each of the $nm$ entries of $AB$ an $\mathcal O(k)$ dot-product $\textstyle\sum_{\mu=1}^ka_{i\mu}b_{\mu j}$ is computed. [More efficient algorithms](https://en.wikipedia.org/wiki/Computational_complexity_of_matrix_multiplication) are used in practice, but analysis in terms of the most basic algorithm is still used to broadly compare methods involving matrix multiplication.

## Thresholding the attention weights

Each output of Scaled Dot-Product Attention (row of $PV$) is a weighted average of the input values (rows of $V$) in which the weights represent similarities between the respective query and all key-value pairs. This original formulation assumes that all inputs contribute to all outputs. If instead we assume that for each query only some values contribute to the respective output, then we may be able to reduce the overall cost.

Here, I consider a simple approach for selecting only some values per query to contribute to the respective output: thresholding attention weights. This approach can be seen as an inference hack in light of [all of the effort](https://arxiv.org/pdf/2009.06732.pdf) that has been dedicated to making Transformers more efficient.

I consider two methods of thresholding:

- **Top-k**: keep the top $k<n_k$ values in each row of $P$, or
- **Top-p**: keep the fewest values whose sum exceeds $p<1$ in each row of $P$.

Both methods, by reducing the size of $P$, reduce only the cost of the second matrix product in Scaled Dot-Product Attention, $PV$:

| Method    | shapes                           | cost                    |
| --------- | -------------------------------- | ----------------------- |
| **Top-k** | $(n_q\times k)(k\times d_v)$     | $\mathcal O(n_qkd_v)$   |
| **Top-p** | $(n_q\times k_p)(k_p\times d_v)$ | $\mathcal O(n_qk_pd_v)$ |

where $k_p$ is the average number of entries less than $p$ in each row of $P$. Now, the total cost is $\mathcal O(n_qn_kd_k + n_qkd_v)$. Thus, if the values have greater dimension than the queries (encodings have greater dimension than that of the decoder outputs in an encoder-decoder Transformer), thresholding $P$ can reduce the total cost upper bound. This doesn't apply for self-attention or decoder-only transformers where $d_k=d_v$, but nonetheless the actual number of operations is reduced.

### Thresholding error

How different are the ouputs when the attention weights are thresholded? And, how many attention weights can we ignore without changing the outputs too much? To answer these questions, I quantified the mean absolute difference between the original output $PV$ and the modified output with thresolded $P$ for the full range of valid thrsholds. I varied the sequence length $n_q=n_k$ across runs. All values were uniform random samples. Interestingly, the error curve remains low and flat for a decent range while the attention weight compression is dialed up:

<img src={MaeVsWeightDensity} alt="Error from clipping attention weights" width="100%"/>

This suggests that a decent amout of computation cost can be realized by sparsifying the attention weights without having a huge impact on performance. But one would need to do a more comprehensive analysis with a full Transformer on real data to have confidence in that suggestion.

### Scaled Dot-Product Attention Implementations

I implemented both methods of thresholding by extending the optional masking logic of the origial Scaled Dot-Product Attention implementation. These implementations don't realize any time complexity reduction -- the unnecessary multiplications are still present as zero multiplications -- but this setup allowed me to quantify the thresholding error.

First, here is a simple Python implementation, using unbatched (2D) NumPy arrays, of the original Scaled Dot-Product Attention. Then, below are modifications that include Top-k and Top-p thresholding.

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

#### Top-k Thresholding Scaled Dot-Product Attention

```python
def scaled_dot_product_attention_topk(
  Q: np.ndarray,
  K: np.ndarray,
  V: np.ndarray,
  topk: int,
  mask: np.ndarray | None = None,
) -> np.ndarray:
  W = np.matmul(Q, K.T)                 # MatMul
  W /= np.sqrt(K.shape[1])              # Scale
  if mask is not None:                  # (opt.)
    W[np.where(mask)] = -np.inf         # Mask
  Wk = -np.sort(-W, 1)[:, topk-1:topk]  # Top-k
  W[np.where(W < Wk)] = -np.inf         # ...
  P = scipy.special.softmax(W, 1)       # SoftMax
  return np.matmul(P, V)                # MatMul
```

#### Top-p Thresholding Scaled Dot-Product Attention

```python
def scaled_dot_product_attention_topp(
  Q: np.ndarray,
  K: np.ndarray,
  V: np.ndarray,
  topp: float,
  mask: np.ndarray | None = None,
) -> np.ndarray:
  W = np.matmul(Q, K.T)                # MatMul
  W /= np.sqrt(K.shape[1])             # Scale
  if mask is not None:                 # (opt.)
    W[np.where(mask)] = -np.inf        # Mask
  P = scipy.special.softmax(W, 1)      # Top-p
  R = np.arange(P.shape[0])[:, None]   # ...
  C = W.argsort(1).argsort(1)          # ...
  P = np.sort(P).cumsum(1)[R, C]       # ...
  W[np.where(P < 1 - topp)] = -np.inf  # ...
  P = scipy.special.softmax(W, 1)      # SoftMax
  return np.matmul(P, V)               # MatMul
```
