# Normalizer Caching

## Formulation

Let $x_k \in \mathbb{R}^N$ denote the pooled observations of feature $k$ over all (date, instrument) pairs in the fit window $[t_0, t_1]$, where $N \approx 1.23 \times 10^6$.

The `RobustZScoreNorm` processor computes:

$$
m_k = \text{med}(x_k), \qquad s_k = 1.4826 \cdot \text{MAD}(x_k) + \varepsilon
$$

where $\text{MAD}(x) = \text{med}(|x - \text{med}(x)|)$ and $\varepsilon = 10^{-12}$.

The transform applied at inference is:

$$
z_k = \text{clip}\!\left(\frac{x_k - m_k}{s_k},\; -3,\; 3\right)
$$

This transform is fully determined by the pair $(m_k, s_k)$. For 22 features, 44 scalars constitute a sufficient statistic for the normalization map. Caching these parameters and discarding the raw fit data produces bit identical output.

## Convergence

The asymptotic standard error of the sample median for $n$ i.i.d. observations with density $f$ at the median is $\text{SE} = (2 f(m) \sqrt{n})^{-1}$. With $n = 1.23 \times 10^6$ and $f(m) \approx 0.4$, this gives $\text{SE} \approx 10^{-3}$. The MAD inherits identical convergence as a median of absolute deviations. Both estimators are effectively converged to their population values.

## Robustness

The median and MAD each have a breakdown point of 50%, the theoretical maximum. Neither can be corrupted unless more than half the sample is adversarial. This is the reason Qlib employs `RobustZScoreNorm` rather than standard $(mean, std)$ normalization, which has a breakdown point of 0%.

## Refitting on recent data is invalid

Let the cached parameters be $(m, s)$ and a hypothetical refit on 200 days yield $(m', s')$. The induced distortion

$$
z' - z = x\left(\frac{1}{s'} - \frac{1}{s}\right) + \left(\frac{m}{s} - \frac{m'}{s'}\right)
$$

is affine in $x$, introducing covariate shift $P_{\text{train}}(z) \neq P_{\text{inference}}(z)$ that violates the identical distribution assumption under which the model was trained.
