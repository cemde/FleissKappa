# Fleiss Kappa Torchmetrics

A torchmetrics (Pytorch) implementation of Fleiss Kappa for interrater agreement.

## Fleiss Kappa

While Cohens Kappa can only measure the inter-rater agreement of 2 raters, Fleiss Kappa is a measure of inter-rater agreement for an arbitrary number of raters and classes.

## Implementation

This implementation is meant to be used for Pytorch. It follows the `torchmetrics` guidelines and inherits from `torchmetrics.Metric`. The metric can be importet from `fleiss_kappa.py`.

## Tests

My `torchmetrics` implementation is tested against the `statsmodels` package. Both implementations yield the same results up to a tolerance of 0.0005. The following test protocol was used:

1 A correlation matrix is sampled for the raters.
2 Predictive logits are sampled from a Multivariate normal distribution using this covariance matrix. 
3 The `torchmetric` implementation is computed as implemented from the `logits`.
4 The logits are transformed into count data (counts of raters per class for each participant).
5 The `statsmodels` implementation is used on the count data.
6 Both metric outputs are compared.

In 100k runs, with a tolerance of 0.0005, number of raters and classes varying between 2 and 20, rater variances between 0 and 1, and number of subjects varying between 10 and 100, no errors were found.

Tests conducted on
- CPU
- MacOS 11.4
- Python 3.9.2
- Pytorch 1.9.0
- Pytorch Lightning 1.3.8
- Numpy 1.20.2
- torchmetrics 0.4.1
- statsmodels 0.12.2

## Contribute

I appreciate contributions.

## References

- [Wikipedia](https://en.wikipedia.org/wiki/Fleiss%27_kappa)
- Fleiss, J. L. (1971) "Measuring nominal scale agreement among many raters." Psychological Bulletin, Vol. 76, No. 5 pp. 378–382