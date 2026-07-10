# Sampler Lab Continuous Benchmark

- Package version: `0.12.0`
- Replicates: 2
- Base seed: 2022
- Successful runs: 108
- Excluded pairings: 2
- Failed runs: 0

## Aggregate metrics

| Target | Method | Semantics | Exact after freeze | Mean error | Covariance error | IMQ-MMD | Mode L1 | Train seconds | Eval seconds | Total seconds |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bimodal-anisotropic-funnel | adaptive-random-walk | markov-chain | True | 0.04398 | 0.9999 | 0.8518 | 1 | 0.006972 | 0.03672 | 0.04369 |
| correlated-gaussian | adaptive-random-walk | markov-chain | True | 0.7208 | 0.8315 | 0.2866 | -- | 0.003202 | 0.01408 | 0.01728 |
| rotated-anisotropic-funnel | adaptive-random-walk | markov-chain | True | 0.3137 | 1 | 0.9483 | -- | 0.003843 | 0.01888 | 0.02272 |
| separated-anisotropic-gaussian-mixture | adaptive-random-walk | markov-chain | True | 0.6707 | 0.8929 | 0.2993 | 1 | 0.005425 | 0.02599 | 0.03142 |
| bimodal-anisotropic-funnel | annealed-smc | weighted-particles | False | 0.8348 | 0.8393 | 0.7302 | 1 | 0 | 0.1091 | 0.1091 |
| correlated-gaussian | annealed-smc | weighted-particles | False | 0.1321 | 0.2482 | 0.1461 | -- | 0 | 0.02739 | 0.02739 |
| rotated-anisotropic-funnel | annealed-smc | weighted-particles | False | 0.4011 | 0.4919 | 0.5099 | -- | 0 | 0.0451 | 0.0451 |
| separated-anisotropic-gaussian-mixture | annealed-smc | weighted-particles | False | 0.3872 | 0.5658 | 0.2482 | 0.6296 | 0 | 0.05629 | 0.05629 |
| bimodal-anisotropic-funnel | direct-oracle | iid-samples | True | 0.03818 | 0.7639 | 0.08179 | 0.09 | 0 | 0.0001498 | 0.0001498 |
| correlated-gaussian | direct-oracle | iid-samples | True | 0.0797 | 0.1364 | 0.0781 | -- | 0 | 0.0001137 | 0.0001137 |
| rotated-anisotropic-funnel | direct-oracle | iid-samples | True | 0.02408 | 0.6953 | 0.07995 | -- | 0 | 7.948e-05 | 7.948e-05 |
| separated-anisotropic-gaussian-mixture | direct-oracle | iid-samples | True | 0.03788 | 0.1428 | 0.07692 | 0.04667 | 0 | 8.466e-05 | 8.466e-05 |
| bimodal-anisotropic-funnel | hmc | markov-chain | True | 0.2173 | 1 | 0.685 | 1 | 0 | 0.5278 | 0.5278 |
| correlated-gaussian | hmc | markov-chain | True | 0.8143 | 0.8994 | 0.2958 | -- | 0 | 0.07698 | 0.07698 |
| rotated-anisotropic-funnel | hmc | markov-chain | True | 0.07655 | 1 | 0.6953 | -- | 0 | 0.1821 | 0.1821 |
| separated-anisotropic-gaussian-mixture | hmc | markov-chain | True | 0.7433 | 0.9141 | 0.3329 | 1 | 0 | 0.1944 | 0.1944 |
| bimodal-anisotropic-funnel | importance | weighted-samples | True | 1.017 | 0.8847 | 0.8124 | 0.8388 | 0 | 0.01241 | 0.01241 |
| correlated-gaussian | importance | weighted-samples | True | 0.06477 | 0.1407 | 0.07666 | -- | 0 | 0.001897 | 0.001897 |
| rotated-anisotropic-funnel | importance | weighted-samples | True | 0.2162 | 0.6441 | 0.5956 | -- | 0 | 0.004442 | 0.004442 |
| separated-anisotropic-gaussian-mixture | importance | weighted-samples | True | 0.1655 | 0.3653 | 0.1596 | 0.2043 | 0 | 0.006944 | 0.006944 |
| bimodal-anisotropic-funnel | mala | markov-chain | True | 0.1716 | 0.9999 | 0.5542 | 1 | 0 | 0.131 | 0.131 |
| correlated-gaussian | mala | markov-chain | True | 0.8599 | 0.8343 | 0.3332 | -- | 0 | 0.0301 | 0.0301 |
| rotated-anisotropic-funnel | mala | markov-chain | True | 0.6686 | 1 | 0.5823 | -- | 0 | 0.04974 | 0.04974 |
| separated-anisotropic-gaussian-mixture | mala | markov-chain | True | 0.663 | 0.8234 | 0.3011 | 0.7167 | 0 | 0.06768 | 0.06768 |
| bimodal-anisotropic-funnel | policy-gradient-mh | markov-chain | True | 0.2362 | 0.9986 | 0.354 | 1 | 0.009434 | 0.06203 | 0.07146 |
| correlated-gaussian | policy-gradient-mh | markov-chain | True | 0.3471 | 0.7414 | 0.2545 | -- | 0.006333 | 0.02408 | 0.03042 |
| rotated-anisotropic-funnel | policy-gradient-mh | markov-chain | True | 0.07773 | 0.9999 | 0.7358 | -- | 0.008188 | 0.03762 | 0.04581 |
| separated-anisotropic-gaussian-mixture | policy-gradient-mh | markov-chain | True | 0.8293 | 0.7709 | 0.3202 | 0.89 | 0.008501 | 0.04568 | 0.05418 |
| bimodal-anisotropic-funnel | random-walk-mh | markov-chain | True | 0.5184 | 0.9872 | 0.8166 | 1 | 0 | 0.04993 | 0.04993 |
| correlated-gaussian | random-walk-mh | markov-chain | True | 0.3002 | 0.6053 | 0.1527 | -- | 0 | 0.01806 | 0.01806 |
| rotated-anisotropic-funnel | random-walk-mh | markov-chain | True | 0.08123 | 1 | 0.9892 | -- | 0 | 0.02784 | 0.02784 |
| separated-anisotropic-gaussian-mixture | random-walk-mh | markov-chain | True | 0.6111 | 0.6949 | 0.2207 | 0.6333 | 0 | 0.03119 | 0.03119 |
| correlated-gaussian | reverse-kl | approximate-iid-samples | False | 0.04596 | 0.7095 | 0.1092 | -- | 0.005888 | 3.922e-05 | 0.005927 |
| rotated-anisotropic-funnel | reverse-kl | approximate-iid-samples | False | 0.04001 | 0.8171 | 0.1935 | -- | 0.015 | 5.486e-05 | 0.01506 |
| bimodal-anisotropic-funnel | reverse-kl-independence-mh | markov-chain | True | 0.09004 | 1 | 0.9877 | 1 | 0.04792 | 0.05469 | 0.1026 |
| correlated-gaussian | reverse-kl-independence-mh | markov-chain | True | 0.4597 | 0.5731 | 0.3355 | -- | 0.005753 | 0.02373 | 0.02948 |
| rotated-anisotropic-funnel | reverse-kl-independence-mh | markov-chain | True | 0.1169 | 1 | 0.9697 | -- | 0.01612 | 0.03544 | 0.05156 |
| separated-anisotropic-gaussian-mixture | reverse-kl-independence-mh | markov-chain | True | 0.4372 | 0.76 | 0.427 | 0.3633 | 0.02167 | 0.03966 | 0.06133 |
| bimodal-anisotropic-funnel | stochastic-newton | markov-chain | True | 0.1473 | 0.9972 | 0.4511 | 1 | 0 | 0.3277 | 0.3277 |
| correlated-gaussian | stochastic-newton | markov-chain | True | 0.1167 | 0.1417 | 0.08972 | -- | 0 | 0.08287 | 0.08287 |
| rotated-anisotropic-funnel | stochastic-newton | markov-chain | True | 0.1066 | 0.991 | 0.4046 | -- | 0 | 0.141 | 0.141 |
| separated-anisotropic-gaussian-mixture | stochastic-newton | markov-chain | True | 0.5301 | 0.8114 | 0.1501 | 1 | 0 | 0.156 | 0.156 |
| bimodal-anisotropic-funnel | stretch-ensemble | ensemble-chain | True | 0.2322 | 0.9198 | 0.1993 | 0.6833 | 0 | 0.02612 | 0.02612 |
| correlated-gaussian | stretch-ensemble | ensemble-chain | True | 0.7697 | 0.6074 | 0.266 | -- | 0 | 0.008737 | 0.008737 |
| rotated-anisotropic-funnel | stretch-ensemble | ensemble-chain | True | 0.03645 | 0.9937 | 0.1989 | -- | 0 | 0.01319 | 0.01319 |
| separated-anisotropic-gaussian-mixture | stretch-ensemble | ensemble-chain | True | 0.8758 | 0.816 | 0.3055 | 0.9833 | 0 | 0.01716 | 0.01716 |
| bimodal-anisotropic-funnel | svgd | approximate-particles | False | 0.003974 | 0.9997 | 0.3127 | 0.7917 | 0.06068 | 0 | 0.06068 |
| correlated-gaussian | svgd | approximate-particles | False | 0.124 | 0.7976 | 0.2168 | -- | 0.04022 | 0 | 0.04022 |
| rotated-anisotropic-funnel | svgd | approximate-particles | False | 0.00403 | 0.9998 | 0.2854 | -- | 0.05128 | 0 | 0.05128 |
| separated-anisotropic-gaussian-mixture | svgd | approximate-particles | False | 0.1413 | 0.8064 | 0.2547 | 0.5 | 0.04588 | 0 | 0.04588 |
| bimodal-anisotropic-funnel | walk-ensemble | ensemble-chain | True | 0.06753 | 0.9853 | 0.2106 | 0.2533 | 0 | 0.04101 | 0.04101 |
| correlated-gaussian | walk-ensemble | ensemble-chain | True | 0.476 | 0.6931 | 0.2731 | -- | 0 | 0.0213 | 0.0213 |
| rotated-anisotropic-funnel | walk-ensemble | ensemble-chain | True | 0.1578 | 0.9604 | 0.2024 | -- | 0 | 0.02976 | 0.02976 |
| separated-anisotropic-gaussian-mixture | walk-ensemble | ensemble-chain | True | 0.9607 | 0.8849 | 0.3714 | 1 | 0 | 0.03505 | 0.03505 |

## Accuracy-total-time Pareto frontiers

The direct oracle is a reference baseline and is excluded from algorithmic frontiers.

- **bimodal-anisotropic-funnel — exact/corrected:** importance, stretch-ensemble
- **bimodal-anisotropic-funnel — approximate:** svgd
- **correlated-gaussian — exact/corrected:** importance
- **correlated-gaussian — approximate:** reverse-kl
- **rotated-anisotropic-funnel — exact/corrected:** importance, stretch-ensemble
- **rotated-anisotropic-funnel — approximate:** reverse-kl
- **separated-anisotropic-gaussian-mixture — exact/corrected:** importance, stochastic-newton
- **separated-anisotropic-gaussian-mixture — approximate:** svgd, annealed-smc

## Explicit exclusions

- `reverse-kl` x `separated-anisotropic-gaussian-mixture`: sampler adapter does not support multimodal targets
- `reverse-kl` x `bimodal-anisotropic-funnel`: sampler adapter does not support multimodal targets
