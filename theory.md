# Rocora: is the residual worth its cost?

Rocora means **Residual Only Coded Optimization with a Regularized Anchor**.
The research question is:

> Does adding residual-only coded curvature measurements to a positive
> Gauss–Newton/Fisher core improve language-model updates enough to pay for the
> extra computation?

This is an experimental question. The decomposition below, positive curvature
core, Hessian-vector products, and randomized diagonal estimators are established
tools. Formalizing their algebra does not establish novelty or training gains.
The proposed contribution is evidence about the value of a specific residual
channel, using a controlled core-only baseline and measured TPU cost.

## Established decomposition

For logits f(theta) and a smooth scalar loss ell, the exact chain rule gives

    H = S + R
    S = J_f^T (Hessian_logits ell) J_f
    R = sum_a (d ell / d f_a) Hessian_theta f_a

S is the generalized Gauss–Newton matrix. For softmax cross-entropy it is also
the model Fisher and is positive semidefinite. R is signed model curvature.
This is the standard GGN-plus-residual decomposition, not a new identity.
[ViViT](https://arxiv.org/abs/2106.02624) and
[Martens](https://jmlr.org/papers/v21/17-678.html) describe this curvature setting.

R is linear in the **logit loss gradient**, lambda. At a fixed model Jacobian and
second derivative, scaling lambda by c scales R by c. This does not mean R
necessarily shrinks monotonically during training: model derivatives also change.
Stationarity in parameter space, J_f^T lambda = 0, does not imply lambda = 0.
An NLP model can retain nonzero per-token logit loss gradients near a training
optimum. We will measure residual size and utility instead of assuming collapse.

## The nonnegative anchor

For a mean cross-entropy over N active tokens, independently sample labels from
the model distribution. Let g_tilde be the gradient of that sampled-label mean
loss. Conditional on the fixed batch and parameters,

    E[N * g_tilde^2] = diag(S)

where the square is coordinatewise. The sample is always nonnegative. It is an
unbiased *sample*, not an exact diagonal. Sampled-label gradients from each TPU
device are averaged with token-count weights before squaring; the result is
multiplied by the global active-token count. This matches the global Bartlett
sample in Sophia-G. A squared real-label gradient would estimate a different
quantity.

Sophia-G uses this Gauss–Newton–Bartlett core. Its original implementation stores
g_tilde^2 and multiplies the denominator by batch token count. We instead store
N*g_tilde^2, so the same normalized diagonal goes directly into the denominator.
Both use the same curvature batch size for every event in the main experiments.
See [Sophia](https://arxiv.org/abs/2305.14342) and its
[official code](https://github.com/Liuhong99/Sophia).

## Residual-only coded measurements

For a symmetric matrix M and a sign vector v,

    v_i (M v)_i = M_ii + sum_(j != i) M_ij v_i v_j.

Assign each parameter coordinate a column of an m-by-m Sylvester Hadamard matrix.
Take its m rows, multiplied coordinatewise by one independent random sign vector
d shared within the group. For the group-mean estimator, conditional on a fixed M,

    E[estimate_i] = M_ii
    Var(estimate_i) = sum_(j != i) M_ij^2 * Cbar_ij^2

where Cbar is the average product of the assigned code columns. Different columns
have zero coherence; collisions have nonzero coherence. Latin assignment across
the last two tensor dimensions is a heuristic for reducing relevant collisions.
It neither proves sparsity of the neural-network Hessian nor cancels every
same-row or same-column pair in matrices wider than m.

Rocora applies this estimator to **R**, and obtains diag(S) through sampled
labels. Residual probing variance then depends on R alone. At fixed model
derivatives, it scales quadratically when lambda is scaled. The *total*
preconditioner remains noisy because its sampled core, minibatch selection,
stale EMA state, and numerical arithmetic also contribute error. There is no
guaranteed zero noise floor for the implemented optimizer.
Residual-only variance is not universally lower than full-Hessian probing
variance: S and R can cancel in the off-diagonal entries of H.

The code measures every row of one coded group at the **same weights and batch**,
using a fresh random flip for the next group. Spreading a group across optimizer
updates would replace one fixed M with changing matrices and would invalidate
the exact cancellation statement. The default m=4 is an engineering cost choice;
it can be changed independently of the curvature refresh interval.

## Direct residual products

At the measurement snapshot theta_0, freeze
lambda = grad_logits ell(f(theta_0)). Define

    phi(theta) = <stop_gradient(lambda), f(theta)>.

Then Hessian_theta phi(theta_0) = R, so forward-over-reverse products of phi
measure Rv directly. This is equivalent to Hv-Sv in exact arithmetic while
avoiding subtraction error and differentiation of the loss's softmax.
The implementation caches one gradient linearization and applies it to the m
sign probes. It never materializes a parameter Hessian.

Normal training/core events use BF16 matrix operations and TPU FlashAttention.
The residual channel uses FP32 master parameters and tiled JAX attention, because
the installed FlashAttention kernel does not implement second derivatives.
The attention formulas agree in exact arithmetic; their finite-precision
differences are numerical implementation error, not an exact split guarantee.
Dense toy tests check the implemented direct operator against H-S.

## Matched updates

All methods use clipped training gradients and decoupled weight decay. AdamW
uses its usual bias-corrected first/second gradient moments. Sophia-G and Rocora
share an uncorrected momentum EMA m and core EMA s. Rocora additionally keeps a
residual EMA r, refreshed only at complete measurement groups.

    h = s                         Sophia-G
    h = s + w*r                   Rocora
    denominator = max(rho*h, 0) + epsilon
    update = clip(m / denominator, -1, 1)
    theta_next = (1 - lr*weight_decay)*theta - lr*update

The positive s is the anchor; epsilon regularizes the denominator, and clipping
bounds each coordinate's update when curvature is negative or uncertain.
For w=0, Rocora matches Sophia-G exactly. Neither clipping nor an accurate Hessian
diagonal proves a better nonconvex training trajectory. Signed residuals may
improve scaling, do nothing, or worsen it; the experiment must distinguish these.

## Study and decision

1. 125M scale: 124,373,760-parameter GPT, seeds 1337/1338/1339, exactly 2.5B processed
   FineWeb-Edu training tokens per run.
2. 350M scale: 354,599,936 parameters and exactly 7B FineWeb-Edu tokens, seed 1337.
3. 1B scale: 984,909,312 parameters and exactly 20B FineWeb-Edu tokens, seed 1337.
   The nominal 125M/350M/1B stages include SOAP and Muon practical baselines.
   Data must come from sufficiently large, document-separated archives.
4. At every scale: identical architectures, data ordering
   within seed, token budgets, evaluation windows and hardware. Sophia-G/Rocora
   share core/momentum/update hyperparameters; residual weight zero is an explicit
   implementation equivalence gate. Learning-rate sensitivity can be explored
   with equal tuning budgets, recorded separately from the matched residual study.

The short-budget sweeps and checked-in starting rates are specified in
[docs/optimizers.md](docs/optimizers.md). They allocate equal processed tokens
and equal candidate counts, not equal compute time. Selection by Sophia's loss
keeps Sophia/Rocora settings matched; independent selection is a different,
explicitly labeled practical comparison. Tuning validation is not an independent
test set. Short-budget ranking may not transfer to the main budget. The 350M/1B
single-seed recipes cannot estimate between-seed variability.

SOAP's Gram matrices are not the sampled-label Sophia core. SOAP maintains Adam
moments in its preconditioner eigenbasis. Its QR basis refresh transports the
first moment exactly (up to arithmetic) but only permutes the diagonal second
moment, following the authors' implementation; it does not reconstruct a full
rotated covariance. Axes larger than 2048 and all one-dimensional gain tensors
are unpreconditioned. Per-layer matrix shapes are preserved even though model
parameters are stored in stacks. Muon uses a finite BF16 quintic approximation
to orthogonalization, with a separate AdamW update on tied input/output,
position and normalization parameters. Neither optimizer implies an exact
curvature inverse. FP32 bases/Grams and BF16 normal model operations introduce
numerical error. Bounded runtime checks do not establish convergence or tuning
quality at any scale.

For Muon, an independent singular-value calculation checks the FP32 finite
polynomial direction. The JAX BF16 implementation also matches nine small
momentum/direction vectors generated from the pinned author implementation on
CPU. This checks rounding behavior against that reference, not the optimizer's
effect on final language-model loss. BF16 arithmetic and the finite polynomial
remain approximations to exact orthogonalization; bounded tests do not establish
their training quality.

The primary comparison is **Rocora versus Sophia-G**. Report final heldout loss
and perplexity at equal tokens, paired seed differences, variability across
seeds, and heldout loss at equal measured training time. Equal-time values between
evaluation points are labeled interpolated. Report curvature overhead and
compile/evaluation/checkpoint costs separately. A token-efficiency gain that
disappears at equal training time has not paid its compute cost under that recipe.

AdamW establishes practical competitiveness. Beating AdamW alone cannot attribute
the improvement to R. One favorable run cannot establish reproducibility or
superiority at larger scale; a null or negative residual result is a valid answer
to the stated question.

## Formal artifact scope

[lean/Rocora.lean](lean/Rocora.lean) retains the carrier composition, exact split,
fixed-matrix probing algebra, and quadratic step identities from the original
development, under the new namespace. These statements concern smooth functions,
exact arithmetic and explicit hypotheses. They do not prove sparse residual
structure, a negligible core estimator variance, superiority to Sophia/AdamW,
TPU performance, or convergence of this nonconvex optimizer. Numerical tests and
full-budget benchmarks supply separate evidence for those implementation and
empirical questions.
