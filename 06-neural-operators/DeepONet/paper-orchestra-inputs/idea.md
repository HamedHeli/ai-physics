# Idea summary

**Working title:** Learning only what physics can't give: a QLV-structured neural operator for fast, joint elastic-viscous characterization from indentation.

**Problem.** Soft tissues and elastomers are characterized with ramp-and-hold flat-punch
indentation, using a quasi-linear viscoelastic (QLV) model: Ogden elasticity (mu, alpha) plus
a Prony relaxation function G(t) = 1 + sum_k g_k exp(-t/tau_k). Each finite-element (Abaqus,
UMAT) simulation takes about 10 hours. Prior work by the authors (Helisaz, Bacca & Chiao,
Exp. Mech. 2022) built a fast table-based method for the g_k. It could not identify the elastic
and viscous parameters together without slow inverse FE.

**Key observation.** That work showed the relaxation is independent of indenter size and
depth. So the indenter response factors exactly into (i) an elastic indentation law
sigma_E(delta/phi, alpha, phi), which needs FE, and (ii) a Prony convolution, which is exact and cheap.

**Method.** A DeepONet whose trunk basis functions are produced by a differentiable Prony
convolution layer. B_k(t) = int exp(-(t-s)/tau_k) d sigma_E(s), and the branch coefficients are
(1, g_1, g_2, g_3). Only sigma_E is learned, as a 3-input MLP in log-log space. Inversion is
multi-start gradient descent through the differentiable surrogate. Uncertainty comes from a
Gauss-Newton (Laplace) approximation.

**Claims.** (1) Much better data efficiency and extrapolation than a black-box DeepONet.
(2) Joint identification of mu, alpha and g_k in about 1 s, compared with hours for inverse FE.
(3) Identifiability analysis that turns protocol rules (tau_i/tau_0 >= 10; hold >= 3 tau_max)
into quantitative design guidance. (4) Validation on Abaqus simulations, and planned
validation on Ecoflex 00-10/30/50 experiments against uniaxial compression.

**Baselines.** Black-box DeepONet. Classical least squares with a known elastic law. The
paper's own eq. (13)/(14) method. Neural-network inverse indentation (Huber & Tsakmakis;
Lu et al., PNAS 2020).
