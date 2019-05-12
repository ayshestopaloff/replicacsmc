Instructions to reproduce the results of ``Replica Conditional Sequential Monte Carlo’’ 
by Shestopaloff and Doucet (2019).

***Section 4.1***

Experiment 1. Verifying that the replica method produces results that agree
with the posterior mean produced by a Kalman smoother.

- Run lg_exp_1.m to produce the .mat files with the MCMC samples. Then, use mu_check.m
to check agreement with the KF smoother mean.

Experiment 2. The effect of using more replicas and a constant versus approximate
predictive density.

- Run lg_exp_2_exact_2.m to produce MCMC samples using exact predictive density
and 2 replicas.

- Run lg_exp_2_exact_75.m to produce MCMC samples using exact predictive density
and 75 replicas.

- Run lg_exp_2_approx_75.m to produce MCMC samples using constant predictive density
and 75 replicas.

Then use tau_batch.m to estimate autocorrelation times.

Experiment 3. A demonstration that a fixed level of precision can be achieved with
much fewer particles when using replica cSMC versus standard iterated cSMC.

- Run lg_exp_3_prior.m to produce .mat files with the MCMC samples for standard iterated
cSMC and lg_exp_3_rep.m for replica cSMC. Then, use mu_check.m to compute standard
errors for x_{1,1} for both samplers.

Experiment 4. A verification that replica cSMC works well on longer sequences.

- Run lg_exp_4.m to produce .mat files with the MCMC samples. Then, use tau_batch.m
to estimate the autocorrelation time for the different latent variables. 

***Section 4.2***

Model 1 experiment.

- Run pg_sample_m1_approx.m to produce the MCMC samples and use tau_batch.m to estimate
autocorrelation times.

Model 2 experiment.

- Run pg_sample_m2_approx.m to produce the MCMC samples and use model_2_fig.m to make the
plots in Figure 6.

***Section 4.3***

Lorenz-96 model.

- Run pg_sample_lorenz_prior.m and pg_sample_lorenz_rep.m to produce the .mat files
with the MCMC samples and use lorenz_fig.m to produce the plots in Figure 7.
