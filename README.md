The code in the repo deals with the problem of modeling the discrepancies
between two sets of observations.

Imagine that you have observations v_1 with Gaussian error e_1,
and observation v_2 with Gaussian error e_2.

We model dv = v_1 - v_2 by a mixture model
P(dv) = f_out * P(dv|outl ) + (1-f_out) * P(dv|good)
where two terms refer to outlier model P(dv|outl) and
good population model
P(dv|good)

All the modeling is done in the interval -vlim<dv<vlim , i.e. we don't model
the distribution outside that interval.

The model for outliers is

$ dv|outl \sim N(0,sig_outl) $

$dv|good \sim N(off,\sqrt{(e_1^2 + e_2^2)*mult^2 + floor^2} )$

