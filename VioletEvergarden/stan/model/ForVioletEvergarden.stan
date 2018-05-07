data{
  int TimeLength;
  int TermLength;
  int TermPoint[TimeLength];
  real Y[TimeLength];
  int TermIndex[TimeLength];
  real TermBase[TermLength];
  int TermNo[TermLength];
  int base_NumNA;
  int each_NumNA;
}

parameters {
  real<lower=0> base_gamma;
  real<lower=1> r;
  real<lower=1> R;
  real<lower=0, upper=1> each_gamma;
  real base_pred_Y[base_NumNA];
  real each_pred_Y[each_NumNA];
  real<lower=0> base_sigma;
  real<lower=0> each_sigma;
}

transformed parameters {
  real<lower=0> base_mu[TermLength];
  real<lower=0> each_mu[TimeLength];
  //Whole Term level
  for (i in 1:TermLength) {
    base_mu[i] = TermBase[1] / (1 + base_gamma * (TermNo[i] - 1)); 
  }
  //Each Term Level
  for (j in 1:TimeLength) {
    if (TermPoint[j] == 1) {
      each_mu[j] = base_mu[TermIndex[j]];
    } else {
      if (TermIndex[j] == 13) {
        each_mu[j] =  (base_mu[TermIndex[j]] * r + R) * each_gamma ^ (TermPoint[j] - 1);
      } else {
        each_mu[j] =  (base_mu[TermIndex[j]] * r) * each_gamma ^ (TermPoint[j] - 1);
      }
    }
  }

}

model {
  int na_ind_base;
  int na_ind_each;
  na_ind_base = 1;
  na_ind_each = 1;
  //Whole Term Level
  for (i in 1:TermLength) {
    if (TermBase[i] != -9999) {
      TermBase[i] ~ normal(base_mu[i], base_sigma);
    } else {
      base_pred_Y[na_ind_base] ~ normal(base_mu[i], base_sigma);
      na_ind_base = na_ind_base + 1;
    }
  }
  
  for (j in 1:TimeLength) {
    if (Y[j] != -9999) {
      Y[j] ~ normal(each_mu[j], each_sigma);
    } else {
      each_pred_Y[na_ind_each] ~ normal(each_mu[j], each_sigma);
      na_ind_each = na_ind_each + 1;
    }
  }
}
