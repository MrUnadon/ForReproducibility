//Bayesian Deep Learning
functions {
  // activation function
  matrix act(matrix X) {
    matrix[rows(X), cols(X)] out;
    for(i in 1:rows(X)) {
      for(j in 1:cols(X)) {
        out[i, j] = inv_logit(X[i,j]) * 2 - 1;
      }
    }
    return out;
  }
}



data {
  int N_train; // train rows
  int N_test; // test rows
  int V; // Number of Feature
  int K; // Number of class label
  matrix[N_train, V] X_train; // Train Input
  int Y_train[N_train]; // Train Label
  matrix[N_test, V] X_test; // Test Input
  
  int M; // Number of Middle Layer
  int NumNode_1st; // Number of nodes in first layer
  int NumNode_2nd; // Number of nodes in second layer
  int NumNode_3rd; // Number of nodes in third layer
}

transformed data {
  vector[NumNode_3rd] zero_vector;
  zero_vector = rep_vector(0, NumNode_3rd); // for softmax
}

parameters {
  vector[M] bias;
  matrix[NumNode_1st, V]  w_in; // weight input 
  matrix[NumNode_2nd, NumNode_1st] w_1;  // weight 1st to 2nd
  matrix[NumNode_3rd, NumNode_2nd] w_2;  // weight 2nd to 3rd
  matrix[NumNode_3rd, K - 1] w_3;  // weight 3rd to output
}



transformed parameters {
  //matrix[N_train, NumNode_1st] v_1; // value 1st layer
  //matrix[N_train, NumNode_2nd] v_2; // value 2nd layer
  //matrix[N_train, NumNode_3rd] v_3; // value 3rd layer
  //matrix[NumNode_3rd, K] w_3_bind_zero; // for softmax weight
  matrix[N_train, K] v_categorical_out; // value output layer

  //from input to first layer
    //v_1 = act(bias[1] + X_train * w_in');
  //from first layer to scond layer
    //v_2 = act(bias[2] + v_1 * w_1');
  //from second to third layer
    //v_3 = act(bias[3] + v_2 * w_2');
  //from third to output layer
  //w_3_bind_zero = append_col(zero_vector, w_3);
  v_categorical_out = act(bias[3] + act(bias[2] + act(bias[1] + X_train * w_in') * w_1') * w_2') * append_col(zero_vector, w_3);
}



model {
  // priors
  bias ~ normal(0, 1);
  to_vector(w_in) ~ normal(0, 1); // weight input
  to_vector(w_1) ~ normal(0, 1); // weight 1st
  to_vector(w_2) ~ normal(0, 1); // weight 2nd 
  to_vector(w_3) ~ normal(0, 1); // weight 3rd
  // output and Y
  for(i in 1:N_train)
    Y_train[i] ~ categorical_logit(v_categorical_out[i, ]');
} 


generated quantities {
  matrix[K, N_test] p;
  //matrix[N_test, NumNode_1st] v_1_test; // value 1st layer
  //matrix[N_test, NumNode_2nd] v_2_test; // value 2nd layer
  //matrix[N_test, NumNode_3rd] v_3_test; // value 3rd layer
  matrix[N_test, K] v_categorical_out_test; // value output layer

  //from input to first layer
    //v_1_test = act(bias[1] + X_test * w_in');
  //from first layer to scond layer
    //v_2_test = act(bias[2] + v_1_test * w_1');
  //from second to third layer
    //v_3_test = act(bias[3] + v_2_test * w_2');
  //from third to output layer
    v_categorical_out_test = act(bias[3] + act(bias[2] + act(bias[1] + X_test * w_in') * w_1') * w_2') * append_col(zero_vector, w_3);
  
  for (i in 1:N_test)
    p[, i] = softmax(v_categorical_out_test[i, ]');
}
