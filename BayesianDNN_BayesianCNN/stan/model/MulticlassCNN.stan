//MulticlassCNN.stan
functions {
  // convolution
  vector conv(vector X, int num_col, int window, int stride, matrix kw, real kb) {
    vector[(((num_col - window) / stride) + 1) * (((num_col - window) / stride) + 1) ] conv;
    matrix[window, window] x [(((num_col - window) / stride) + 1) * (((num_col - window) / stride) + 1)];
    // tmp matrix
    for(l in 1:(((num_col - window) / stride) + 1) ){ 
      for(k in 1:(((num_col - window) / stride) + 1) ){ 
        for(i in 1:window) {
          for(j in 1:window) {
            x[(l + ((l - 1) * (num_col - window) / stride) + (k - 1)), i, j] = X[(i * num_col - num_col) + (stride * (k - 1)) + num_col * (stride * (l - 1)) + j];
            }
          }
        }
      }
    //convolution(stride = 1)
    for(m in 1 :(((num_col - window) / stride) + 1) * (((num_col - window) / stride) + 1)) {
      conv[m] = inv_logit(sum(kb + x[m,,] .* kw)) * 2 - 1;
    }
    return(conv);
  }
  
  // max pooling
  vector maxpool(vector X, int num_col, int window, int stride) {
    vector[(((num_col - window) / stride) + 1) * (((num_col - window) / stride) + 1)] pool;
    matrix[window, window] x [(((num_col - window) / stride) + 1) * (((num_col - window) / stride) + 1)];
    // tmp matrix
    for(l in 1:(((num_col - window) / stride) + 1) ){ 
      for(k in 1:(((num_col - window) / stride) + 1) ){ 
        for(i in 1:window) {
          for(j in 1:window) {
            x[(l + ((l - 1) * (num_col - window) / stride) + (k - 1)), i, j] = X[(i * num_col - num_col) + (stride * (k - 1)) + num_col * (stride * (l - 1)) + j];
            }
          }
        }
      }
    //max pooling(stride = 1)
    for(m in 1 : (((num_col - window) / stride) + 1) * (((num_col - window) / stride) + 1)) {
      pool[m] = max(x[m,,]);
    }
    return(pool);
  }
  
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
  int stride; //convolutional stride = 2
  int ck_size1; // convolution kernel size1 = 4
  int ck_size2; // convolution kernel size2 = 4
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
  //first convolution
  real kb1_1;
  matrix[ck_size1, ck_size1] kw1_1;
  real kb1_2;
  matrix[ck_size1, ck_size1] kw1_2;
  real kb1_3;
  matrix[ck_size1, ck_size1] kw1_3;
  real kb1_4;
  matrix[ck_size1, ck_size1] kw1_4;
  real kb1_5;
  matrix[ck_size1, ck_size1] kw1_5;


  //second convolution
  real kb2_1;
  matrix[ck_size2, ck_size2] kw2_1;
  real kb2_2;
  matrix[ck_size2, ck_size2] kw2_2;
  real kb2_3;
  matrix[ck_size2, ck_size2] kw2_3;
  real kb2_4;
  matrix[ck_size2, ck_size2] kw2_4;
  real kb2_5;
  matrix[ck_size2, ck_size2] kw2_5;

  
  vector[M] bias;
  matrix[NumNode_1st, 16 * 5]  w_in; // weight input 
  matrix[NumNode_2nd, NumNode_1st] w_1;  // weight 1st to 2nd
  matrix[NumNode_3rd, NumNode_2nd] w_2;  // weight 2nd to 3rd
  matrix[NumNode_3rd, K - 1] w_3;  // weight 3rd to output
}



transformed parameters {
  matrix[N_train, 16 * 5] conv_pooled_vector;
  matrix[N_train, K] v_categorical_out; // value output layer
  //convolution and maxpooling
  for(i in 1 : N_train) {
    conv_pooled_vector[i, ] = to_row_vector(
      append_row(
        append_row(
          append_row(
            append_row(

      maxpool(conv(maxpool(conv(to_vector(X_train[i,]), 28, ck_size1, stride, kw1_1, kb1_1), 13, 2, 1), 12, ck_size2, stride, kw2_1, kb2_1), 5, 2, 1),
      maxpool(conv(maxpool(conv(to_vector(X_train[i,]), 28, ck_size1, stride, kw1_2, kb1_2), 13, 2, 1), 12, ck_size2, stride, kw2_2, kb2_2), 5, 2, 1)),
      maxpool(conv(maxpool(conv(to_vector(X_train[i,]), 28, ck_size1, stride, kw1_3, kb1_3), 13, 2, 1), 12, ck_size2, stride, kw2_3, kb2_3), 5, 2, 1)),
      maxpool(conv(maxpool(conv(to_vector(X_train[i,]), 28, ck_size1, stride, kw1_4, kb1_4), 13, 2, 1), 12, ck_size2, stride, kw2_4, kb2_4), 5, 2, 1)),
      maxpool(conv(maxpool(conv(to_vector(X_train[i,]), 28, ck_size1, stride, kw1_5, kb1_5), 13, 2, 1), 12, ck_size2, stride, kw2_5, kb2_5), 5, 2, 1)));
  }

  //full connected
  v_categorical_out = act(bias[3] + act(bias[2] + act(bias[1] + conv_pooled_vector * w_in') * w_1') * w_2') * append_col(zero_vector, w_3);
}



model {
  //first convolution
  kb1_1 ~ normal(0,1);
  to_vector(kw1_1) ~ normal(0,1);
  kb1_2 ~ normal(0,1);
  to_vector(kw1_2) ~ normal(0,1);
  kb1_3 ~ normal(0,1);
  to_vector(kw1_3) ~ normal(0,1);


  //second convolution
  kb2_1 ~ normal(0,1);
  to_vector(kw2_1) ~ normal(0,1);
  kb2_2 ~ normal(0,1);
  to_vector(kw2_2) ~ normal(0,1);
  kb2_3 ~ normal(0,1);
  to_vector(kw2_3) ~ normal(0,1);

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
  matrix[N_test, 16 * 5] conv_pooled_vector_test;
  matrix[N_test, K] v_categorical_out_test; // value output layer
  matrix[K, N_test] p;
  for(i in 1:N_test) {
        conv_pooled_vector_test[i, ] = to_row_vector(
          append_row(
            append_row(
              append_row(
                append_row(

      maxpool(conv(maxpool(conv(to_vector(X_test[i,]), 28, ck_size1, stride, kw1_1, kb1_1), 13, 2, 1), 12, ck_size2, stride, kw2_1, kb2_1), 5, 2, 1),
      
      maxpool(conv(maxpool(conv(to_vector(X_test[i,]), 28, ck_size1, stride, kw1_2, kb1_2), 13, 2, 1), 12, ck_size2, stride, kw2_2, kb2_2), 5, 2, 1)),
      
      maxpool(conv(maxpool(conv(to_vector(X_test[i,]), 28, ck_size1, stride, kw1_3, kb1_3), 13, 2, 1), 12, ck_size2, stride, kw2_3, kb2_3), 5, 2, 1)),
      
      maxpool(conv(maxpool(conv(to_vector(X_test[i,]), 28, ck_size1, stride, kw1_4, kb1_4), 13, 2, 1), 12, ck_size2, stride, kw2_4, kb2_4), 5, 2, 1)),
      
      maxpool(conv(maxpool(conv(to_vector(X_test[i,]), 28, ck_size1, stride, kw1_5, kb1_5), 13, 2, 1), 12, ck_size2, stride, kw2_5, kb2_5), 5, 2, 1)));

  }
  
  
    v_categorical_out_test = act(bias[3] + act(bias[2] + act(bias[1] + conv_pooled_vector_test * w_in') * w_1') * w_2') * append_col(zero_vector, w_3);
  
  for (i in 1:N_test)
    p[, i] = softmax(v_categorical_out_test[i, ]');
}
