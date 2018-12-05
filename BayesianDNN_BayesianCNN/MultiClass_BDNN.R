################################################################################
# Stan Advent Calendar 2018: Bayesian Deep Neulral Network with R and Stan
# Author: MrUnadon
# Date: 2018-12-11
################################################################################

# clear work space
rm(list = ls())

# library
library(tidyverse)
library(xgboost)
library(data.table)
library(Matrix)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# data(TJOさんのMNIST、お借りしています)
train <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_train.csv')
test <- read.csv('https://github.com/ozt-ca/tjo.hatenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_test.csv')

# balance
table(train$label)
table(test$label)

# MNIST sample Visualize
sample_ind <- c(1, 101, 201, 301, 401, 501, 601, 701, 801, 901)

# formatting
df_sample <- test[sample_ind, ] %>%
  dplyr::mutate(ID = 0:9) %>%
  tidyr::gather(key = "pixel", value = "value", -label, -ID) %>%
  dplyr::arrange(ID) %>%
  dplyr::mutate(x = rep(1:28, length(sample_ind) * 28), y = rep(rep(1:28, each = 28), length(sample_ind))) 

# plot
g_mnist <- ggplot(df_sample, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  theme_void() +
  scale_y_reverse() +
  scale_fill_gradient(high = c("black"), low = "white") +
  facet_wrap(~ID, scales = "free", nrow=2) +
  guides(fill = "none")

# draw
print(g_mnist)

# save
ggsave(g_mnist, filename = "picture/BayesianDNN_MNIST_sample_chr.png", w = 12, h = 4, dpi = 400)





# Index ------------------------------------------------------------------------
# 1. XGboost (Multi Class, merror)
# 2. Bayesian Multi Class Deep Neural Network(three hidden layer)
# 3. Posterior Distribution
# 4. Accuracy and Confusion Matrix
# 5. Posterior Interval and Error
# 6. Correct and Error Sample
# ------------------------------------------------------------------------------





# 1. XGboost (Multi Class, merror) ---------------------------------------------

# sparse matrix
smat_train <- Matrix::sparse.model.matrix(label ~., train)
smat_test <- Matrix::sparse.model.matrix(label ~., test)

# data matrix for xgboost
xgb_train <- xgboost::xgb.DMatrix(smat_train, label = train$label)
xgb_test <- xgboost::xgb.DMatrix(smat_test, label = test$label)

# learning (tuned)
set.seed(71)
xgb_fit <- xgboost::xgb.train(
  data = xgb_train,
  nrounds = 185,
  params = list(label = dt_train$label,
                eval_metric="merror",
                max_depth = 6,
                eta = 0.218,
                subsample = 1,
                colsample_bytree = 0.84,
                alpha = 0, 
                objective = "multi:softmax",
                num_class = 10),
  watchlist=list(train = xgb_train, test = xgb_test)
)


# save model
  #saveRDS(xgb_fit, file = "ml/xgb_fit_multiclass.rds", compress = "xz")
  #xgb_fit <- readRDS(file = "ml/xgb_fit_multiclass.rds")


# prediction with test data
pred_xgb <- predict(xgb_fit, newdata = smat_test, type = "class")

# confusion matrix
(tbl_xgb <- table(pred_xgb, test$label))

# accuracy
(acc_xgb <- sum(diag(tbl_xgb)) / sum(tbl_xgb))


# for plot
df_plot_xgb <- data.frame(
  Label = test$label,
  PredictClassLabel = pred_xgb
) %>%
  dplyr::group_by(Label, PredictClassLabel) %>%
  dplyr::summarise(count = n()) %>%
  dplyr::ungroup()

# confusion matrix plot
g_mat_xgb <- ggplot(df_plot_xgb, aes(x = factor(PredictClassLabel), y = factor(Label), fill = count)) +
  geom_tile() +
  geom_text(aes(label = count)) +
  theme_light() +
  scale_y_discrete(expand = c(0,0)) +
  scale_x_discrete(expand = c(0,0)) +
  scale_fill_gradient(low = "white", high = c(c("#B23AEE"))) +
  labs(title = "Confusion Matrix", subtitle = paste("Accuracy:", acc_xgb),
       x = "Predict", y = "Label")

# draw
print(g_mat_xgb)

# save
ggsave(g_mat_xgb, filename = "picture/xgb_MNIST.png", w = 7, h = 6, dpi = 400)






# 2. Bayesian Multi Class Deep Neural Network(three hidden layer) --------------


# data preparation
normalize <- function(x) {x / 255} # normalize funtion only for MNIST

# data preparation for stan
N_train <- nrow(train) # number of records of train dataset
N_test  <- nrow(test)  # number of records of test dataset
K <- 10 # Number of Category
X_train <- train %>% dplyr::select(-label) %>% mutate_all(funs(normalize)) # train Feature
Y_train <- train$label + 1 # train Label, Stanのsoftmaxは1からなので
X_test <- test %>% dplyr::select(-label) %>% mutate_all(funs(normalize)) # test Features
V <- ncol(X_train) # Number of Feature

# Hyper Parameter Setting
M <- 3 # Number of Middle Layer
NumNode_1st = 256 # Number of nodes in first layer
NumNode_2nd = 128 # Number of nodes in second layer
NumNode_3rd = 64 # Number of nodes in third layer


# datastan
datastan <- list(
  N_train = N_train,
  N_test = N_test,
  V = V,
  M = M,
  K = K,
  X_train = X_train,
  Y_train = Y_train,
  X_test = X_test,
  NumNode_1st = NumNode_1st,
  NumNode_2nd = NumNode_2nd,
  NumNode_3rd = NumNode_3rd
)


# comple and fit
model <- stan_model("stan/model/Multi_BDNN.stan")
fit_BDNN <- sampling(model,
                     data = datastan,
                     iter = 10,
                     seed = 123
                     )

# save model and posterior samples
  # saveRDS(fit_BDNN, file = "stan/fit/fit_BDNN_MultiClass.rds", compress = "xz")
  # fit_BDNN <- readRDS("stan/fit/fit_BDNN_MultiClass.rds")




# 3. Posterior Distribution ----------------------------------------------------

# extract samples(posterior softmax probability)
p <- rstan::extract(fit_BDNN, pars = "p")

# formatting  and select records
p_sample <- p %>%
  data.frame() %>%
  tidyr::gather(key = "Parameter", value = "value") %>%
  tidyr::separate(Parameter, into = c("Parameter", "Class", "ID")) %>%
  dplyr::mutate(ID = as.integer(ID)) %>%
  dplyr::mutate(Class = as.integer(Class) - 1) %>%
  dplyr::filter(ID == 1 |  ID == 101 | ID == 201 | ID == 301 | ID == 401 | ID == 501 |
                  ID == 601 | ID == 701 | ID == 801 | ID == 901)

# plot
g_pred <- ggplot(p_sample, aes(x = factor(Class), y = value, fill = factor(Class), colour = factor(Class))) +
  geom_jitter(alpha = 0.2, width = 0.1) +
  geom_violin(alpha = 0.2, colour = FALSE) +
  facet_wrap(~ID, ncol = 1) +
  theme_light() +
  labs(title = "Posterior Samples in each picture", y = "Probability", x = "Predict Label")

# draw
print(g_pred)

# sabe
ggsave(g_pred, filename = "picture/BayesianDNN_MNIST_pred.png", w = 7, h = 9, dpi = 400)





# 4. Accuracy and Confusion Matrix ---------------------------------------------

# summarizing posterior samples
df_res <- rstan::extract(fit_BDNN, pars = "p") %>%  # extract samples (softmax probability)
  data.frame() %>%
  tidyr::gather(key = "Parameter", value = "value") %>% # gathering
  tidyr::separate(Parameter, into = c("Parameter", "Class", "ID")) %>% # separate index
  dplyr::group_by(Parameter, Class, ID) %>% # processing by groups
  dplyr::summarise(
    EAP = mean(value), # EAP
    Lower = quantile(value, 0.05), # 90% Credible Interval Lower
    Upper = quantile(value, 0.95)  # 90% Credible Interval Upper
  ) %>%
  dplyr::ungroup() %>%
  dplyr::group_by(ID) %>%
  dplyr::mutate(
    PredictClassProb = max(EAP), # The Highest Softmax Probability Class 
    PredictClassLabel = ifelse(PredictClassProb == EAP, as.integer(Class) - 1, NA) # Labeling
  ) %>%
  dplyr::ungroup() %>%
  tidyr::drop_na() %>% # Drop Other Labels
  dplyr::mutate(
    ID = as.integer(ID) # Index of Records
  ) %>%
  dplyr::arrange(ID) %>% # Arrange for Bind Raw Data  Label
  dplyr::mutate(
    Interval = Upper - Lower, # Posterior Interval 
    Label = test$label, # Correct Label
    Error = ifelse(PredictClassLabel == Label, 0, 1), # Error or Not
    # Interval Categories
    IntervalCategory = case_when( 
      Interval < 0.1 ~ "< 0.1",
      Interval >= 0.1 & Interval < 0.2 ~ "< 0.2",
      Interval >= 0.2 & Interval < 0.3 ~ "< 0.3",
      Interval >= 0.3 & Interval < 0.4 ~ "< 0.4",
      Interval >= 0.4 & Interval < 0.5 ~ "< 0.5",
      Interval >= 0.5 & Interval < 0.6 ~ "< 0.6",
      Interval >= 0.6 & Interval < 0.7 ~ "< 0.7",
      Interval >= 0.7 & Interval < 0.8 ~ "< 0.8",
      Interval >= 0.8 & Interval < 0.9 ~ "< 0.9",
      Interval >= 0.8 & Interval < 0.9 ~ "< 0.9",
      Interval >= 0.9 & Interval < 0.95 ~ "< 0.95",
      Interval >= 0.95 & Interval < 0.975 ~ "< 0.975",
      Interval >= 0.975 & Interval < 0.995 ~ "< 0.995",
      Interval >= 0.90 ~ "< 1.0"
    )
  ) 



# Record and Label
pred_class <- df_res %>%
  dplyr::select(ID, PredictClassProb, PredictClassLabel)

# Confusion Matrix
(tbl_BDNN <- table(as.factor(test$label), pred_class$PredictClassLabel))

# accuracy
(acc <- sum(diag(tbl_BDNN)) / sum(tbl_BDNN))

# plot confusion matrix
df_plot <- df_res %>%
  dplyr::group_by(Label, PredictClassLabel) %>%
  dplyr::summarise(count = n()) %>%
  dplyr::ungroup()

# plot
g_mat <- ggplot(df_plot, aes(x = factor(PredictClassLabel), y = factor(Label), fill = count)) +
  geom_tile() +
  geom_text(aes(label = count)) +
  theme_light() +
  scale_y_discrete(expand = c(0,0)) +
  scale_x_discrete(expand = c(0,0)) +
  scale_fill_gradient(low = "white", high = c("#B0E2FF")) +
  labs(title = "Confusion Matrix", subtitle = paste("Accuracy:", acc),
       x = "Predict", y = "Label")

# draw
print(g_mat)

# save
ggsave(g_mat, filename = "picture/BayesianDNN_MNIST.png", w = 7, h = 6, dpi = 400)






# 5. Posterior Interval and Error ----------------------------------------------

# plot error rate in each interval categories
error_rate <- ggplot(df_res, aes(x = IntervalCategory, y = Error, fill = IntervalCategory)) +
  stat_summary(geom = "errorbar", width = 0.2) +
  stat_summary(geom = "bar", colour = "gray25", alpha = 0.7) +
  theme_minimal() +
  labs(title = "Posterior Intervals and Error Rates", y = "Error Rate")

# draw
print(error_rate)

# save
ggsave(error_rate, filename = "picture/BayesianDNN_MNIST_ErrorRate.png", w = 7, h = 6, dpi = 400)





# 6. Correct and Error Sample --------------------------------------------------



# Correct with High Confidence Sample
index2 <- df_res %>%
  dplyr::filter(IntervalCategory == "< 0.1") %>% # filter the narrow interval sample
  dplyr::select(ID)

# formatting
df_corr <- test[index2$ID, ] %>%
  dplyr::mutate(ID = index2$ID) %>%
  tidyr::gather(key = "pixel", value = "value", -label, -ID) %>%
  dplyr::arrange(ID) %>%
  dplyr::mutate(x = rep(1:28, nrow(index2) * 28), y = rep(rep(1:28, each = 28), nrow(index2))) 

# plot
g_corr <- ggplot(df_corr, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  theme_void() +
  scale_y_reverse() +
  scale_fill_gradient(high = c("#3CB371"), low = "white") +
  facet_wrap(~ID, scales = "free") +
  guides(fill = "none")

# draw
print(g_corr)
ggsave(g_corr, filename = "picture/BayesianDNN_MNIST_highconf_chr.png", w = 12, h = 10, dpi = 400)









# Low Confidence Sample
index <- df_res %>%
  dplyr::filter(IntervalCategory == "< 1.0") %>%
  dplyr::select(ID)

df_lowconf <- test[index$ID, ] %>%
  dplyr::mutate(ID = index$ID) %>%
  tidyr::gather(key = "pixel", value = "value", -label, -ID) %>%
  dplyr::arrange(ID) %>%
  dplyr::mutate(x = rep(1:28, nrow(index) * 28), y = rep(rep(1:28, each = 28), nrow(index))) 

# plot
g_lowconf <- ggplot(df_lowconf, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  theme_void() +
  scale_y_reverse() +
  scale_fill_gradient(high = "red", low = "white") +
  facet_wrap(~ID, scales = "free") +
  guides(fill = "none")

# draw
print(g_lowconf)

# save
ggsave(g_lowconf, filename = "picture/BayesianDNN_MNIST_lowconf_chr.png", w = 12, h = 10, dpi = 400)





# インターバル広くてみすってるやつ
index <- df_res %>%
  dplyr::filter(IntervalCategory == "< 1.0"&Error==1) %>%
  dplyr::select(ID)

df_miss <- test[index$ID, ] %>%
  dplyr::mutate(ID = index$ID) %>%
  tidyr::gather(key = "pixel", value = "value", -label, -ID) %>%
  dplyr::arrange(ID) %>%
  dplyr::mutate(x = rep(1:28, nrow(index) * 28), y = rep(rep(1:28, each = 28), nrow(index))) 


g_miss <- ggplot(df_miss, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  theme_void() +
  scale_y_reverse() +
  scale_fill_gradient(high = "darkred", low = "white") +
  facet_wrap(~ID, scales = "free") +
  guides(fill = "none")

g_miss
ggsave(g_miss, filename = "picture/BayesianDNN_MNIST_lowconf_miss_chr.png", w = 12, h = 10, dpi = 400)



