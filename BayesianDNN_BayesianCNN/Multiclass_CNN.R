
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
test <- read.csv('https://github.com/ozt-ca/tjo.h/tenablog.samples/raw/master/r_samples/public_lib/jp/mnist_reproduced/short_prac_test.csv')
train <- train %>% sample_frac(0.3)


# balance
table(train$label)
table(test$label)


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
NumNode_1st = 3 # Number of nodes in first layer
NumNode_2nd = 3 # Number of nodes in second layer
NumNode_3rd = 1 # Number of nodes in third layer


# datastan
datastan <- list(
  stride = 2,
  ck_size1 = 4,
  ck_size2 = 4,
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
model <- stan_model("stan/model/MulticlassCNN.stan")
fit_CNN <- sampling(model,
                     data = datastan,
                     iter = 10,
                     seed = 123,
                     init="0"
)


# saveRDS(fit_CNN, file = "stan/fit/fit_BDNN_MultiClass.rds", compress = "xz")
# fit_CNN <- readRDS("stan/fit/fit_CNN_MultiClass.rds")




# summarizing posterior samples
df_res <- rstan::extract(fit_CNN, pars = "p") %>%  # extract samples (softmax probability)
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


df_res[1:1000,] %>% formattable::formattable()


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
ggsave(g_mat, filename = "picture/BayesianCNN_MNIST.png", w = 7, h = 6, dpi = 400)

