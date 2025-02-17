X <- as.matrix(train_set[, 2:17])
Y <- as.matrix(train_set[, 1])
install.packages('glmnet')
library(glmnet)
lambdas <- seq(from = 0, to = 0.5, length.out = 200)
set.seed(123)
train_cv.lasso <- cv.glmnet(x = X, y = Y, alpha = 1, lambda = lambdas, nfolds = 3, family = "binomial")
lambda_1se <- train_cv.lasso$lambda.1se
lambda_1se
lambda_1se.coef <- coef(train_cv.lasso$glmnet.fit, s = lambda_1se)
lambda_1se.coef

train_set2 <- train_set[, 1:5]
valid_set2 <- valid_set[, 1:5]
test_set2 <- test_set[, 1:5]

train_set2$Subgroup <- factor(train_set2$Subgroup)
train_set2$Sex <- factor(train_set2$Sex)
View(train_set2)
summary(train_set2)
table(train_set2$Subgroup)
dim(train_set2)
valid_set2$Subgroup <- factor(valid_set2$Subgroup)
valid_set2$Sex <- factor(valid_set2$Sex)
View(valid_set2)
summary(valid_set2)
table(valid_set2$Subgroup)
dim(valid_set2)
test_set2$Subgroup <- factor(test_set2$Subgroup)
test_set2$Sex <- factor(test_set2$Sex)
View(test_set2)
summary(test_set2)
table(test_set2$Subgroup)
dim(test_set2)


train_log <- glm(formula = Subgroup ~ ., data = train_set2, family = binomial)
summary(train_log)
log_train_prob <- predict(train_log, newdata = train_set2, type = "response")
log_train_prob
write.csv(log_train_prob, "train set-logistic-probability.csv")
log_valid_response <- predict(train_log, newdata = valid_set2, type = "response")
log_valid_response
write.csv(log_valid_response, "valid set-logistic-probability.csv")
log_test_response <- predict(train_log, newdata = test_set2, type = "response")
log_test_response
write.csv(log_test_response, "test set-logistic-probability.csv")

install.packages("riskRegression") 
library(riskRegression) 
log_train_cal <- Score(object = list(train_log), 
                       formula = Subgroup ~ 1,
                       plots = "calibration", 
                       metrics = "brier", 
                       B = 500, 
                       M = 50, 
                       data = train_set2 
)
log_train_cal[["Brier"]][["score"]] 
log_train_cal_1 <- plotCalibration(log_train_cal)$plotFrames$glm 
write.csv(log_train_cal_1, "training set-logistic calibration curve.csv") 
log_valid_cal <- Score(object = list(train_log), 
                       formula = Subgroup ~ 1, 
                       plots = "calibration", 
                       metrics = "brier", 
                       B = 500, 
                       M = 50, 
                       data = valid_set2 
)
log_valid_cal[["Brier"]][["score"]] 
log_valid_cal_1 <- plotCalibration(log_valid_cal)$plotFrames$glm 
write.csv(log_valid_cal_1, "valid set-logistic calibration curve.csv") 
log_test_cal <- Score(object = list(train_log), 
                      formula = Subgroup ~ 1, 
                      plots = "calibration", 
                      metrics = "brier",
                      B = 500, 
                      M = 50, 
                      data = test_set2 
)
log_test_cal[["Brier"]][["score"]] 
log_test_cal_1 <- plotCalibration(log_test_cal)$plotFrames$glm 
write.csv(log_test_cal_1, "test set-logistic calibration curve.csv") 

source("E:dca.r")
train_set3 <- train_set[, 1:5] 
train_set3$log_train_prob <- log_train_prob 
log_train_dca <- dca(data = train_set3, 
                     outcome = "Subgroup", 
                     predictors = "log_train_prob", 
                     probability = T, 
                     graph = T 
)
write.csv(log_train_dca[["net.benefit"]], "training set-logistic decision curve.csv")
valid_set3 <- valid_set[, 1:5] 
valid_set3$log_valid_response <- log_valid_response 
log_valid_dca <- dca(data = valid_set3, 
                     outcome = "Subgroup",
                     predictors = "log_valid_response", 
                     probability = T, 
                     graph = T 
)
write.csv(log_valid_dca[["net.benefit"]], "valid set-logistic decision curve.csv")
test_set3 <- test_set[, 1:5] 
test_set3$log_test_response <- log_test_response 
log_test_dca <- dca(data = test_set3, 
                    outcome = "Subgroup",
                    predictors = "log_test_response", 
                    probability = T, 
                    graph = T 
)
write.csv(log_test_dca[["net.benefit"]], "test set-logistic decision curve.csv")

install.packages("randomForest")
library(randomForest)
set.seed(123)
train_rf <- randomForest(formula = Subgroup ~ ., data = train_set2 , ntree = 500, mtry = 2, nodesize = 1, replace = TRUE, localImp = TRUE, nPerm = 1000)
train_rf
rf_train_prob <- predict(train_rf, newdata = train_set2[, -1], type = "prob") 
rf_train_prob 
write.csv(rf_train_prob, 'training set-random forest-probability.csv') 
rf_train_class <- predict(train_rf, newdata = train_set2[,-1], type = "class") 
rf_train_class
rf_train_cf <- caret::confusionMatrix(as.factor(rf_train_class), train_set2$Subgroup)
rf_train_cf
rf_valid_prob <- predict(train_rf, newdata = valid_set2, type = "prob") 
rf_valid_prob
write.csv(rf_valid_prob, "valid set-random forest-probability.csv") 
rf_test_prob <- predict(train_rf, newdata = test_set2, type = "prob")
rf_test_prob 
write.csv(rf_test_prob, "test set-random forest-probability.csv") 
install.packages("riskRegression") 

library(riskRegression) 
rf_train_cal <- Score(object = list(train_rf), 
                      formula = Subgroup ~ 1, 
                      plots = "calibration", 
                      metrics = "brier", 
                      B = 500, 
                      M = 50, 
                      data = train_set2 
)
rf_train_cal[["Brier"]][["score"]] 
rf_train_cal_1 <- plotCalibration(rf_train_cal)$plotFrames$randomForest.formula
rf_train_cal_1
write.csv(rf_train_cal_1, "training set-random forest calibration curve.csv") 
rf_valid_cal <- Score(object = list(train_rf), 
                      formula = Subgroup ~ 1, 
                      plots = "calibration",
                      metrics = "brier", 
                      B = 500, 
                      M = 50, 
                      data = valid_set2 
)
rf_valid_cal[["Brier"]][["score"]] 
rf_valid_cal_1 <- plotCalibration(rf_valid_cal)$plotFrames$randomForest.formula 
write.csv(rf_valid_cal_1, "valid set-random forest calibration curve.csv") 
rf_test_cal <- Score(object = list(train_rf), 
                     plots = "calibration", 
                     metrics = "brier", 
                     B = 500, 
                     M = 50, 
                     data = test_set2 
)
rf_test_cal[["Brier"]][["score"]] 
rf_test_cal_1 <- plotCalibration(rf_test_cal)$plotFrames$randomForest.formula 
write.csv(rf_test_cal_1, "test set-random forest calibration curve.csv") 

source("E:dca.r")
rf_train_prob[,2]
train_set3 <- train_set[, 1:5] 
train_set3$rf_train_prob <- rf_train_prob[,2] 
train_set3
rf_train_dca <- dca(data = train_set3, 
                    outcome = "Subgroup", 
                    predictors = "rf_train_prob", 
                    probability = T, 
                    graph = T 
)
write.csv(rf_train_dca[["net.benefit"]], "training set-random forest decision curve.csv")
rf_valid_prob[,2]
valid_set3 <- valid_set[, 1:5] 
valid_set3$rf_valid_prob <- rf_valid_prob[,2] 
rf_valid_dca <- dca(data = valid_set3, 
                    outcome = "Subgroup", 
                    predictors = "rf_valid_prob", 
                    probability = T, 
                    graph = T 
)
write.csv(rf_valid_dca[["net.benefit"]], "valid set-random forest decision curve.csv")
rf_test_prob[,2]
test_set3 <- test_set[, 1:5] 
test_set3$rf_test_prob <- rf_test_prob[,2] 
test_set3
rf_test_dca <- dca(data = test_set3, 
                   outcome = "Subgroup", 
                   predictors = "rf_test_prob", 
                   probability = T, 
                   graph = T 
)
write.csv(rf_test_dca[["net.benefit"]], "test set-random forest decision curve.csv")

install.packages("e1071") 
library(e1071) 
train_svm <- svm(Subgroup ~ ., data = train_set2, probability = TRUE) 
summary(train_svm) 
svm_train_result <- predict(train_svm, newdata = train_set2[,-1], probability = TRUE) 
svm_train_result 
svm_train_prob <- attr(svm_train_result,"probabilities") 
svm_train_prob 
write.csv(svm_train_prob,"training set-SVM-probability.csv") 
svm_valid_result <- predict(train_svm, newdata = valid_set2, probability = TRUE)
svm_valid_result 
svm_valid_prob <- attr(svm_valid_result,"probabilities") 
svm_valid_prob 
write.csv(svm_valid_prob, "valid set-SVM-probability.csv") 
svm_test_result <- predict(train_svm, newdata = test_set2, probability = TRUE)
svm_test_result 
svm_test_prob <- attr(svm_test_result,"probabilities") 
svm_test_prob 
write.csv(svm_test_prob, "test set-SVM-probability.csv") 
svm_train_prob[,2]
train_set3 <- train_set[, 1:5] 
train_set3$svm_train_prob <- svm_train_prob[,2] 
train_set3
svm_train_dca <- dca(data = train_set3, 
                     outcome = "Subgroup", 
                     predictors = "svm_train_prob", 
                     probability = T, 
                     graph = T 
)
write.csv(svm_train_dca[["net.benefit"]], "training set-SVM decision curve.csv")
svm_valid_prob[,2]
valid_set3 <- valid_set[, 1:5] 
valid_set3$svm_valid_prob <- svm_valid_prob[,2] 
valid_set3
svm_valid_dca <- dca(data = valid_set3, 
                     outcome = "Subgroup", 
                     predictors = "svm_valid_prob", 
                     probability = T, 
                     graph = T 
)
write.csv(svm_valid_dca[["net.benefit"]], "valid set-SVM decision curve.csv")
svm_test_prob[,2]
test_set3 <- test_set[, 1:5] 
test_set3$svm_test_prob <- svm_test_prob[,2] 
test_set3
svm_test_dca <- dca(data = test_set3, 
                    outcome = "Subgroup",
                    predictors = "svm_test_prob", 
                    probability = T, 
                    graph = T 
)
write.csv(svm_test_dca[["net.benefit"]], "test set-SVM decision curve.csv")

clinic_set <- read_excel("Clinical Samples.xlsx") 
clinic_set <- as.data.frame(clinic_set) 
row.names(clinic_set) <- clinic_set[, 1]
clinic_set <- clinic_set[, -1]
clinic_set2 <-  clinic_set[,1:5] 
clinic_set2$Subgroup <- factor(clinic_set2$Subgroup, levels = c(0,1), labels = c("Healthy People", "Liver Cancer Patients"))
clinic_set2$Sex <- factor(clinic_set2$Sex, levels = c(0,1), labels = c("Female", "Male"))
View(clinic_set2)
summary(clinic_set2)
table(clinic_set2$Subgroup)
dim(clinic_set2)

log_clinic_prob <- predict(train_log, newdata = clinic_set2, type = "response") 
log_clinic_prob 
write.csv(log_clinic_prob, "clinical samples-logistic-probability.csv") 

rf_clinic_prob <- predict(train_rf, newdata = clinic_set2, type = "prob") 
rf_clinic_prob 
write.csv(rf_clinic_prob, "clinical samples-random forest-probability.csv") 

svm_clinic_result <- predict(train_svm, newdata = clinic_set2, probability = TRUE) 
svm_clinic_result 
svm_clinic_prob <- attr(svm_clinic_result,"probabilities") 
svm_clinic_prob 
write.csv(svm_clinic_prob, "clinical samples-SVM-probability.csv") 

remotes::install_github("tidymodels/probably")
suppressMessages(library(tidymodels))
suppressMessages(library(probably))
train_set3
valid_set3
test_set3

rms::val.prob(
  p = test_set3$log_test_response, 
  y = test_set3$Subgroup, 
  cex = 1, 
  logistic.cal = F
)

rms::val.prob(
  p = test_set3$rf_test_prob,
  y = test_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)

rms::val.prob(
  p = test_set3$svm_test_prob,
  y = test_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)

rms::val.prob(
  p = valid_set3$log_valid_response,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)

rms::val.prob(
  p = valid_set3$rf_valid_prob,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)

rms::val.prob(
  p = valid_set3$svm_valid_prob,
  y = valid_set3$Subgroup,
  cex = 1,
  logistic.cal = F
)
test_log_cal <- test_set3 %>% 
  mutate(pred_rnd = round(log_test_response, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(log_test_response),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
write.csv(test_log_cal,"test set-logistic-calibration curve-mlr3.csv")

head (test_set3)
test_rf_cal <- test_set3 %>% 
  mutate(pred_rnd = round(rf_test_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(rf_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_rf_cal
write.csv(test_rf_cal,"test set-random forest-calibration curve-mlr3.csv")

head (test_set3)
test_svm_cal <- test_set3 %>% 
  mutate(pred_rnd = round(svm_test_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(svm_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_svm_cal
write.csv(test_svm_cal,"test set-SVM-calibration curve-mlr3.csv")


head (valid_set3)

valid_log_cal <- valid_set3 %>% 
  mutate(pred_rnd = round(log_valid_response, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(log_valid_response),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_log_cal
write.csv(valid_log_cal,"valid set-logistic-calibration curve-mlr3.csv")

valid_rf_cal <- valid_set3 %>% 
  mutate(pred_rnd = round(rf_valid_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(rf_valid_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_rf_cal
write.csv(valid_rf_cal,"valid set-random forest-calibration curve-mlr3.csv")

head (valid_set3)
valid_svm_cal <- valid_set3 %>% 
  mutate(pred_rnd = round(svm_valid_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(svm_valid_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_svm_cal
write.csv(valid_svm_cal,"valid set-SVM-calibration curve-mlr3.csv")


