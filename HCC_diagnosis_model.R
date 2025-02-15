#交叉lasso回归进行自变量筛选
X <- as.matrix(train_set[ , 2:17])
Y <- as.matrix (train_set [ , 1])
install.packages('glmnet')
library (glmnet)
lambdas <- seq(from=0, to=0.5, length.out=200)
set.seed (123)
train_cv.lasso <- cv.glmnet (x=X, y=Y, alpha =1, lambda=lambdas, nfolds=3,  family="binomial")
lambda_1se <- train_cv.lasso$lambda.1se
lambda_1se
lambda_1se.coef <- coef (train_cv.lasso$glmnet.fit, s=lambda_1se)
lambda_1se.coef

#导入数据集
train_set2 <- train_set[, 1:5] #训练集
valid_set2 <- valid_set[, 1:5] #验证集
test_set2 <- test_set [, 1:5] #测试集

#将训练集中的二分类变量改成因子形式
train_set2$Subgroup <- factor (train_set2$Subgroup)
train_set2$Sex <- factor (train_set2$Sex)
#查看更改后的训练集
View (train_set2)
summary(train_set2)
table (train_set2$Subgroup)
dim (train_set2)
#将验证集中的二分类变量改成因子形式
valid_set2$Subgroup <- factor (valid_set2$Subgroup)
valid_set2$Sex <- factor (valid_set2$Sex)
#查看更改后的验证集
View (valid_set2)
summary(valid_set2)
table (valid_set2$Subgroup)
dim (valid_set2)
#将测试集中的二分类变量改成因子形式
test_set2$Subgroup <- factor (test_set2$Subgroup)
test_set2$Sex <- factor (test_set2$Sex)
#查看更改后的验证集
View (test_set2)
summary (test_set2)
table (test_set2$Subgroup)
dim (test_set2)

#建立逻辑回归模型
train_log <- glm (formula = Subgroup~ . , data = train_set2, family = binomial)
summary(train_log)

#预测
#训练集预测
log_train_prob <- predict(train_log, newdata=train_set2, type = "response")
log_train_prob
write.csv (log_train_prob, "训练集-logistic-概率.csv")
#验证集预测
log_valid_response <- predict (train_log, newdata = valid_set2, type = "response")
log_valid_response
write.csv(log_valid_response, "验证集-logistic-概率.csv")
#测试集预测
log_test_response <- predict (train_log, newdata = test_set2, type = "response")
log_test_response
write.csv (log_test_response, "测试集-logistic-概率.csv")

#logistic模型评价
#校准曲线
install.packages("riskRegression") #安装riskRegression包
library (riskRegression) #加载riskRegression包
#训练集
log_train_cal <- Score(object = list(train_log), #指定模型，并通过list()函数将模型包装起来
                       formula = Subgroup ~ 1, #设置因变量是什么，波浪线（~）之后为1，表示不设置自变量，因为这里只需要设置因变量
                       plots = "calibration", #绘制那些图形，calibration表示绘制校准曲线
                       metrics = "brier", #得到那些统计量，brier评分是用来评价calibration曲线表现的指标，brier数值越小，模型的准确性越高。
                       B=500, #设置bootstrap次数
                       M=50, #设置每次bootstrap的样本大小
                       data= train_set2 #指定数据集
)
log_train_cal[["Brier"]][["score"]] #查看brier评分
log_train_cal_1 <- plotCalibration(log_train_cal)$plotFrames$glm #将预测概率和实际概率提取出来，也就是x轴数据和y数据
write.csv(log_train_cal_1, "训练集-logistic-校准曲线.csv") #导出结果
#验证集
log_valid_cal <- Score(object = list(train_log), #指定模型，并通过list()函数将模型包装起来
                       formula = Subgroup ~ 1, #设置因变量是什么，波浪线（~）之后为1，表示不设置自变量，因为这里只需要设置因变量
                       plots = "calibration", #绘制那些图形，calibration表示绘制校准曲线
                       metrics = "brier", #得到那些统计量，brier评分是用来评价calibration曲线表现的指标，brier数值越小，模型的准确性越高。
                       B=500, #设置bootstrap次数
                       M=50, #设置每次bootstrap的样本大小
                       data= valid_set2 #指定数据集
)
log_valid_cal[["Brier"]][["score"]] #查看brier评分
log_valid_cal_1 <- plotCalibration(log_valid_cal)$plotFrames$glm #将预测概率和实际概率提取出来，也就是x轴数据和y数据
write.csv(log_valid_cal_1, "验证集-logistic-校准曲线.csv") #导出结果
#测试集
log_test_cal <- Score(object = list(train_log), #指定模型，并通过list()函数将模型包装起来
                      formula = Subgroup ~ 1, #设置因变量是什么，波浪线（~）之后为1，表示不设置自变量，因为这里只需要设置因变量
                      plots = "calibration", #绘制那些图形，calibration表示绘制校准曲线
                      metrics = "brier", #得到那些统计量，brier评分是用来评价calibration曲线表现的指标，brier数值越小，模型的准确性越高。
                      B=500, #设置bootstrap次数
                      M=50, #设置每次bootstrap的样本大小
                      data= test_set2 #指定数据集
)
log_test_cal[["Brier"]][["score"]] #查看brier评分
log_test_cal_1 <- plotCalibration(log_test_cal)$plotFrames$glm #将预测概率和实际概率提取出来，也就是x轴数据和y数据
write.csv(log_test_cal_1, "测试集-logistic-校准曲线.csv") #导出结果

#DCA曲线
source("E:dca.r")
#训练集
train_set3 <- train_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为train_set3
train_set3$log_train_prob <- log_train_prob #将logistic回归模型预测的概率添加到训练集train_set3中
log_train_dca <- dca(data = train_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="log_train_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (log_train_dca[["net.benefit"]],"训练集-logistic-决策曲线.csv")
#验证集
valid_set3 <- valid_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为valid_set3
valid_set3$log_valid_response <- log_valid_response #将logistic回归模型预测的概率添加到训练集valid_set3中
log_valid_dca <- dca(data = valid_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="log_valid_response", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (log_valid_dca[["net.benefit"]],"验证集-logistic-决策曲线.csv")
#测试集
test_set3 <- test_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为test_set3
test_set3$log_test_response <- log_test_response #将logistic回归模型预测的概率添加到训练集test_set3中
test_set3
log_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="log_test_response", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (log_test_dca[["net.benefit"]],"测试集-logistic-决策曲线.csv")


#建立随机森林模型
install.packages("randomForest")
library(randomForest)
set.seed(123)
train_rf <- randomForest (formula = Subgroup ~ . , data = train_set2 , ntree = 500, mtry=2, nodesize=1, replace=TRUE, localImp=TRUE, nPerm=1000)
train_rf

#预测
#训练集预测
rf_train_prob <- predict (train_rf, newdata=train_set2[, -1], type = "prob") #预测概率
rf_train_prob #查看预测结果
write.csv(rf_train_prob, '训练集-随机森林-概率.csv') #导出结果

rf_train_class <- predict(train_rf, newdata=train_set2[,-1], type = "class") #预测类型
rf_train_class
rf_train_cf <- caret::confusionMatrix (as.factor(rf_train_class),train_set2$Subgroup)
rf_train_cf
#验证集预测
rf_valid_prob <- predict (train_rf, newdata = valid_set2, type = "prob") #预测概率
rf_valid_prob #查看预测结果
write.csv (rf_valid_prob, "验证集-随机森林-概率.csv") #导出结果

#测试集预测
rf_test_prob <- predict (train_rf, newdata = test_set2, type = "prob") #预测概率
rf_test_prob #查看预测结果
write.csv (rf_test_prob, "测试集-随机森林-概率.csv") #导出结果

#随机森林模型评价
#校准曲线
install.packages("riskRegression") #安装riskRegression包
library (riskRegression) #加载riskRegression包
#训练集
rf_train_cal <- Score(object = list(train_rf), #指定模型，并通过list()函数将模型包装起来
                      formula = Subgroup ~ 1, #设置因变量是什么，波浪线（~）之后为1，表示不设置自变量，因为这里只需要设置因变量
                      plots = "calibration", #绘制那些图形，calibration表示绘制校准曲线
                      metrics = "brier", #得到那些统计量，brier评分是用来评价calibration曲线表现的指标，brier数值越小，模型的准确性越高。
                      B=500, #设置bootstrap次数
                      M=50, #设置每次bootstrap的样本大小
                      data= train_set2 #指定数据集
)
rf_train_cal[["Brier"]][["score"]] #查看brier评分
rf_train_cal_1 <- plotCalibration(rf_train_cal)$plotFrames$randomForest.formula #将预测概率和实际概率提取出来，也就是x轴数据和y数据
rf_train_cal_1
write.csv(rf_train_cal_1, "训练集-随机森林-校准曲线.csv") #导出结果
#验证集
rf_valid_cal <- Score(object = list(train_rf), #指定模型，并通过list()函数将模型包装起来
                      formula = Subgroup ~ 1, #设置因变量是什么，波浪线（~）之后为1，表示不设置自变量，因为这里只需要设置因变量
                      plots = "calibration", #绘制那些图形，calibration表示绘制校准曲线
                      metrics = "brier", #得到那些统计量，brier评分是用来评价calibration曲线表现的指标，brier数值越小，模型的准确性越高。
                      B=500, #设置bootstrap次数
                      M=50, #设置每次bootstrap的样本大小
                      data= valid_set2 #指定数据集
)
rf_valid_cal[["Brier"]][["score"]] #查看brier评分
rf_valid_cal_1 <- plotCalibration(rf_valid_cal)$plotFrames$randomForest.formula #将预测概率和实际概率提取出来，也就是x轴数据和y数据
write.csv(rf_valid_cal_1, "验证集-随机森林-校准曲线.csv") #导出结果
#测试集
rf_test_cal <- Score(object = list(train_rf), #指定模型，并通过list()函数将模型包装起来
                     formula = Subgroup ~ 1, #设置因变量是什么，波浪线（~）之后为1，表示不设置自变量，因为这里只需要设置因变量
                     plots = "calibration", #绘制那些图形，calibration表示绘制校准曲线
                     metrics = "brier", #得到那些统计量，brier评分是用来评价calibration曲线表现的指标，brier数值越小，模型的准确性越高。
                     B=500, #设置bootstrap次数
                     M=50, #设置每次bootstrap的样本大小
                     data= test_set2 #指定数据集
)
rf_test_cal[["Brier"]][["score"]] #查看brier评分
rf_test_cal_1 <- plotCalibration(rf_test_cal)$plotFrames$randomForest.formula #将预测概率和实际概率提取出来，也就是x轴数据和y数据
write.csv(rf_test_cal_1, "测试集-随机森林-校准曲线.csv") #导出结果

#DCA曲线
source("E:dca.r")
#训练集
rf_train_prob[,2]
train_set3 <- train_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为train_set3
train_set3$rf_train_prob <- rf_train_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
train_set3
rf_train_dca <- dca(data = train_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="rf_train_prob", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (rf_train_dca[["net.benefit"]],"训练集-随机森林-决策曲线.csv")
#验证集
rf_valid_prob[,2]
valid_set3 <- valid_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为valid_set3
valid_set3$rf_valid_prob <- rf_valid_prob[,2] #将随机森林回归模型预测的概率添加到训练集valid_set3中
rf_valid_dca <- dca(data = valid_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="rf_valid_prob", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (rf_valid_dca[["net.benefit"]],"验证集-随机森林-决策曲线.csv")
#测试集
rf_test_prob[,2]
test_set3 <- test_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为test_set3
test_set3$rf_test_prob <- rf_test_prob[,2] #将随机森林模型预测的概率添加到训练集test_set3中
test_set3
rf_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                   outcome="Subgroup", # 指定结果变量
                   predictors="rf_test_prob", # 指定预测变量
                   probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                   graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (rf_test_dca[["net.benefit"]],"测试集-随机森林-决策曲线.csv")


#建立SVM模型
install.packages("e1071") #安装包
library (e1071) #加载包
train_svm <- svm (Subgroup ~ . , data = train_set2, probability = TRUE) #构建模型
summary (train_svm)  #查看模型

#预测
#训练集预测
svm_train_result <- predict (train_svm, newdata = train_set2[,-1], probability = TRUE) #进行预测
svm_train_result #查看预测结果,结果会将预测的类型和概率全显示出来
svm_train_prob <- attr(svm_train_result,"probabilities") #将预测概率提取出来
svm_train_prob #查看概率预测结果
write.csv(svm_train_prob,"训练集-SVM-概率.csv") #导出结果
#验证集预测
svm_valid_result <- predict (train_svm, newdata = valid_set2, probability = TRUE)
svm_valid_result #查看预测结果,结果会将预测的类型和概率全显示出来
svm_valid_prob <- attr (svm_valid_result,"probabilities") #将预测概率提取出来
svm_valid_prob #查看概率预测结果
write.csv(svm_valid_prob, "验证集-SVM-概率.csv") #导出结果
#测试集预测
svm_test_result <- predict (train_svm, newdata = test_set2, probability = TRUE)
svm_test_result #查看预测结果,结果会将预测的类型和概率全显示出来
svm_test_prob <- attr (svm_test_result,"probabilities") #将预测概率提取出来
svm_test_prob #查看概率预测结果
write.csv(svm_test_prob, "测试集-SVM-概率.csv") #导出结果

#DCA曲线
#训练集
svm_train_prob[,2]
train_set3 <- train_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为train_set3
train_set3$svm_train_prob <- svm_train_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
train_set3
svm_train_dca <- dca(data = train_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="svm_train_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (svm_train_dca[["net.benefit"]],"训练集-SVM-决策曲线.csv")
#验证集
svm_valid_prob[,2]
valid_set3 <- valid_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取验证集，命名为valid_set3
valid_set3$svm_valid_prob <- svm_valid_prob[,2] #将随机森林模型预测的概率添加到训练集valid_set3中
valid_set3
svm_valid_dca <- dca(data = valid_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="svm_valid_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (svm_valid_dca[["net.benefit"]],"验证集-SVM-决策曲线.csv")
#测试集
svm_test_prob[,2]
test_set3 <- test_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取测试集，命名为test_set3
test_set3$svm_test_prob <- svm_test_prob[,2] #将随机森林模型预测的概率添加到训练集test_set3中
test_set3
svm_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="svm_test_prob", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (svm_test_dca[["net.benefit"]],"测试集-SVM-决策曲线.csv")

#KNN模型
install.packages("caret") #安装包
library (caret) #加载包
set.seed(123) #设置随机数
train_knn <- knn3 (formula = Subgroup ~ . , data = train_set2, k=10) #构建模型
summary (train_knn) #查看模型

#预测
#训练集预测
knn_train_prob <- predict (train_knn, newdata = train_set2, type = "prob") #预测概率
knn_train_prob #查看预测结果
write.csv(knn_train_prob, "训练集-KNN-概率.csv") #导出结果
knn_valid_prob <- predict (train_knn, newdata = valid_set2, type = "prob") #预测概率
knn_valid_prob #查看预测结果
write.csv (knn_valid_prob, "验证集-KNN-概率.csv") #导出结果
knn_test_prob <- predict (train_knn, newdata = test_set2, type = "prob") #预测概率
knn_test_prob #查看预测结果
write.csv (knn_test_prob, "测试集-KNN-概率.csv") #导出结果

#DCA曲线
#训练集
knn_train_prob[,2]
train_set3 <- train_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为train_set3
train_set3$knn_train_prob <- knn_train_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
train_set3
knn_train_dca <- dca(data = train_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="knn_train_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (knn_train_dca[["net.benefit"]],"训练集-KNN-决策曲线.csv")
#验证集
knn_valid_prob[,2]
valid_set3 <- valid_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为train_set3
valid_set3$knn_valid_prob <- knn_valid_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
valid_set3
knn_valid_dca <- dca(data = valid_set3, # 指定数据集,必须是data.frame类型
                     outcome="Subgroup", # 指定结果变量
                     predictors="knn_valid_prob", # 指定预测变量
                     probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                     graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (knn_valid_dca[["net.benefit"]],"验证集-KNN-决策曲线.csv")
#测试集
knn_test_prob[,2]
test_set3 <- test_set[, 1:5] #因为前面已经将二分类变量改成因子形式，但是绘制DCA曲线不能是因子形式，所以重新提取训练集，命名为train_set3
test_set3$knn_test_prob <- knn_test_prob[,2] #将随机森林模型预测的概率添加到训练集train_set3中
test_set3
knn_test_dca <- dca(data = test_set3, # 指定数据集,必须是data.frame类型
                    outcome="Subgroup", # 指定结果变量
                    predictors="knn_test_prob", # 指定预测变量
                    probability = T, #表示predictors="log_prob"是否为概率，若不是概率，就写F
                    graph = T #是否输出图片，T为是，F为否，这里可以输出看一下
)
write.csv (knn_test_dca[["net.benefit"]],"测试集-KNN-决策曲线.csv")


#导入临床样本数据
clinic_set <- read_excel("临床样本.xlsx") #读取数据
#将第一列变成每一行的名字
clinic_set <- as.data.frame (clinic_set) 
row.names(clinic_set) <- clinic_set [ , 1]
clinic_set <- clinic_set [, -1]
#取出第1~5列，第一列为因变量，第2到5列为自变量
clinic_set2 <-  clinic_set [,1:5] 
#将临床样本中的二分类变量改成因子形式
clinic_set2$Subgroup <- factor (clinic_set2$Subgroup, levels = c(0,1), labels = c("健康人", "肝癌患者"))
clinic_set2$Sex <- factor (clinic_set2$Sex, levels = c(0,1), labels = c("女", "男"))
#查看更改后的训练集
View (clinic_set2)
summary (clinic_set2)
table (clinic_set2$Subgroup)
dim (clinic_set2)

#预测
#逻辑回归预测
log_clinic_prob <- predict(train_log, newdata=clinic_set2, type = "response") #预测概率
log_clinic_prob #查看预测结果
write.csv (log_clinic_prob, "临床样本-logistic-概率.csv") #导出结果
#随机森林预测
rf_clinic_prob <- predict (train_rf, newdata = clinic_set2, type = "prob") #预测概率
rf_clinic_prob #查看预测结果
write.csv (rf_clinic_prob, "临床样本-随机森林-概率.csv") #导出结果
#SVM模型预测
svm_clinic_result <- predict (train_svm, newdata = clinic_set2, probability = TRUE) #进行预测
svm_clinic_result #查看预测结果,结果会将预测的类型和概率全显示出来
svm_clinic_prob <- attr (svm_clinic_result,"probabilities") #将预测概率提取出来
svm_clinic_prob #查看概率预测结果
write.csv(svm_clinic_prob, "临床样本-SVM-概率.csv") #导出结果
#KNN模型预测
knn_clinic_prob <- predict (train_knn, newdata = clinic_set2, type = "prob") #预测概率
knn_clinic_prob #查看概率预测结果
write.csv(knn_clinic_prob, "临床样本-KNN-概率-1.csv") #导出结果


#校准曲线

remotes::install_github("tidymodels/probably")
suppressMessages(library(tidymodels))
suppressMessages(library(probably))
train_set3
valid_set3
test_set3

rms::val.prob(
  p = test_set3$log_test_response, #预测概率
  y = test_set3$Subgroup, #实际类型
  cex = 1, #输出图的字体大小
  logistic.cal = F #是否输出对结果进行logistic回归后的图，这里选F就可以了
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
  p = test_set3$knn_test_prob,
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

rms::val.prob(
  p = valid_set3$knn_valid_prob,
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
write.csv(test_log_cal,"测试集-logistic-校准曲线-mlr3包.csv")

head (test_set3)
test_rf_cal <- test_set3 %>% 
  mutate(pred_rnd = round(rf_test_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(rf_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_rf_cal
write.csv(test_rf_cal,"测试集-随机森林-校准曲线-mlr3包.csv")

head (test_set3)
test_svm_cal <- test_set3 %>% 
  mutate(pred_rnd = round(svm_test_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(svm_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_svm_cal
write.csv(test_svm_cal,"测试集-SVM-校准曲线-mlr3包.csv")

head (test_set3)
test_knn_cal <- test_set3 %>% 
  mutate(pred_rnd = round(knn_test_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(knn_test_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
test_knn_cal
write.csv(test_knn_cal,"测试集-KNN-校准曲线-mlr3包.csv")

head (valid_set3)

valid_log_cal <- valid_set3 %>% 
  mutate(pred_rnd = round(log_valid_response, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(log_valid_response),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_log_cal
write.csv(valid_log_cal,"验证集-logistic-校准曲线-mlr3包.csv")

valid_rf_cal <- valid_set3 %>% 
  mutate(pred_rnd = round(rf_valid_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(rf_valid_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_rf_cal
write.csv(valid_rf_cal,"验证集-随机森林-校准曲线-mlr3包.csv")

head (valid_set3)
valid_svm_cal <- valid_set3 %>% 
  mutate(pred_rnd = round(svm_valid_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(svm_valid_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_svm_cal
write.csv(valid_svm_cal,"验证集-SVM-校准曲线-mlr3包.csv")

head (valid_set3)
valid_knn_cal <- valid_set3 %>% 
  mutate(pred_rnd = round(knn_valid_prob, 1)) %>% 
  group_by(pred_rnd) %>% 
  dplyr::summarize(mean_pred = mean(knn_valid_prob),
                   mean_obs = mean(Subgroup),
                   n = n()
  )
valid_knn_cal
write.csv(valid_knn_cal,"验证集-KNN-校准曲线-mlr3包.csv

          
