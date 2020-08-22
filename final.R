library(data.table)
library(ggplot2)
library(plyr)
library(dplyr)
library(corrplot)
library(pROC)
library(glmnet)
library(caret)
library(Rtsne)
library(xgboost)
library(caret)
library(e1071)
library(caTools)
library(ROSE)
library(smotefamily)
library(rpart)
library(rpart.plot)
library(caTools)
library(ranger)
library(caret)
library(data.table)
library(randomForest)
library(neuralnet)
library(gbm, quietly=TRUE)


#importing datatset
credit_card <- read.csv(file.choose())
str(credit_card)

q<-credit_card

#convert class to a factor variable
credit_card$Class <- factor(credit_card$Class, levels = c(0,1))

summary(credit_card)

#count the missing values
sum(is.na(credit_card))

#Distribution and % of fraud and legit transactions
table(credit_card$Class)
prop.table(table(credit_card$Class))

#Pie chart of credit card transactions
labels <- c("legit", "fraud")
labels <- paste(labels, round(100*prop.table(table(credit_card$Class)),2))
labels <- paste(labels, "%")
pie(table(credit_card$Class), labels, col = c("orange", "red"), main = "Pie chart of credit card transactions")

#no model prediction
predictions <- rep.int(0,nrow(credit_card))
predictions <- factor(predictions, levels = c(0,1))


confusionMatrix(data = predictions, reference = credit_card$Class)


ggplot(data = credit_card, aes(x = V1, y = V2, col = Class)) +
  geom_point() + 
  theme_bw()


#Creating training and test sets
set.seed(123)
data_sample = sample.split(credit_card$Class, SplitRatio = 0.60)
train_data = subset(credit_card, data_sample==TRUE)
test_data = subset(credit_card, data_sample==FALSE)
dim(train_data)
dim(test_data)

#ROS
table(train_data$Class)
n_legit <- 170589
new_frac_legit <- 0.50
new_n_total <- n_legit/new_frac_legit


oversampling_result <- ovun.sample(Class ~ ., data = train_data,
                                   method = "over", N = new_n_total, seed = 2019)
oversampled_credit <- oversampling_result$data
table(oversampled_credit$Class)
ggplot(data = oversampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) + 
  theme_bw()

#RUS
table(train_data$Class)
n_fraud <- 295
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud
undersampling_result <- ovun.sample(Class ~ ., data = train_data,
                                    method = "under",
                                    N = new_n_total,
                                    seed = 2019)
undersampling_credit <- undersampling_result$data
ggplot(data = undersampling_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point() + 
  theme_bw()

#ROS and RUS
n_new <- nrow(train_data)
fraction_fraud_new <- 0.50
sampling_result <- ovun.sample(Class ~ ., data = train_data,
                               method = "both",
                               N = n_new,
                               p = fraction_fraud_new,
                               seed = 2019)
sampled_credit <- sampling_result$data
table(sampled_credit$Class)
prop.table(table(sampled_credit$Class))
ggplot(data = sampled_credit, aes(x = V1, y = V2, col = Class)) +
  geom_point(position = position_jitter(width = 0.1)) + 
  theme_bw()

#SMOTE to balance the data
table(train_data$Class)
n0 <- 170589
n1 <- 295
r0 <- 0.6 

#calculate value for dup_size
ntimes <- ((1-r0)/r0)*(n0/n1)-1
smote_output <- SMOTE(X = train_data[,-c(1,31)],
                      target = train_data$Class,
                      K = 5,
                      dup_size = ntimes)
credit_smote <- smote_output$data
colnames(credit_smote)[30] <- "Class"

#Class distribution for oversampled dataset using SMOTE
prop.table(table(credit_smote$Class))
ggplot(data = credit_smote, aes(x = V1, y = V2, col = Class)) +
  geom_point() + 
  theme_bw()


CART_model <- rpart(Class ~ ., credit_smote)
rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)

#Predict fraud classes
predicted_val <- predict(CART_model, test_data, type = 'class')

#Build confusion matrix
confusionMatrix(predicted_val, test_data$Class)

#Decision tree without SMOTE
CART_model <- rpart(Class ~ ., train_data[,-1])
rpart.plot(CART_model, extra = 0, type = 5, tweak = 1.2)
predicted_val <- predict(CART_model, test_data[,-1], type = 'class')
confusionMatrix(predicted_val, test_data$Class)


#RANDOM FOREST
rf <- q
rf$Class <- as.factor(rf$Class)

rows <- nrow(rf)
cols <- ncol(rf)

set.seed(40)
rf <- rf[sample(rows),]
ntr <- as.integer(round(0.6*rows))

rf.train <- rf[1:ntr, 1:cols]
rf.test <- rf[(ntr+1):rows, -cols]
rf.testc <- rf[(ntr+1):rows, cols]

rf.testc <- as.data.frame(rf.testc)
colnames(rf.testc)[1] <- c("Class")

samp <- as.integer(0.5 * ntr)

model <- randomForest(Class~.,data = rf.train, importance = TRUE, ntree = 35, samplesize = samp, 
                      maxnodes = 45)

rf.pred <- predict(model, rf.test)
rf.testc$Pred <- rf.pred

confusionMatrix(rf.testc$Pred, rf.testc$Class)

rf.testc$Class <- ordered(rf.testc$Class, levels = c("0", "1"))
rf.testc$Pred <- ordered(rf.testc$Pred, levels = c("0", "1"))
auc(rf.testc$Class, rf.testc$Pred)

cur = roc(rf.testc$Class, rf.testc$Pred, plot = TRUE, col = "red")
print(cur)


#LOGISTIC REGRESSION
cc <- q

cc$Amount=scale(cc$Amount)
ND=cc[,-c(1)]

set.seed(100)
data_sample = sample.split(ND$Class,SplitRatio=0.60)
tr_data = subset(ND,data_sample==TRUE)
te_data = subset(ND,data_sample==FALSE)

Log_Mod=glm(Class~.,tr_data,family=binomial())
summary(Log_Mod)
plot(Log_Mod)

lr.p <- predict(Log_Mod,te_data, probability = TRUE)
auc = roc(te_data$Class, lr.p, plot = TRUE, col = "blue")
print(auc)


#NEURAL NET
ANN_mod = neuralnet (Class~.,tr_data,linear.output=FALSE)
plot(ANN_mod)
p.ANN=compute(ANN_mod,te_data)
res.ANN=p.ANN$net.result
res.ANN=ifelse(res.ANN>0.5,1,0)


#GRADIENT BOOSTING
mod_gbm <- gbm(Class ~ .
                   , distribution = "bernoulli"
                   , data = rbind(tr_data, te_data)
                   , n.trees = 500
                   , interaction.depth = 3
                   , n.minobsinnode = 100
                   , shrinkage = 0.01
                   , bag.fraction = 0.5
                   , train.fraction = nrow(tr_data) / (nrow(tr_data) + nrow(te_data))
  )

i = gbm.perf(mod_gbm, method = "test")

mod.inf = relative.influence(mod_gbm, n.trees = i, sort. = TRUE)

plot(mod_gbm)

gbm.test = predict(mod_gbm, newdata = te_data, n.trees = i)
gbm.auc = roc(te_data$Class, gbm.test, plot = TRUE, col = "red")
print(gbm.auc)