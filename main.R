## ----setup, include=FALSE--------------------------------------------------------------------------
knitr::opts_chunk$set(echo = TRUE)


## --------------------------------------------------------------------------------------------------
#load the data set and the required packages
library(tidyverse)
library(FSelector)
library(caTools)
library(randomForest)
library(caret)
library(mltools)
library(data.table)
library(ROSE)
library(e1071)
#install.packages(c("tree","maptree"))
library(tree)
library(maptree)
#install.packages("gbm")
library(gbm)
#install.packages("pROC")
library(pROC)
#install.packages("CustomerScoringMetrics")
library(CustomerScoringMetrics)
library(ggplot2)
df <- read_csv("assignment_data.csv")

summary(df)


## --------------------------------------------------------------------------------------------------
sum(is.na(df))
na_df <- df[rowSums(is.na(df)) > 0,]


## --------------------------------------------------------------------------------------------------
summary(na_df)


## --------------------------------------------------------------------------------------------------

df <- na.omit(df) #228 rows are omitted

na_df <- na_df %>% filter(CLASS == 1)

df <- rbind(df,na_df)

summary(df)
df2 <- df %>% filter(CLASS == 1) %>% 
   mutate_at(vars(-group_cols()),~ifelse(is.na(.),
                                         mean(.,na.rm=TRUE),.))


df <- df %>% filter(CLASS == 0)

df <- rbind(df,df2)

df <- floor(df) #Round off the decimals calculated when replacing NA's with variable mean

summary(df) #Now all NA's replaced with the variable mean



## --------------------------------------------------------------------------------------------------
df[,c(3:5,7,26:39)] <- lapply(df[,c(3:5,7,26:39)], as.factor)



## --------------------------------------------------------------------------------------------------

df$ID <- NULL

df$CM_HIST <- NULL



## --------------------------------------------------------------------------------------------------
# Check distribution of some of the attributes
dataplot <- df
dataplot$PY1 <- factor(dataplot$PY1, levels = c("-2", "-1","0","1","2","3","4","5","6","7","8"), 
                       labels = c ("No consumption/transaction", "Paid in full","small payment","payment delay for 1 period","payment delay for 2 periods","payment delay for 3 periods","payment delay for 4 periods","payment delay for 5 periods","payment delay for 6 periods","payment delay for 7 periods","payment delay for 8 periods"))
dataplot$GENDER <- factor(dataplot$GENDER, levels = c("1","2"), 
                          labels = c ("Male","Female"))
dataplot$EDUCATION <- factor(dataplot$EDUCATION, levels = c("0","1","2","3","4","5","6"), 
                             labels = c ("Others","Graduate school","University","High school","Others#2","Special program","Unknown"))
ggplot(dataplot) + geom_bar(aes(x=PY1,fill=PY1))+
  theme(axis.text.x=element_text(angle=45,hjust=1)) + ggtitle("The repayment status in period X")
ggplot(dataplot) + geom_bar(aes(x=GENDER,fill=GENDER)) + ggtitle("Gender")
ggplot(dataplot) + geom_bar(aes(x=EDUCATION,fill=EDUCATION))+
  theme(axis.text.x=element_text(angle=45,hjust=1)) + ggtitle("Education")


## --------------------------------------------------------------------------------------------------
#Now partition the data into test and training
set.seed(112233)

partition = sample.split(df$CLASS, SplitRatio = 0.8)

training = subset(df, partition == T)

test = subset(df, partition == F)


## --------------------------------------------------------------------------------------------------
#Find the information gain of the attributes
ig <- information.gain(CLASS ~ ., training)
#Sort the information gain into descending order
sorted_weights <- ig[order(-ig$attr_importance), , drop = F]
#Plot the information gain using barplot
barplot(unlist(sorted_weights), names.arg = rownames(sorted_weights), las = "2", cex.names = 0.7, ylim = c(0,0.08), space = 0.5)



## --------------------------------------------------------------------------------------------------
filtered_attr <- cutoff.k(ig, 19)
filtered_attr
datamodelling <- training[filtered_attr]
datamodelling$CLASS <- training$CLASS
summary(datamodelling)


## --------------------------------------------------------------------------------------------------
#Now model with random forest using the datamodelling dataset
set.seed(1100)

rf_default <- randomForest(CLASS ~ ., datamodelling)
predict_1 <- predict(rf_default, test)
confusionMatrix(predict_1,test$CLASS, positive='1', mode = "prec_recall")

#Accuracy: 0.8291, Precision: 0.68671, Recall: 0.42642, F1: 0.52613, No info rate: 0.7775




## --------------------------------------------------------------------------------------------------

#rf with all attributes

all_rf <- randomForest(CLASS ~., training)
predict_2 <- predict(all_rf, test)
confusionMatrix(predict_2, test$CLASS, positive='1', mode = "prec_recall")
#Accuracy: 0.8274, precision:0.68224, recall:0.41924, F1: 0.51934
#Similar to using attributes with the highest information gain


## --------------------------------------------------------------------------------------------------
#undersampling
undersampled <- ovun.sample(CLASS ~ ., data = datamodelling, method = "under", p=0.4, seed=12321)$data


## --------------------------------------------------------------------------------------------------
#Now model with random forest using the datamodelling dataset
set.seed(1100)

rf_default <- randomForest(CLASS ~ ., undersampled)
predict_1 <- predict(rf_default, test)
confusionMatrix(predict_1,test$CLASS, positive='1', mode = "prec_recall")

#Accuracy: 0.7934, Precision: 0.5322, Recall: 0.5865, F1: 0.5581, No info rate: 0.7775




## --------------------------------------------------------------------------------------------------

set.seed(1100)

rf_default <- randomForest(CLASS ~ ., datamodelling)
predict_1 <- predict(rf_default, test)
confusionMatrix(predict_1,test$CLASS, positive='1', mode = "prec_recall")

#Accuracy: 0.8291, Precision: 0.68671, Recall: 0.42642, F1: 0.52613, No info rate: 0.7775




## --------------------------------------------------------------------------------------------------
#SVM
svm_model <- svm(CLASS ~., data= undersampled , kernal = "radial", scale = TRUE, probability = TRUE)
confusionMatrix(predict(svm_model,test),test$CLASS, positive='1', mode = "prec_recall")
#Accuracy:0.7911, precision: 0.5330, recall:0.4925, F1:0.5119


## --------------------------------------------------------------------------------------------------
#SVM
svm_model <- svm(CLASS ~., data= datamodelling , kernal = "radial", scale = TRUE, probability = TRUE)
confusionMatrix(predict(svm_model,test),test$CLASS, positive='1', mode = "prec_recall")
#Accuracy:0.7911, precision: 0.5330, recall:0.4925, F1:0.5119


## --------------------------------------------------------------------------------------------------

LR_spam <- glm(CLASS ~. , data = undersampled, family = "binomial")
LR_prob <- predict(LR_spam, test, type = "response")
LR_class <- ifelse(LR_prob >= 0.49, "1", "0")
LR_class <- as.factor(LR_class)
confusionMatrix(LR_class,test$CLASS, positive='1', mode = "prec_recall")
#0.7905, 0.5326, 0.4745, 0.5019


## --------------------------------------------------------------------------------------------------

LR_spam <- glm(CLASS ~. , data = datamodelling, family = "binomial")
LR_prob <- predict(LR_spam, test, type = "response")
LR_class <- ifelse(LR_prob >= 0.295, "1", "0") #0.448 & 0.295
LR_class <- as.factor(LR_class)
confusionMatrix(LR_class,test$CLASS, positive='1', mode = "prec_recall")
#0.794, 0.5426, 0.4709, 0.5042


## --------------------------------------------------------------------------------------------------
#Decision Tree
 
tree_spam <- tree(CLASS ~., undersampled)
predict_tree <- predict(tree_spam, test, type = "class")
confusionMatrix(predict_tree,test$CLASS, positive='1', mode = "prec_recall")
#Accuracy: 0.7708, Precision:0.4855, Recall: 0.5047, F1: 0.4949


## --------------------------------------------------------------------------------------------------
#Decision Tree
 
tree_spam <- tree(CLASS ~., datamodelling)
predict_tree <- predict(tree_spam, test, type = "class")
confusionMatrix(predict_tree,test$CLASS, positive='1', mode = "prec_recall")
#Accuracy: 0.8183, Precision:0.69347, Recall: 0.32807, F1: 0.44542


## --------------------------------------------------------------------------------------------------
# GBM requires target variable to be numeric, so make a copy of the undersampled set to avoid confusion
datamodelling_GBM <- undersampled # change to datamodelling to use the whole data set
datamodelling_GBM$CLASS <- as.numeric(datamodelling_GBM$CLASS) -1
summary(datamodelling_GBM$CLASS)



## --------------------------------------------------------------------------------------------------

set.seed(1100)
GBM_fit_default <- gbm(formula = CLASS ~.,
    #distribution = "adaboost",
    distribution = "bernoulli",
    data = datamodelling_GBM)

GBM_d_prob <- predict(GBM_fit_default, test, type = "response")

GBM_d_pred <- ifelse(GBM_d_prob > 0.2875, "1", "0") #0.475 0.2875

GBM_d_pred <- as.factor(GBM_d_pred)

confusionMatrix(GBM_d_pred,test$CLASS, positive='1', mode = "prec_recall")



## --------------------------------------------------------------------------------------------------

set.seed(1100)
GBM_fit_default <- gbm(formula = CLASS ~.,
    #distribution = "adaboost",
    distribution = "bernoulli",
    data = datamodelling_GBM)

GBM_d_prob <- predict(GBM_fit_default, test, type = "response")

GBM_d_pred <- ifelse(GBM_d_prob > 0.47, "1", "0") #0.47

GBM_d_pred <- as.factor(GBM_d_pred)

confusionMatrix(GBM_d_pred,test$CLASS, positive='1', mode = "prec_recall")
#0.8183, 0.66580, 0.36755, 0.47364


## --------------------------------------------------------------------------------------------------


hyper_grid <- expand.grid(
  n.trees = c(500,1000),
  interaction.depth = c( 5,8),
  shrinkage = c(.001, .01),
  n.minobsinnode = c(10,15,20),
  bag.fraction = c(.5,1)
)
err_GBM <- c()

for (i in 1:nrow(hyper_grid)){
    set.seed(1100)
    
  model_GBM <- gbm(
    formula = CLASS ~.,
    distribution = "bernoulli",
    data = datamodelling_GBM, 
    n.trees = hyper_grid$n.trees[i],
    interaction.depth = hyper_grid$interaction.depth[i],
    n.minobsinnode = hyper_grid$n.minobsinnode[i],
    bag.fraction = hyper_grid$bag.fraction[i]
  )
err_GBM[i] <- tail(model_GBM$train.error, 1)

}

best_comb <- which.min(err_GBM)
print(hyper_grid[best_comb,])
#1000， 8， 0.001， 10， 0.5


## --------------------------------------------------------------------------------------------------
set.seed(1100)


GBM_fit_final <- gbm(
    formula = CLASS ~.,
    distribution = "bernoulli",
    data = datamodelling_GBM,
    n.trees = 1000,
    interaction.depth = 8,
    shrinkage = 0.001,
    n.minobsinnode = 10,
    bag.fraction = 0.5,
    cv.folds = 5
  )

ntree_opt <- gbm.perf(GBM_fit_final, method = "cv")

GBM_fit_prob <- predict(GBM_fit_final, test,n.trees = ntree_opt, type = "response")

GBM_fit_pred <- ifelse(GBM_fit_prob > 0.46, "1", "0") 

GBM_fit_pred <- as.factor(GBM_fit_pred)

confusionMatrix(GBM_fit_pred,test$CLASS, positive='1', mode = "prec_recall")



## --------------------------------------------------------------------------------------------------


sampsize_val <- floor(nrow(undersampled)*c(0.5, 0.65, 0.8, 1))

hyper_grid_rf <- expand.grid(
mtry = c(4,5,6),
nodesize = c(1,3,5,8,10),
sampsize = sampsize_val
)

err <- c()

for (i in 1:nrow(hyper_grid_rf)){
    set.seed(1100)
    
    model <- randomForest(CLASS ~ ., undersampled,
                          mtry = hyper_grid_rf$mtry[i],
                          nodesize = hyper_grid_rf$nodesize[i],
                          sampsize = hyper_grid_rf$sampsize[i])

err[i] <- model$err.rate[nrow(model$err.rate), "OOB"]

}

best_comb <- which.min(err)
print(hyper_grid_rf[best_comb,])


## --------------------------------------------------------------------------------------------------
RF_opt <- randomForest(CLASS ~., undersampled, mtry = 4, nodesize = 3, sampsize = 9048)
predict_RF_opt <- predict(RF_opt, test)

confusionMatrix(predict_RF_opt,test$CLASS, positive='1', mode = "prec_recall")


## --------------------------------------------------------------------------------------------------
#Preparation

SVMpred <- predict(svm_model, test, probability = TRUE)
prob_SVM <- attr(SVMpred, "probabilities")

TreePred <- predict(tree_spam, test, probability = TRUE)
prob_Tree <- NULL

prob_RF_default <- predict(rf_default, test, type = "prob")
ROC_RF_default <- roc(test$CLASS, prob_RF_default[,2])
df_RF_default = data.frame((1-ROC_RF_default$specificities), ROC_RF_default$sensitivities)

ROC_SVM <- roc(test$CLASS, prob_SVM[,1])
df_SVM = data.frame((1-ROC_SVM$specificities), ROC_SVM$sensitivities)

ROC_GBM_default <- roc(test$CLASS, GBM_d_prob)
df_GBM_default = data.frame((1-ROC_GBM_default$specificities), ROC_GBM_default$sensitivities)

ROC_LR <- roc(test$CLASS, LR_prob)
df_LR = data.frame((1-ROC_LR$specificities), ROC_LR$sensitivities)

ROC_Tree <- roc(test$CLASS, TreePred[,1])
df_Tree = data.frame((1-ROC_LR$specificities), ROC_LR$sensitivities)


#AUC for 5 models
auc(ROC_RF_default)
auc(ROC_GBM_default)
auc(ROC_SVM)
auc(ROC_LR)
auc(ROC_Tree)


#Graph
plot(df_RF_default, col="red", type="l",        # first adds ROC curve for Random Forest
xlab="False Positive Rate (1-Specificity)", ylab="True Positive Rate (Sensitivity)")
lines(df_SVM, col="blue")               # adds ROC curve for SVM
lines(df_LR, col="green")
lines(df_GBM_default, col = "brown")
lines(df_Tree, col = "purple")
abline(a = 0, b = 1, col = "lightgray") # adds a diagonal line

legend("bottomright",
c("Random Forest", "SVM","Logistic Regression", "GBM","Decision Tree"),
fill=c("red", "blue","green","brown", "purple"))


## --------------------------------------------------------------------------------------------------
# Provide probabilities for the interested outcome and obtain the gain chart data

GainTable_SVM <- cumGainsTable(prob_SVM[,2], test$CLASS, resolution = 1/100)

GainTable_RF <- cumGainsTable(prob_RF_default[,2], test$CLASS, resolution = 1/100)

GainTable_GBM <- cumGainsTable(GBM_d_prob, test$CLASS, resolution = 1/100)

GainTable_LR <- cumGainsTable(LR_prob, test$CLASS, resolution = 1/100)

GainTable_Tree <- cumGainsTable(TreePred[,2], test$CLASS, resolution = 1/100)

#Plot the gain chart
plot(GainTable_SVM[,4], col="blue", type="l",    
xlab="Percentage of test instances", ylab="Percentage of correct predictions")
lines(GainTable_RF[,4], col="red", type ="l")
lines(GainTable_GBM[,4], col="brown", type ="l")
lines(GainTable_LR[,4], col="green", type ="l")
lines(GainTable_Tree[,4], col="purple", type ="l")
abline(a = 0, b = 1, col = "lightgray")
grid(NULL, lwd = 1)

legend("bottomright",
c("SVM", "Random Forest", "GBM", "LR", "Decision Tree"),
fill=c("blue","red", "brown", "green", "purple"))


## --------------------------------------------------------------------------------------------------
#After tuning
ROC_GBM <- roc(test$CLASS, GBM_fit_prob)
df_GBM = data.frame((1-ROC_GBM$specificities), ROC_GBM$sensitivities)

prob_RF <- predict(RF_opt, test, type = "prob")
ROC_RF <- roc(test$CLASS, prob_RF[,2])
df_RF = data.frame((1-ROC_RF$specificities), ROC_RF$sensitivities)

auc(ROC_RF)

auc(ROC_GBM)




## --------------------------------------------------------------------------------------------------
# try to manipulate the data set
df2 <- read_csv("assignment_data.csv")
df2 <- na.omit(df2)

df2 %>% group_by(CLASS) %>% summarise(sum = sum(BILL1, na.rm = T))

