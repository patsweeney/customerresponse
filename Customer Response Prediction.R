
#Load relevant libraries
library(tidyverse)
library(fastDummies)
library(caret)
library(splitstackshape)
library(mlbench)
library(rattle)
library(corrplot)
library(psych)
library(plotly)
library(AMR)
library(ggthemes)
library(RColorBrewer)
library(scales)
library(reshape2)
library(caTools)
library(pROC)
library(party)
library(ROCR)
library(kableExtra)
library(xtable)
library(devtools)
library(GoodmanKruskal)
library(wesanderson)
library(gridExtra)


#Dataset summary
str(bank_full)
View(bank_full)

#Count missing values
sapply(bank_full, function(x) sum(is.na(x)))

#Convert classes to dummies for use as needed
bankdummies <- dummy_cols(bank_full, select_columns = c("job", "poutcome", "marital", "education", "default", "housing", "loan", "month", "y"))

#Drop extraneous 'no' columns in dummy dataframe
bankdummies$default_no <- NULL
bankdummies$housing_no <- NULL
bankdummies$loan_no <- NULL
bankdummies$y_no <- NULL
bankdummies$job <- NULL
bankdummies$marital <- NULL
bankdummies$education <- NULL
bankdummies$default <- NULL
bankdummies$housing <- NULL
bankdummies$loan <- NULL
bankdummies$month <- NULL
bankdummies$contact <- NULL
bankdummies$poutcome <-  NULL
bankdummies$y <- NULL

#Summary statistics
summary <- describe(bankdummies)
summarytrimmed <- summary %>%
select(mean, min, max, sd, se)
summaryround <- round(summarytrimmed, digits = 2)
summaryround

#Find nummber of rows and columns in dataset
dimensions <- dim(bank_full)
dimensions

#Find feature correlations with outcome variable
corrs <- cor(bankdummies, use = "complete.obs")
corrs <- as.data.frame(corrs)
ycorrs <- corrs %>% select(y_yes)
ycorrs

#Find  count and proportion of y outcomes
proptable <- bankdummies %>% freq(y_yes)
proptable

#Find number of classes
agescount <- length(unique(bank_full$age))
jobcount <- length(unique(bank_full$job))
maritalcount <- length(unique(bank_full$marital))
educationcount <- length(unique(bank_full$education))
classes <- c("Age Count" = agescount, "Job Count" = jobcount, "Marital Count" =  maritalcount, "Education" = educationcount)
classes

#Find highly correlated features
library(corrplot)
correlationMatrix <- cor(bankdummies)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.6, names = TRUE)
highlyCorrelated
corrplot(correlationMatrix)

options(scipen=999)  # turn-off scientific notation like 1e+48
  theme_set(theme_bw())  # pre-set the bw theme.

  #Duration vs y
  gg <- ggplot(bankdummies, aes(x=duration, y=y_yes)) +
    geom_jitter(aes(col=loan_yes)) +
    geom_smooth(method = "loess", se = F) +
    labs(subtitle="Campaign Outcome vs Last Contact Duration",
         y="Campaign Outcome",
         x="Last Contact Duration (secs)",
         title="Scatterplot",
         caption = "Source: [Moro et al., 2011]")+
         scale_color_brewer(palette="Set2")
  plot(gg)


  #Poutcome Success vs y
  gg <- ggplot(bankdummies, aes(x=poutcome_success, y=y_yes)) +
    geom_jitter(aes(col=loan_yes)) +
    labs(subtitle="Campaign Outcome vs Previous Campaign Outcome",
         y="Campaign Outcome",
         x="Previous Campaign Outcome",
         title="Scatterplot",
         caption = "Source: [Moro et al., 2011]")+
         scale_color_brewer(palette="Set2")
  plot(gg)


  #Pdays vs y
  gg <- ggplot(bank_full, aes(x=pdays, y=y)) +
    geom_jitter(aes(col=month)) +
    scale_y_discrete() +
    labs(subtitle="Campaign Outcome vs Days Since Last Contact",
         y="Campaign Outcome",
         x="Last Contact (days)",
         title="Scatterplot",
         caption = "Source: [Moro et al., 2011]")+
         scale_color_brewer(palette="Set3")
  plot(gg)

  #Age vs Balance
  gg <- ggplot(bank_full, aes(x=age, y=balance)) +
    geom_jitter(aes(col=marital)) +
    geom_smooth(method = "loess", se = F) +
    scale_y_discrete() +
    labs(subtitle="Age vs Balance",
         y="Balance",
         x="Age",
         title="Scatterplot",
         caption = "Source: [Moro et al., 2011]")+
         scale_color_brewer(palette="Set2")
  plot(gg)


#Monthly vs y
  g <- ggplot(bank_full, aes(month))
  g + geom_bar(aes(fill=y), width = 0.5) +
    theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
    labs(subtitle="Campaign Outcome vs Month",
         y="Number Contacted",
         x="Month",
         title="Histogram",
         caption = "Source: [Moro et al., 2011]")+
         scale_fill_brewer(palette="Set2")

#Job vs y
Job / y (diverging bars)
g <- ggplot(bank_full, aes(job))
g + geom_bar(aes(fill=y), width = 0.5) +
  theme(axis.text.x = element_text(angle=65, vjust=0.6)) +
  labs(subtitle="Campaign Outcome vs Job",
       y="Number Contacted",
       x="Job",
       title="Histogram",
       caption = "Source: [Moro et al., 2011]")+
       scale_fill_brewer(palette="Spectral")



#Sample stratified 10% subset
sample <- stratified(bankdummies, "y_yes", size = 0.1)

#Check that proprtions are representative
prop.table(table(bankdummies$y_yes))
prop.table(table(sample$y_yes))

#Make y outcome a factor
sample$y_yes <- as.factor(sample$y_yes)

#Create test and train datasets
set.seed(123)
indexes <- createDataPartition(sample$y_yes,
                               times = 1,
                               p = 0.7,
                               list = FALSE)
sample.train <- sample[indexes,]
sample.test <- sample[-indexes,]

#Check stratification proportions of test and train are representative
prop.table(table(sample$y_yes))
prop.table(table(sample.train$y_yes))
prop.table(table(sample.test$y_yes))

f1 <- freq(sample$y_yes)
f2 <- freq(sample.train$y_yes)
f3 <- freq(sample.test$y_yes)

#Set parameters: 10-fold cross validation with 2 repeats.
train.control <- trainControl(method = "repeatedcv",
                              number = 10,
                              repeats = 1,
                              verboseIter = TRUE,
                              search = "grid")

#Set up tuning hyperparameters
tune.grid <- expand.grid(eta = c(0.05, 0.075, 0.1),
                         nrounds = c(50, 75, 100),
                         max_depth = 6:8,
                         min_child_weight = c(2.0, 2.25, 2.5),
                         colsample_bytree = c(0.3, 0.4, 0.5),
                         gamma = 0,
                         subsample = 1)

#Create and plot extreme gradient boosting model
xgbmodel <- train(y_yes ~ ., data = sample.train, method = "xgbTree", trControl = train.control, tuneGrid = tune.grid)
xgbmodel

plot(xgbmodel)


#Summarize and plot features by importance
importance <- varImp(xgbmodel, scale = TRUE)
importance
plot(importance)

#Plot important variables against y with scatterplot matrix
pairs(~y_yes + campaign + age,data=sample,
   main="Important Features")


#Compare XGB model with boosted logit model
logitmodel <- train(y_yes ~ ., data = sample.train, method = "LogitBoost", trControl = train.control)
logitmodel
plot(logitmodel)

#Predictions on test data using XGB model and logit model
predsxgb <- predict(xgbmodel, sample.test)
predslogit <- predict(logitmodel, sample.test)

# Test data metrics for XGB
confusionMatrix(predsxgb, sample.test$y_yes)

#Teset data metrics for boosted logit
confusionMatrix(predslogit, sample.test$y_yes)

#Predictions on full set using XGB and boosted logit
bankdummies$y_yes <- as.factor(bankdummies$y_yes)
predsfullxgb <- predict(xgbmodel, bankdummies)
predsfulllogit <- predict(logitmodel, bankdummies)

#Full set data metrics for XGB
confusionMatrix(predsfullxgb, bankdummies$y_yes)
recall(predsfullxgb, bankdummies$y_yes)
precision(predsfullxgb, bankdummies$y_yes)
F_meas(predsfullxgb, bankdummies$y_yes)

#Full data set metrics for boosted logit
confusionMatrix(predsfulllogit, bankdummies$y_yes)
recall(predsfulllogit, bankdummies$y_yes)
precision(predsfulllogit, bankdummies$y_yes)
F_meas(predsfulllogit, bankdummies$y_yes)

#Plot ROC curve for XGB
bankdummies$y_yes <- as.numeric(bankdummies$y_yes)
xgbROC <- roc(predsfullxgb, bankdummies$y_yes)
xgbAUC <- auc(xgbROC)

plot(xgbROC, print.thres="best", print.thres.best.method="closest.topleft")
result.coords <- coords(XGBroc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))
print(result.coords)F

#Plot ROC curve for boosted logit
bankdummies$y_yes <- as.numeric(bankdummies$y_yes)
logitROC <- roc(predsfulllogit, bankdummies$y_yes)
logitAUC <- auc(logitROC)
logitAUC

plot(logitROC, print.thres="best", print.thres.best.method="closest.topleft")
result.coords <- coords(XGBroc, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))
print(result.coords)

plot(xgbROC, print.thres="best", print.thres.best.method="closest.topleft")
result.coords <- coords(logitROC, "best", best.method="closest.topleft", ret=c("threshold", "accuracy"))
print(result.coords)
