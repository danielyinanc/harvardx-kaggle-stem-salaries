#Dependencies installed if required
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(janitor)) install.packages("janitor", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")

# GPU version needs to be installed from daily
#if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")

# Enable required dependencies
library(tidyverse)
library(caret)
library(data.table)
library(readr)
library(janitor)
library(corrplot)
library(xgboost)
library(caretEnsemble)

fl <- 'data-science-and-stem-salaries.zip'
raw_fl <- 'Levels_Fyi_Salary_Data.csv'
unzip(fl)
raw_data <- read_csv(raw_fl)

raw_data_sanitized <- clean_names(raw_data)

# Drop id columns and derived columns
raw_data_sanitized <- raw_data_sanitized %>% select(-cityid, -dmaid, -row_number)

# Drop base_salary and bonus and stock grant variables as they are part of totalyearlycompensation
# Drop highly correlated columns
raw_data_sanitized <- raw_data_sanitized %>% select(-basesalary, -stockgrantvalue, -bonus)


summary(raw_data_sanitized)



# Create train and test datasets -- 0.9 train, 0.1 test

set.seed(1,sample.kind="Rounding")
size <- nrow(raw_data_sanitized)

relevant_data <- raw_data_sanitized %>% select(totalyearlycompensation, yearsofexperience, company, location)
complete_data <- relevant_data[sample(nrow(raw_data_sanitized), size),]

test_index <- createDataPartition(y=complete_data$totalyearlycompensation, times=1, p=0.1, list=FALSE)

train_set <- complete_data[-test_index,]
temp <- complete_data[test_index,]

test_set <- temp %>% 
  semi_join(train_set, by = "company") %>%
  semi_join(train_set, by = "location")
  
# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

# Base model
base_lm <- lm(totalyearlycompensation ~ yearsofexperience + company + location, data=train_set)
lm_predictions <- predict(base_lm, test_set, type="response")
lm_rmse <- RMSE(lm_predictions,test_set$totalyearlycompensation)
lm_rmse


# Hyperparameter Tuning
xg_tune_grid <- expand.grid(
  nrounds = seq(from = 200, to = 1000, by = 50),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

train_control_default <- caret::trainControl(
  method = "cv",
  number=2,
  verboseIter = FALSE, # no training log
  allowParallel = TRUE, # FALSE for reproducible results 
  savePredictions="final"
)

train_control_none <- caret::trainControl(
  method = "none",
  savePredictions="final",
)

library(doParallel)
cl <- makePSOCKcluster(4)
registerDoParallel(cl)

lm_optimized <- train(totalyearlycompensation ~ yearsofexperience + company + location,
                      data=train_set,
                      method="lm",
                      trControl = train_control_default)

glmnet_optimized <- train(totalyearlycompensation ~ yearsofexperience + company + location,
                          data=train_set,
                          method="glmnet",
                          na.action  = na.pass,
                          trControl = train_control_default)


## When you are done:
stopCluster(cl)
registerDoSEQ()



# GPU Models

xg_optimized_model <- train(totalyearlycompensation ~ yearsofexperience + company + location,
                          data=train_set,
                          trControl = train_control_default,
                          tuneGrid = xg_tune_grid,
                          method = "xgbTree",
                          tree_method="gpu_hist",
                          verbose = TRUE)

xg_normal_model <- train(totalyearlycompensation ~ yearsofexperience + company + location,
                            data=train_set,
                            trControl = train_control_default,
                            method = "xgbTree",
                            tree_method="gpu_hist",
                            verbose = TRUE)

lm_optimized_predictions <- predict(lm_optimized, test_set, type="raw")
lm_optimized_rmse <- RMSE(lm_optimized_predictions,test_set$totalyearlycompensation)


glm_optimized_predictions <- predict(glmnet_optimized, test_set, type="raw")
glm_optimized_rmse <- RMSE(glm_optimized_predictions,test_set$totalyearlycompensation)


xg_default_predictions <- predict(xg_default_model, test_set, type="raw")
xg_default_rmse <- RMSE(xg_default_predictions,test_set$totalyearlycompensation)


xg_optimized_predictions <- predict(xg_optimized_model, test_set, type="raw")
xg_optimized_rmse <- RMSE(xg_optimized_predictions,test_set$totalyearlycompensation)

xg_normal_predictions <- predict(xg_normal_model, test_set, type="raw")
xg_normal_rmse <- RMSE(xg_normal_predictions,test_set$totalyearlycompensation)

# Top 20 important variables of the model
varImp(xg_optimized_model)

# Standard deviation of predicted variable
sd(complete_data$totalyearlycompensation)

# Error as a percentage of standard deviation
round(xg_optimized_rmse/sd(complete_data$totalyearlycompensation) * 100)

# Maximum compensation
max(test_set$totalyearlycompensation)

# Minimum compensation
min(test_set$totalyearlycompensation)

results <- tibble(method="Linear Regression", rmse=lm_rmse)
results <- results %>% add_row(method="ElasticNet Regression", rmse=glm_optimized_rmse)
results <- results %>% add_row(method="XGBoost Default Tuning", rmse=xg_normal_rmse)
results <- results %>% add_row(method="XGBoost Custom Tuned", rmse=xg_optimized_rmse)

saveRDS(results,"final_results.RDS")