# initial setup

# load packages
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(skimr)

tidymodels_prefer()

# load data
train <- read_csv("data/raw/train.csv")
test <- read_csv("data/raw/test.csv")

# set seed
set.seed(3013)

# splitting the data
initial_split <- initial_split(train, prop = 0.75, strata = y)

train_data <- training(initial_split)
test_data <- testing(initial_split)

# skim for missingness
skim_without_charts(train)
skim(train)

# test the data
ggplot(train_data, aes(x = y)) +
  geom_bar() +
  theme_minimal()

# address missingness
missing_list <- list()
var <- "x001"

for (var in colnames(train_data)) {
  missing_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    filter(is.na(!!sym(var))) %>%  
    summarize(num_missing = n())
}

missing_tibble <- enframe(unlist(missing_list))

missing_tibble %>% 
  mutate(pct = value/4034) %>% 
  arrange(desc(pct)) 

# variance
var_list <- list()

for (var in colnames(train_data)) {
  var_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    summarize(sd = sd(!!sym(var), na.rm = TRUE))
}

var_table <- enframe(unlist(var_list))

zero_var <- var_table %>% 
  filter(value == 0) %>% 
  pull(name)

# turn data into factors
train_data <- train_data %>% 
  mutate(y = factor(y, levels = c(0,1)))

# create folds
folds_data <- vfold_cv(train_data, v = 5, repeats = 3, strata = y)

# create recipe
kitchen_sink <- recipe(y ~ ., data = train_data) %>% 
  step_rm(id) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_predictors()) 


# save data
save(train, train_data, test, test_data, file = "data/clean/data_clean.rda")
save(kitchen_sink, folds_data, file = "data/results/kitchen_sink.rda")
