# initial exploration

# load packages
library(tidyverse)
library(tidymodels)

test <- read_csv("data/raw/test.csv")
train <- read_csv("data/raw/train.csv")

# set seed
set.seed(1234)

# split data
my_split <- initial_split(train, prop = 0.75, strata = y)

train_data <- training(my_split) %>% mutate(y = as.factor(y))
test_data <- testing(my_split) %>% mutate(y = as.factor(y))

# exploration
boxplot_fun <- function(var = NULL) {
  ggplot(train_data, aes(x = !!sym(var), y = y, group = y)) +
    geom_boxplot()
}

boxplot_log_fun <- function(var = NULL) {
  ggplot(train_data, aes(x = !!sym(var), y = log(y), group = y)) +
    geom_boxplot()
}

# create zero variance
var_list <- list()
for(var in colnames(train_data)){
  if (var != "y") {
    var_list[var] <- train_data %>% 
      select(any_of(var)) %>% 
      summarize(sd = sd(!!sym(var), na.rm = TRUE))
  }
}

var_tbl <- enframe(unlist(var_list))

zero_var <- var_tbl %>% 
  filter(value == 0) %>% 
  pull(name)

train_data <- train_data %>% 
  select(!all_of(zero_var))

# miscoded categorical variables
cat_list <- list()

for(var in colnames(train_data)){
  cat_list[var] <- train_data %>% 
    select(any_of(var)) %>% 
    summarize(unique = length(unique(!!sym(var))))
}

cat_tbl <- enframe(unlist(cat_list))

cat_var <- cat_tbl %>% 
  filter(value <= 10) %>% 
  pull(name)

boxplot_fun <- function(var = NULL) {
  ggplot(train_data, aes(x = !!sym(var), y = y, group = y)) +
    geom_boxplot()
}

boxplot_log_fun <- function(var = NULL) {
  ggplot(train_data, aes(x = !!sym(var), y = log(y), group = y)) +
    geom_boxplot()
}

# save data
write_rds(train_data, "data/results/train_data.rds")
write_rds(test_data, "data/results/test_data.rds")
