# lasso var select

# load packages
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(skimr)

tidymodels_prefer()

# load data
load("data/clean/data_clean.rda")
load("data/results/kitchen_sink.rda")

# create model ----
lasso_mod <- logistic_reg(mode = "classification", 
                            penalty = tune(), 
                            mixture = 1) %>% 
  set_engine("glmnet")

# set parameters ----
lasso_params <- extract_parameter_set_dials(lasso_mod) %>% 
  update(penalty = penalty(range = c(0.01, 0.1), trans = NULL))

# create grid ----
lasso_grid <- grid_regular(lasso_params, levels = 5)

# create workflow ----
lasso_workflow <- workflow() %>% 
  add_model(lasso_mod) %>% 
  add_recipe(kitchen_sink)

# tune model ----
lasso_tune <- lasso_workflow %>% 
  tune_grid(resamples = folds_data, 
            grid = lasso_grid,
            control = control_grid(save_pred = TRUE,
                                   save_workflow = TRUE,
                                   parallel_over = "everything"),
            metric = metric_set("roc_auc"))

# save tuned model ----
save(lasso_tune, file = "data/results/lasso_tune.rda")

# load results ----
load("data/results/lasso_tune.rda")

# create final workflow ----
lasso_workflow_final <- lasso_workflow %>%
  finalize_workflow(select_best(lasso_tune, metric = "roc_auc"))

# fit final model ----
lasso_fit <- fit(lasso_workflow_final, data = train_data)

# variance ----
lasso_var_select <- lasso_fit %>%
  tidy() %>%
  filter(estimate != 0, term != "(Intercept)") %>%
  pull(term)

print(lasso_var_select)