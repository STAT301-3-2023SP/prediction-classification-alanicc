# mars tune lasso

# load packages
library(tidyverse)
library(tidymodels)

# load data
load(file = "data/results/lasso_recipe.rda")

# create model
mars_mod <- mars(mode = "classification",
                    num_terms = tune(),
                    prod_degree = tune()) %>% 
  set_engine("earth")

# create parameters
mars_params <- extract_parameter_set_dials(mars_mod) %>% 
  update(num_terms = num_terms(range = c(1, 23)))

# create grid
mars_grid <- grid_regular(mars_params, levels = 5)

# create workflow
mars_workflow <- workflow() %>% 
  add_model(mars_mod) %>% 
  add_recipe(lasso_rec)

# tune model
mars_tune <- tune_grid( 
  mars_workflow,
  resamples = class_folds,
  grid = mars_grid,
  control = control_grid(save_pred = TRUE, 
                         save_workflow = TRUE))

# save results
save(mars_tune, mars_workflow,
     file = "data/results/tuned_mars.rda")

# select best
mars_workflow_best <- mars_workflow %>% 
  finalize_workflow(select_best(mars_tune, metric = "roc_auc"))

# fit final workflow
mars_fit <- fit(mars_workflow_best, train)

# predictions
mars_lasso_pred <- predict(mars_fit, test) %>% 
  bind_cols(test %>% 
              select(id)) %>% 
  rename(y = .pred_class)



# save results
write_csv(mars_lasso_pred, file = "data/submissions/attempt_mars_lasso.csv")
