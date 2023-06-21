# rf tuning

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)
library(parsnip)

# load data
load("data/results/kitchen_sink.rda")
load("data/clean/data_clean.rda")
load("data/results/rec_2.rda")

tidymodels_prefer()


# register cores
registerDoMC(cores = 8)

# Define model ----
rf_mod <- rand_forest(mode = "classification"
                     ) %>% 
  set_engine("ranger", importance = "impurity")

# set-up tuning grid ----
rf_params <- parameters(rf_mod)

# create grid 
rf_grid <- grid_regular(rf_params, levels = 5)


# workflow ----
rf_workflow <- workflow() %>% 
  add_recipe(rec_2) %>% 
  add_model(rf_mod)

set.seed(1234)

# tune grid ----
rf_tune <- tune_grid(
  rf_workflow,
  resamples = folds_data,
  grid = rf_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)


# save results ----
save(rf_tune, 
     file = "data/results/tuning_rf.rda")


# load data ----
load("data/results/tuning_rf.rda")
rf_tune %>%
  show_best(metric = "roc_auc")


# final workflow ----
final_wkflow <- rf_workflow %>%
  finalize_workflow(select_best(rf_tune, metric = "roc_auc"))

# final fit
fit_final <- fit(final_wkflow, train_data)

final_metrics <- metric_set(roc_auc)

# predictions
rf_pred <- predict(fit_final, test) %>%
  bind_cols(test %>% select(id)) %>%
  rename(y = .pred_class)

# save results
write_csv(rf_pred, file = "submissions/attempt20.csv")

