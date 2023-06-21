# nn tuning

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(tictoc)
library(doMC)

# load data
load("data/results/kitchen_sink.rda")
load("data/clean/data_clean.rda")
load("data/results/rec_2.rda")

tidymodels_prefer()


# register cores
registerDoMC(cores = 8)

# create model
nn_mod <- mlp(mode = "classification",
                  hidden_units = tune(),
                  penalty = tune()) %>% 
  set_engine("nnet")

# create paraams
nn_params <- extract_parameter_set_dials(nn_mod)

# create grid
nn_grid <- grid_regular(nn_params, levels = 5)

# create workflow
nn_workflow <- workflow() %>%
  add_model(nn_mod) %>%
  add_recipe(rec_2)

set.seed(1234)

# tune grid ----
nn_tune <- tune_grid(
  nn_workflow,
  resamples = folds_data,
  grid = nn_grid,
  control = control_grid(save_pred = TRUE,
                         save_workflow = TRUE,
                         parallel_over = "everything"),
  metrics = metric_set(roc_auc)
)


# save results 
save(nn_tune, 
     file = "data/results/tuning_nn.rda")


# load data
load("data/results/tuning_nn.rda")
nn_tune %>%
  show_best(metric = "roc_auc")


# final workflow
final_wkflow <- nn_workflow %>%
  finalize_workflow(select_best(nn_tune, metric = "roc_auc"))

# final fit
fit_final <- fit(final_wkflow, train_data)

final_metrics <- metric_set(roc_auc)

# predictions
nn_pred <- predict(fit_final, test) %>%
  bind_cols(test %>% select(id)) %>%
  rename(y = .pred_class)

# save results
write_csv(nn_pred, file = "submissions/attempt20.csv")

