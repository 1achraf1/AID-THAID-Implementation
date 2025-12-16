# ============================================================================
# AID Testing Suite - Legacy Output Compatible + Dataset Harmonization
# ============================================================================
# Goal:
# - Keep the SAME console output structure/section titles as your original tests
# - Make the script runnable (the previous file had broken nesting)
# - Prefer loading shared datasets exported by the Python test suite:
#     shared_datasets/<key>_X.csv and shared_datasets/<key>_y.csv
# - If shared datasets are not present, fall back to classic R datasets (so it still runs)
#
# Expected: AIDRegressor class is available (source("aid.R") before running this file)
# ============================================================================

suppressWarnings({
  if (!require(datasets, quietly = TRUE)) {}
  if (!require(caret, quietly = TRUE)) {}
  if (!require(ggplot2, quietly = TRUE)) {}
  if (!require(gridExtra, quietly = TRUE)) {}
})

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
.rmse <- function(y_true, y_pred) {
  sqrt(mean((as.numeric(y_true) - as.numeric(y_pred))^2))
}

.load_local_dataset <- function(key) {
  x_path <- file.path("shared_datasets", paste0(key, "_X.csv"))
  y_path <- file.path("shared_datasets", paste0(key, "_y.csv"))

  if (file.exists(x_path) && file.exists(y_path)) {
    X <- read.csv(x_path, check.names = FALSE)
    y <- read.csv(y_path, header = FALSE)[, 1]
    return(list(X = X, y = y))
  }
  NULL
}

.split_data <- function(X, y, test_size = 0.3, seed = 42) {
  set.seed(seed)
  X_mat <- as.matrix(X)
  n <- nrow(X_mat)
  train_idx <- sample(seq_len(n), size = floor((1 - test_size) * n))
  test_idx <- setdiff(seq_len(n), train_idx)
  list(
    X_train = X_mat[train_idx, , drop = FALSE],
    X_test  = X_mat[test_idx, , drop = FALSE],
    y_train = as.numeric(y[train_idx]),
    y_test  = as.numeric(y[test_idx])
  )
}

# ------------------------------------------------------------
# Tester class (keeps your original output headings)
# ------------------------------------------------------------
AIDTester <- setRefClass(
  "AIDTester",
  fields = list(
    model_class = "ANY",
    results = "list"
  ),

  methods = list(
    initialize = function(model_class) {
      "Initialize tester with the AIDRegressor class"
      model_class <<- model_class
      results <<- list()
    },

    load_datasets = function() {
      "Load regression datasets for AID testing (prefer shared CSVs)"
      datasets <- list()

      # Preferred shared datasets (exported by Python test suite)
      for (key in c("california", "diabetes", "synthetic_friedman1", "ames", "cpu_act")) {
        d <- .load_local_dataset(key)
        if (!is.null(d)) {
          datasets[[key]] <- list(
            X = d$X,
            y = d$y,
            description = sprintf("%s (shared CSV) (%d samples, %d features)",
                                  key, nrow(d$X), ncol(d$X))
          )
        }
      }

      # Fallbacks if shared_datasets is missing/empty (keeps script runnable)
      if (length(datasets) == 0) {
        # 1) mtcars: mpg regression
        data(mtcars)
        X1 <- mtcars[, c("disp", "hp", "wt", "qsec", "drat")]
        y1 <- mtcars$mpg
        datasets$mtcars_mpg <- list(
          X = X1, y = y1,
          description = sprintf("mtcars->mpg (%d samples, %d features)", nrow(X1), ncol(X1))
        )

        # 2) airquality (remove NAs): Ozone regression
        data(airquality)
        aq <- airquality[complete.cases(airquality), ]
        X2 <- aq[, c("Solar.R", "Wind", "Temp")]
        y2 <- aq$Ozone
        datasets$airquality_ozone <- list(
          X = X2, y = y2,
          description = sprintf("airquality->Ozone (%d samples, %d features)", nrow(X2), ncol(X2))
        )

        # 3) Boston if MASS available
        if (require(MASS, quietly = TRUE)) {
          data(Boston)
          datasets$boston_medv <- list(
            X = Boston[, setdiff(colnames(Boston), "medv")],
            y = Boston$medv,
            description = sprintf("Boston->medv (%d samples, %d features)", nrow(Boston), ncol(Boston) - 1)
          )
        }

        # 4) Synthetic linear
        set.seed(0)
        n <- 20000; p <- 10
        Xs <- matrix(rnorm(n * p), nrow = n, ncol = p)
        ys <- 2 * Xs[, 1] + rnorm(n)
        datasets$synthetic_linear <- list(
          X = Xs, y = ys,
          description = sprintf("Synthetic linear (%d samples, %d features)", nrow(Xs), ncol(Xs))
        )

        # 5) Synthetic non-linear
        set.seed(0)
        Xn <- cbind(runif(800, -2, 2), runif(800, -2, 2))
        yn <- ifelse(Xn[, 1] > 0, 1, 0) + 0.4 * sin(2 * Xn[, 2]) + rnorm(800, sd = 0.2)
        datasets$synthetic_nonlinear <- list(
          X = Xn, y = yn,
          description = sprintf("Synthetic non-linear (%d samples, %d features)", nrow(Xn), ncol(Xn))
        )
      }

      cat(sprintf("Loaded %d datasets\n", length(datasets)))
      datasets
    },

    test_basic_functionality = function(datasets) {
      "Test basic functionality of AID regressor"
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("BASIC FUNCTIONALITY TEST (REGRESSION)\n")
      cat(strrep("=", 80), "\n")

      test_results <- list()

      for (name in names(datasets)) {
        d <- datasets[[name]]
        cat(sprintf("\n%s: %s\n", toupper(name), d$description))
        cat(strrep("-", 60), "\n")

        tryCatch({
          sp <- .split_data(d$X, d$y, test_size = 0.3, seed = 42)

          model <- model_class$new(
            min_samples_split = 30,
            min_samples_leaf = 15,
            max_depth = 6,
            min_split_gain = 1e-3,
            max_features = NULL,
            store_history = TRUE
          )

          t0 <- Sys.time()
          model$fit(sp$X_train, sp$y_train)
          fit_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

          t1 <- Sys.time()
          pred <- model$predict(sp$X_test)
          predict_time <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

          rmse_val <- sqrt(model$mse(sp$X_test, sp$y_test))
          mae_val <- mean(abs(sp$y_test - pred))
          r2_val <- model$score(sp$X_test, sp$y_test)

          n_splits <- if (!is.null(model$history_)) length(model$history_) else 0
          n_leaves <- if (!is.null(model$n_leaves_)) model$n_leaves_ else NA

          test_results[[name]] <- list(
            rmse = rmse_val, mae = mae_val, r2 = r2_val,
            fit_time = fit_time, predict_time = predict_time,
            n_splits = n_splits, n_leaves = n_leaves,
            success = TRUE
          )

          cat(sprintf("✓ RMSE:            %.4f\n", rmse_val))
          cat(sprintf("✓ MAE:             %.4f\n", mae_val))
          cat(sprintf("✓ R²:              %.4f\n", r2_val))
          cat(sprintf("✓ Fit Time:        %.4f s\n", fit_time))
          cat(sprintf("✓ Predict Time:    %.4f s\n", predict_time))
          cat(sprintf("✓ Splits stored:   %d\n", n_splits))
          cat(sprintf("✓ Number of leaves:%s\n", as.character(n_leaves)))

        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          test_results[[name]] <- list(success = FALSE, error = e$message)
        })
      }

      results$basic_functionality <<- test_results
      test_results
    },

    test_cross_validation = function(datasets, cv = 5) {
      "Perform cross-validation testing"
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("CROSS-VALIDATION TEST (REGRESSION)\n")
      cat(strrep("=", 80), "\n")

      cv_results <- list()

      for (name in names(datasets)) {
        d <- datasets[[name]]
        cat(sprintf("\n%s: %s\n", toupper(name), d$description))
        cat(strrep("-", 60), "\n")

        tryCatch({
          set.seed(42)
          X <- as.matrix(d$X)
          y <- as.numeric(d$y)
          n <- nrow(X)

          folds <- caret::createFolds(y, k = cv, list = TRUE)

          rmse_scores <- numeric(cv)
          r2_scores <- numeric(cv)

          for (fold in seq_len(cv)) {
            test_idx <- folds[[fold]]
            train_idx <- setdiff(seq_len(n), test_idx)

            X_train <- X[train_idx, , drop = FALSE]
            X_test  <- X[test_idx, , drop = FALSE]
            y_train <- y[train_idx]
            y_test  <- y[test_idx]

            model <- model_class$new(
              min_samples_split = 30,
              min_samples_leaf = 15,
              max_depth = 6,
              min_split_gain = 1e-3,
              store_history = FALSE
            )
            model$fit(X_train, y_train)

            pred <- model$predict(X_test)
            rmse_scores[fold] <- .rmse(y_test, pred)
            r2_scores[fold] <- model$score(X_test, y_test)

            cat(sprintf("  Fold %d: RMSE=%.4f | R²=%.4f\n", fold, rmse_scores[fold], r2_scores[fold]))
          }

          cv_results[[name]] <- list(
            rmse_scores = rmse_scores,
            r2_scores = r2_scores,
            mean_rmse = mean(rmse_scores),
            std_rmse = sd(rmse_scores),
            mean_r2 = mean(r2_scores),
            std_r2 = sd(r2_scores),
            success = TRUE
          )

          cat(sprintf("\n✓ Mean RMSE: %.4f (+/- %.4f)\n", mean(rmse_scores), sd(rmse_scores)))
          cat(sprintf("✓ Mean R²:   %.4f (+/- %.4f)\n", mean(r2_scores), sd(r2_scores)))

        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          cv_results[[name]] <- list(success = FALSE, error = e$message)
        })
      }

      results$cross_validation <<- cv_results
      cv_results
    },

    test_parameter_sensitivity = function(datasets) {
      "Test sensitivity to different parameter configurations"
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("PARAMETER SENSITIVITY TEST (AID)\n")
      cat(strrep("=", 80), "\n")

      param_configs <- list(
        list(min_samples_split = 10,  min_samples_leaf = 5,  max_depth = 3,  min_split_gain = 0.0),
        list(min_samples_split = 20,  min_samples_leaf = 10, max_depth = 4,  min_split_gain = 1e-4),
        list(min_samples_split = 30,  min_samples_leaf = 15, max_depth = 6,  min_split_gain = 1e-3),
        list(min_samples_split = 60,  min_samples_leaf = 30, max_depth = 8,  min_split_gain = 1e-3),
        list(min_samples_split = 100, min_samples_leaf = 50, max_depth = 10, min_split_gain = 1e-2)
      )

      param_results <- list()

      for (name in names(datasets)) {
        d <- datasets[[name]]
        cat(sprintf("\n%s\n", toupper(name)))
        cat(strrep("-", 60), "\n")

        sp <- .split_data(d$X, d$y, test_size = 0.3, seed = 42)
        X_train <- sp$X_train
        X_test  <- sp$X_test
        y_train <- sp$y_train
        y_test  <- sp$y_test

        config_results <- list()

        for (i in seq_along(param_configs)) {
          params <- param_configs[[i]]
          tryCatch({
            model <- do.call(model_class$new, c(params, list(store_history = FALSE)))
            model$fit(X_train, y_train)

            pred <- model$predict(X_test)
            rmse_val <- .rmse(y_test, pred)
            r2_val <- model$score(X_test, y_test)

            config_results[[i]] <- list(params = params, rmse = rmse_val, r2 = r2_val)

            cat(sprintf(
              "  Config %d: min_samples_split=%d min_samples_leaf=%d max_depth=%d min_split_gain=%g -> RMSE=%.4f | R²=%.4f\n",
              i, params$min_samples_split, params$min_samples_leaf, params$max_depth, params$min_split_gain, rmse_val, r2_val
            ))
          }, error = function(e) {
            cat(sprintf("  Config %d: ERROR - %s\n", i, e$message))
          })
        }

        param_results[[name]] <- config_results
      }

      results$parameter_sensitivity <<- param_results
      param_results
    },

    test_comparison_with_rpart = function(datasets) {
      "Compare AID performance with rpart"
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("COMPARISON WITH RPART (REGRESSION TREE)\n")
      cat(strrep("=", 80), "\n")

      if (!require(rpart, quietly = TRUE)) {
        cat("rpart package not available\n")
        return(invisible(NULL))
      }

      comp_results <- list()

      for (name in names(datasets)) {
        d <- datasets[[name]]
        cat(sprintf("\n%s\n", toupper(name)))
        cat(strrep("-", 60), "\n")

        tryCatch({
          sp <- .split_data(d$X, d$y, test_size = 0.3, seed = 42)

          # AID
          t0 <- Sys.time()
          aid_model <- model_class$new(
            min_samples_split = 30,
            min_samples_leaf = 15,
            max_depth = 6,
            min_split_gain = 1e-3,
            store_history = TRUE
          )
          aid_model$fit(sp$X_train, sp$y_train)
          aid_fit <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

          t1 <- Sys.time()
          aid_pred <- aid_model$predict(sp$X_test)
          aid_pred_t <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

          aid_rmse <- .rmse(sp$y_test, aid_pred)
          aid_r2 <- aid_model$score(sp$X_test, sp$y_test)

          # rpart
          df_train <- data.frame(sp$X_train, y = as.numeric(sp$y_train))
          df_test <- data.frame(sp$X_test)

          t2 <- Sys.time()
          rp <- rpart::rpart(
            y ~ ., data = df_train,
            control = rpart::rpart.control(minsplit = 30, minbucket = 15, cp = 0)
          )
          rp_fit <- as.numeric(difftime(Sys.time(), t2, units = "secs"))

          t3 <- Sys.time()
          rp_pred <- predict(rp, df_test)
          rp_pred_t <- as.numeric(difftime(Sys.time(), t3, units = "secs"))

          rp_rmse <- .rmse(sp$y_test, rp_pred)
          rp_r2 <- 1 - sum((sp$y_test - rp_pred)^2) / sum((sp$y_test - mean(sp$y_test))^2)

          comp_results[[name]] <- list(
            aid_rmse = aid_rmse, aid_r2 = aid_r2,
            rpart_rmse = rp_rmse, rpart_r2 = rp_r2,
            aid_fit_time = aid_fit, rpart_fit_time = rp_fit,
            aid_predict_time = aid_pred_t, rpart_predict_time = rp_pred_t,
            success = TRUE
          )

          cat("AID:\n")
          cat(sprintf("  RMSE:         %.4f\n", aid_rmse))
          cat(sprintf("  R²:           %.4f\n", aid_r2))
          cat(sprintf("  Fit Time:     %.4f s\n", aid_fit))
          cat(sprintf("  Predict Time: %.4f s\n", aid_pred_t))

          cat("\nrpart:\n")
          cat(sprintf("  RMSE:         %.4f\n", rp_rmse))
          cat(sprintf("  R²:           %.4f\n", rp_r2))
          cat(sprintf("  Fit Time:     %.4f s\n", rp_fit))
          cat(sprintf("  Predict Time: %.4f s\n", rp_pred_t))

          cat("\nComparison:\n")
          cat(sprintf("  RMSE Diff (AID - rpart): %+.4f\n", aid_rmse - rp_rmse))
          cat(sprintf("  R² Diff (AID - rpart):   %+.4f\n", aid_r2 - rp_r2))
          cat(sprintf("  Speed Ratio (rpart/AID fit): %.2fx\n", rp_fit / aid_fit))

        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          comp_results[[name]] <- list(success = FALSE, error = e$message)
        })
      }

      results$rpart_comparison <<- comp_results
      comp_results
    },

    test_edge_cases = function(datasets) {
      "Test edge cases with small datasets"
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("EDGE CASES TEST (AID)\n")
      cat(strrep("=", 80), "\n")

      edge_results <- list()

      name <- names(datasets)[1]
      d <- datasets[[name]]

      X <- as.matrix(d$X)
      y <- as.numeric(d$y)

      test_cases <- list(
        single_sample = list(X = X[1, , drop = FALSE], y = y[1]),
        two_samples = list(X = X[1:2, , drop = FALSE], y = y[1:2]),
        small_sample = list(X = X[1:min(10, nrow(X)), , drop = FALSE], y = y[1:min(10, nrow(X))]),
        single_feature = list(X = X[, 1, drop = FALSE], y = y)
      )

      for (test_name in names(test_cases)) {
        cat(sprintf("\n%s\n", toupper(test_name)))
        cat(strrep("-", 60), "\n")

        td <- test_cases[[test_name]]

        tryCatch({
          model <- model_class$new(
            min_samples_split = 2,
            min_samples_leaf = 2,
            max_depth = 3,
            min_split_gain = 0,
            store_history = FALSE
          )
          model$fit(td$X, td$y)
          pred <- model$predict(td$X)
          rmse_val <- .rmse(td$y, pred)

          cat(sprintf("✓ SUCCESS: RMSE = %.4f\n", rmse_val))
          edge_results[[test_name]] <- list(success = TRUE, rmse = rmse_val)

        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          edge_results[[test_name]] <- list(success = FALSE, error = e$message)
        })
      }

      results$edge_cases <<- edge_results
      edge_results
    },

    visualize_results = function() {
      "Generate visualization plots"
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("GENERATING VISUALIZATIONS\n")
      cat(strrep("=", 80), "\n")

      # Basic plots
      if ("basic_functionality" %in% names(results)) {
        data <- results$basic_functionality
        successful <- names(data)[sapply(data, function(x) isTRUE(x$success))]

        if (length(successful) > 0) {
          df <- data.frame(
            dataset = successful,
            rmse = sapply(successful, function(n) data[[n]]$rmse),
            r2 = sapply(successful, function(n) data[[n]]$r2),
            fit_time = sapply(successful, function(n) data[[n]]$fit_time),
            predict_time = sapply(successful, function(n) data[[n]]$predict_time)
          )

          p1 <- ggplot(df, aes(x = dataset, y = rmse)) +
            geom_bar(stat = "identity", fill = "steelblue", alpha = 0.85) +
            labs(title = "AID — RMSE Across Datasets", x = "Dataset", y = "RMSE") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))

          p2 <- ggplot(df, aes(x = dataset, y = r2)) +
            geom_bar(stat = "identity", fill = "darkgreen", alpha = 0.85) +
            labs(title = "AID — R² Across Datasets", x = "Dataset", y = "R²") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))

          p3 <- ggplot(df, aes(x = dataset, y = fit_time)) +
            geom_bar(stat = "identity", fill = "coral", alpha = 0.85) +
            labs(title = "AID — Fit Time", x = "Dataset", y = "Seconds") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))

          p4 <- ggplot(df, aes(x = dataset, y = predict_time)) +
            geom_bar(stat = "identity", fill = "purple", alpha = 0.85) +
            labs(title = "AID — Predict Time", x = "Dataset", y = "Seconds") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))

          tryCatch({
            png("aid_performance.png", width = 1400, height = 900, res = 120)
            gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2)
            dev.off()
            cat("✓ Saved: aid_performance.png\n")
          }, error = function(e) {
            cat(sprintf("Warning: Could not save plots: %s\n", e$message))
          })
        }
      }

      # CV plots
      if ("cross_validation" %in% names(results)) {
        data <- results$cross_validation
        successful <- names(data)[sapply(data, function(x) isTRUE(x$success))]

        if (length(successful) > 0) {
          df_cv <- do.call(rbind, lapply(successful, function(nm) {
            data.frame(
              dataset = nm,
              fold = seq_along(data[[nm]]$rmse_scores),
              rmse = data[[nm]]$rmse_scores,
              r2 = data[[nm]]$r2_scores
            )
          }))

          p_cv_rmse <- ggplot(df_cv, aes(x = fold, y = rmse, color = dataset, group = dataset)) +
            geom_line(linewidth = 1.2) +
            geom_point(size = 2.8) +
            labs(title = "AID — CV RMSE Across Folds", x = "Fold", y = "RMSE") +
            theme_minimal()

          p_cv_r2 <- ggplot(df_cv, aes(x = fold, y = r2, color = dataset, group = dataset)) +
            geom_line(linewidth = 1.2) +
            geom_point(size = 2.8) +
            labs(title = "AID — CV R² Across Folds", x = "Fold", y = "R²") +
            theme_minimal()

          tryCatch({
            png("aid_cv_scores.png", width = 1200, height = 700, res = 120)
            gridExtra::grid.arrange(p_cv_rmse, p_cv_r2, ncol = 2)
            dev.off()
            cat("✓ Saved: aid_cv_scores.png\n")
          }, error = function(e) {
            cat(sprintf("Warning: Could not save CV plots: %s\n", e$message))
          })
        }
      }
    },

    generate_report = function() {
      "Generate comprehensive test report"
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("COMPREHENSIVE TEST REPORT\n")
      cat(strrep("=", 80), "\n")

      report <- c()
      report <- c(report, "\nAID ALGORITHM TEST REPORT (REGRESSION)")
      report <- c(report, strrep("=", 80))
      report <- c(report, sprintf("\nGenerated: %s\n", Sys.time()))

      if ("basic_functionality" %in% names(results)) {
        report <- c(report, "\n1. BASIC FUNCTIONALITY")
        report <- c(report, strrep("-", 80))
        data <- results$basic_functionality
        success_count <- sum(sapply(data, function(x) isTRUE(x$success)))
        report <- c(report, sprintf("Datasets Tested: %d", length(data)))
        report <- c(report, sprintf("Successful: %d/%d", success_count, length(data)))

        for (nm in names(data)) {
          r <- data[[nm]]
          if (isTRUE(r$success)) {
            report <- c(report, sprintf("\n%s:", nm))
            report <- c(report, sprintf("  RMSE:           %.4f", r$rmse))
            report <- c(report, sprintf("  MAE:            %.4f", r$mae))
            report <- c(report, sprintf("  R²:             %.4f", r$r2))
            report <- c(report, sprintf("  Fit Time:       %.4f s", r$fit_time))
            report <- c(report, sprintf("  Predict Time:   %.4f s", r$predict_time))
            report <- c(report, sprintf("  Splits stored:  %d", r$n_splits))
            report <- c(report, sprintf("  Number of leaves:%s", as.character(r$n_leaves)))
          }
        }
      }

      if ("cross_validation" %in% names(results)) {
        report <- c(report, "\n\n2. CROSS-VALIDATION RESULTS")
        report <- c(report, strrep("-", 80))
        data <- results$cross_validation

        for (nm in names(data)) {
          r <- data[[nm]]
          if (isTRUE(r$success)) {
            report <- c(report, sprintf("\n%s:", nm))
            report <- c(report, sprintf("  Mean RMSE: %.4f (+/- %.4f)", r$mean_rmse, r$std_rmse))
            report <- c(report, sprintf("  Mean R²:   %.4f (+/- %.4f)", r$mean_r2, r$std_r2))
          }
        }
      }

      if ("rpart_comparison" %in% names(results)) {
        report <- c(report, "\n\n3. COMPARISON WITH RPART")
        report <- c(report, strrep("-", 80))
        data <- results$rpart_comparison

        for (nm in names(data)) {
          r <- data[[nm]]
          if (isTRUE(r$success)) {
            report <- c(report, sprintf("\n%s:", nm))
            report <- c(report, sprintf("  AID RMSE:   %.4f | rpart RMSE: %.4f | Diff: %+.4f",
                                         r$aid_rmse, r$rpart_rmse, r$aid_rmse - r$rpart_rmse))
            report <- c(report, sprintf("  AID R²:     %.4f | rpart R²:   %.4f | Diff: %+.4f",
                                         r$aid_r2, r$rpart_r2, r$aid_r2 - r$rpart_r2))
            report <- c(report, sprintf("  AID Fit:    %.4f s | rpart Fit:  %.4f s | Ratio: %.2fx",
                                         r$aid_fit_time, r$rpart_fit_time, r$rpart_fit_time / r$aid_fit_time))
          }
        }
      }

      if ("parameter_sensitivity" %in% names(results)) {
        report <- c(report, "\n\n4. PARAMETER SENSITIVITY")
        report <- c(report, strrep("-", 80))
        data <- results$parameter_sensitivity

        for (nm in names(data)) {
          report <- c(report, sprintf("\n%s:", nm))
          configs <- data[[nm]]
          for (i in seq_along(configs)) {
            cfg <- configs[[i]]
            if (!is.null(cfg$params)) {
              report <- c(report, sprintf(
                "  Config %d: min_samples_split=%d, max_depth=%d -> RMSE=%.4f, R²=%.4f",
                i, cfg$params$min_samples_split, cfg$params$max_depth, cfg$rmse, cfg$r2
              ))
            }
          }
        }
      }

      report_text <- paste(report, collapse = "\n")
      cat(report_text, "\n")

      tryCatch({
        writeLines(report_text, "aid_test_report.txt")
        cat("\n✓ Saved: aid_test_report.txt\n")
      }, error = function(e) {
        cat(sprintf("Warning: Could not save report: %s\n", e$message))
      })

      report_text
    }
  )
)

# ------------------------------------------------------------
# MAIN runner (keeps your original heading)
# ------------------------------------------------------------
run_aid_complete_test_suite <- function() {
  "Run complete AID test suite with all tests"
  cat("\n", strrep("=", 80), "\n", sep = "")
  cat("AID ALGORITHM - COMPLETE TEST SUITE (REGRESSION)\n")
  cat(strrep("=", 80), "\n")

  tester <- AIDTester$new(AIDRegressor)

  cat("\nLoading datasets...\n")
  datasets <- tester$load_datasets()
  cat(sprintf("✓ Loaded %d datasets\n", length(datasets)))

  tester$test_basic_functionality(datasets)
  tester$test_cross_validation(datasets, cv = 5)
  tester$test_parameter_sensitivity(datasets)
  tester$test_comparison_with_rpart(datasets)
  tester$test_edge_cases(datasets)

  tester$visualize_results()
  tester$generate_report()

  cat("\n", strrep("=", 80), "\n", sep = "")
  cat("TEST SUITE COMPLETED\n")
  cat(strrep("=", 80), "\n")

  invisible(tester)
}

# If you want to run automatically when sourcing:
# run_aid_complete_test_suite()
