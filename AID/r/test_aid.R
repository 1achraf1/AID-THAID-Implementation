# ============================================================================
# AID (Automatic Interaction Detection) Regressor in R
# Morgan & Sonquist (1963) — 
# ============================================================================

# -----------------------------
# Utilities
# -----------------------------
aid_validate_input <- function(X, y) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
    feature_names <- colnames(X)
  } else {
    X <- as.matrix(X)
    feature_names <- paste0("X", seq_len(ncol(X)))
  }

  storage.mode(X) <- "double"
  y <- as.numeric(y)

  if (nrow(X) != length(y)) {
    stop(sprintf("Shape mismatch: X has %d samples, y has %d", nrow(X), length(y)))
  }
  if (any(is.na(X)) || any(is.infinite(X))) stop("X contains NA or Inf")

  list(X = X, y = y, feature_names = feature_names)
}

aid_sse_from_stats <- function(sum_y, sum_y2, n) {
  if (n <= 0) return(0)
  sum_y2 - (sum_y * sum_y) / n
}

aid_sse_vec <- function(y) {
  # SSE = sum(y^2) - (sum(y)^2)/n
  n <- length(y)
  if (n <= 0) return(0)
  sum(y * y) - (sum(y)^2) / n
}

# Complexity (training, rough but useful for reporting)
aid_training_complexity <- function(n, p, Q, max_leaves = Inf) {
  # At each node: for each feature sort O(n_node log n_node) + scan O(n_node).
  # Worst-case: roughly O(p * sum_nodes n_node log n_node).
  # A common upper-bound for balanced-ish splits: O(p * n * log(n) * depth).
  # We'll report the classical practical bound:
  list(
    time = sprintf("O(p * n * log(n) * Q) (approx., numeric features; per-node sorting)"),
    memory = sprintf("O(n + number_of_nodes) (data + tree); history adds O(number_of_splits)")
  )
}

# -----------------------------
# Node class 
# -----------------------------
AIDNode <- setRefClass("AIDNode",
  fields = list(
    depth = "numeric",
    n_samples = "numeric",
    sse = "numeric",
    mean = "numeric",
    split_feature_idx = "numeric",
    split_value = "numeric",
    gain = "numeric",
    f_stat = "numeric",
    left = "ANY",
    right = "ANY",
    is_leaf = "logical"
  ),
  methods = list(
    initialize = function() {
      depth <<- 0
      n_samples <<- 0
      sse <<- 0.0
      mean <<- 0.0
      split_feature_idx <<- NA_real_
      split_value <<- NA_real_
      gain <<- 0.0
      f_stat <<- NA_real_
      left <<- NULL
      right <<- NULL
      is_leaf <<- TRUE
    }
  )
)

# -----------------------------
# AID Regressor 
# -----------------------------
AIDRegressor <- setRefClass("AIDRegressor",
  fields = list(
    # Hyperparameters (names intentionally similar to your AID)
    R = "numeric",           # min_child_size
    M = "numeric",           # min_samples_split (node must have >=M to attempt split)
    Q = "numeric",           # max_depth
    min_gain = "numeric",
    store_history = "logical",
    max_leaves = "numeric",

    # Fitted state 
    root_ = "ANY",
    feature_names_ = "ANY",
    n_features_ = "numeric",

    # Optional stored diagnostics
    history_ = "list",
    leaves_ = "numeric"
  ),

  methods = list(
    initialize = function(R = 5, M = 10, Q = 5, min_gain = 0.0,
                          store_history = FALSE, max_leaves = Inf) {
      "Initialize AID regressor (SSE reduction tree).
      Parameters:
        R: minimum samples per child
        M: minimum samples in node to try split
        Q: maximum depth (root depth=0)
        min_gain: minimum SSE reduction to accept split
        store_history: store split history
        max_leaves: cap on number of leaves"
      R <<- as.numeric(R)
      M <<- as.numeric(M)
      Q <<- as.numeric(Q)
      min_gain <<- as.numeric(min_gain)
      store_history <<- as.logical(store_history)
      max_leaves <<- max_leaves

      root_ <<- NULL
      feature_names_ <<- NULL
      n_features_ <<- 0
      history_ <<- list()
      leaves_ <<- 0
    },

    fit = function(X, y) {
      "Fit the AID regression tree.
      Returns: self (invisibly)"
      v <- aid_validate_input(X, y)
      X <- v$X
      y <- v$y
      feature_names_ <<- v$feature_names
      n_features_ <<- ncol(X)

      history_ <<- list()
      leaves_ <<- 0

      total_sse <- aid_sse_from_stats(sum(y), sum(y * y), length(y))
      node <- AIDNode$new()
      node$depth <- 0
      node$n_samples <- length(y)
      node$sse <- total_sse
      node$mean <- mean(y)

      root_ <<- grow_node(node, X, y, depth = 0)
      invisible(.self)
    },

    can_split = function(node, depth) {
      if (depth >= Q) return(FALSE)
      if (node$n_samples < M) return(FALSE)
      if (leaves_ >= max_leaves) return(FALSE)
      TRUE
    },

    find_best_split = function(X, y, parent_sse, min_child_size) {
      # Vectorized per-feature split scan after sorting
      n <- nrow(X); p <- ncol(X)
      if (n < 2 * min_child_size) return(NULL)

      best <- NULL

      for (j in seq_len(p)) {
        x <- X[, j]
        ord <- order(x)  # stable enough here
        x_sorted <- x[ord]
        y_sorted <- y[ord]

        diff_mask <- diff(x_sorted) != 0
        if (!any(diff_mask)) next

        split_pos <- which(diff_mask)  # cut between pos and pos+1

        left_n <- split_pos + 1
        right_n <- n - left_n
        ok <- left_n >= min_child_size & right_n >= min_child_size
        split_pos <- split_pos[ok]
        if (length(split_pos) == 0) next

        csum_y <- cumsum(y_sorted)
        csum_y2 <- cumsum(y_sorted * y_sorted)
        total_sum <- csum_y[length(csum_y)]
        total_sum2 <- csum_y2[length(csum_y2)]

        left_sum <- csum_y[split_pos]
        left_sum2 <- csum_y2[split_pos]
        right_sum <- total_sum - left_sum
        right_sum2 <- total_sum2 - left_sum2

        left_n2 <- split_pos + 1
        right_n2 <- n - left_n2

        left_sse <- left_sum2 - (left_sum * left_sum) / left_n2
        right_sse <- right_sum2 - (right_sum * right_sum) / right_n2
        gains <- parent_sse - (left_sse + right_sse)

        within <- left_sse + right_sse
        denom <- within / max(n - 2, 1)
        f_stats <- ifelse(denom > 0, gains / denom, 0)

        k <- which.max(gains)
        if (gains[k] <= 0) next

        pos <- split_pos[k]
        threshold <- (x_sorted[pos] + x_sorted[pos + 1]) / 2.0

        cand <- list(
          feature_idx = j,
          split_value = threshold,
          gain = gains[k],
          f_stat = f_stats[k],
          left_idx = ord[1:pos],
          right_idx = ord[(pos + 1):n]
        )

        if (is.null(best) || cand$gain > best$gain) best <- cand
      }

      best
    },

    grow_node = function(node, X_sub, y_sub, depth) {
      if (!can_split(node, depth)) {
        leaves_ <<- leaves_ + 1
        node$is_leaf <- TRUE
        return(node)
      }

      cand <- find_best_split(X_sub, y_sub, parent_sse = node$sse, min_child_size = R)
      if (is.null(cand) || cand$gain <= min_gain) {
        leaves_ <<- leaves_ + 1
        node$is_leaf <- TRUE
        return(node)
      }

      left_y <- y_sub[cand$left_idx]
      right_y <- y_sub[cand$right_idx]
      if (length(left_y) < R || length(right_y) < R) {
        leaves_ <<- leaves_ + 1
        node$is_leaf <- TRUE
        return(node)
      }

      # Fill split info
      node$is_leaf <- FALSE
      node$split_feature_idx <- cand$feature_idx
      node$split_value <- cand$split_value
      node$gain <- cand$gain
      node$f_stat <- cand$f_stat

      if (store_history) {
        history_[[length(history_) + 1]] <<- list(
          depth = depth,
          feature_idx = cand$feature_idx,
          feature_name = feature_names_[cand$feature_idx],
          split_value = cand$split_value,
          gain = cand$gain,
          f_stat = cand$f_stat,
          n_left = length(left_y),
          n_right = length(right_y)
        )
      }

      # Create children
      left_node <- AIDNode$new()
      left_node$depth <- depth + 1
      left_node$n_samples <- length(left_y)
      left_node$mean <- mean(left_y)
      left_node$sse <- aid_sse_vec(left_y)

      right_node <- AIDNode$new()
      right_node$depth <- depth + 1
      right_node$n_samples <- length(right_y)
      right_node$mean <- mean(right_y)
      right_node$sse <- aid_sse_vec(right_y)

      node$left <- grow_node(left_node, X_sub[cand$left_idx, , drop = FALSE], left_y, depth + 1)
      node$right <- grow_node(right_node, X_sub[cand$right_idx, , drop = FALSE], right_y, depth + 1)

      node
    },

    predict_one = function(node, row) {
      while (TRUE) {
        if (isTRUE(node$is_leaf) || is.na(node$split_feature_idx) || is.na(node$split_value)) {
          return(node$mean)
        }
        if (row[node$split_feature_idx] <= node$split_value) {
          node <- node$left
        } else {
          node <- node$right
        }
        if (is.null(node)) return(NA_real_)
      }
    },

    predict = function(X) {
      "Predict numeric targets."
      if (is.null(root_)) stop("Model not fitted")

      if (is.data.frame(X)) X <- as.matrix(X)
      X <- as.matrix(X)
      storage.mode(X) <- "double"

      apply(X, 1, function(row) predict_one(root_, row))
    },

    score_rmse = function(X, y) {
      "Compute RMSE."
      y <- as.numeric(y)
      pred <- predict(X)
      sqrt(mean((y - pred)^2))
    },

    summary = function() {
      "Return a short summary string."
      if (is.null(root_)) return("AIDRegressor(not fitted)")
      n_splits <- if (store_history) length(history_) else NA_integer_
      comp <- aid_training_complexity(n = root_$n_samples, p = n_features_, Q = Q, max_leaves = max_leaves)

      paste0(
        "AIDRegressor(R=", R, ", M=", M, ", Q=", Q, ", min_gain=", min_gain, ")\n",
        "Root: n=", root_$n_samples, ", mean=", sprintf("%.4f", root_$mean), ", sse=", sprintf("%.4f", root_$sse), "\n",
        "Leaves: ", leaves_, "\n",
        "Splits stored: ", ifelse(is.na(n_splits), "N/A", n_splits), "\n",
        "Complexity (train): ", comp$time, "\n",
        "Complexity (memory): ", comp$memory
      )
    },

    print_tree = function(max_depth = NULL) {
      "Print tree structure "
      if (is.null(root_)) { cat("Model not fitted\n"); return(invisible(NULL)) }

      print_node <- function(node, depth = 0) {
        if (!is.null(max_depth) && depth > max_depth) return()

        indent <- paste(rep("  ", depth), collapse = "")
        if (isTRUE(node$is_leaf)) {
          cat(sprintf("%sLeaf: mean=%.4f, sse=%.4f, n=%d\n",
                      indent, node$mean, node$sse, node$n_samples))
        } else {
          fname <- feature_names_[node$split_feature_idx]
          cat(sprintf("%s%s <= %.6f (gain=%.4f, f=%.4f, n=%d)\n",
                      indent, fname, node$split_value, node$gain, node$f_stat, node$n_samples))
          print_node(node$left, depth + 1)
          print_node(node$right, depth + 1)
        }
      }

      print_node(root_, 0)
      invisible(NULL)
    },

    to_list = function(node = NULL) {
      if (is.null(node)) node <- root_
      if (is.null(node)) return(NULL)

      list(
        depth = node$depth,
        n = node$n_samples,
        sse = node$sse,
        mean = node$mean,
        feature_idx = node$split_feature_idx,
        feature_name = if (!is.na(node$split_feature_idx)) feature_names_[node$split_feature_idx] else NA,
        split_value = node$split_value,
        gain = node$gain,
        f_stat = node$f_stat,
        left = if (!is.null(node$left)) to_list(node$left) else NULL,
        right = if (!is.null(node$right)) to_list(node$right) else NULL,
        is_leaf = node$is_leaf
      )
    }
  )
)

# ============================================================================
# AID Testing Suite 
# ============================================================================
# Required libraries 
suppressWarnings({
  if (!require(datasets, quietly = TRUE)) {}
  if (!require(caret, quietly = TRUE)) {}
  if (!require(ggplot2, quietly = TRUE)) {}
  if (!require(gridExtra, quietly = TRUE)) {}
})

AIDTester <- setRefClass("AIDTester",
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
      "Load regression datasets for AID"
      datasets <- list()

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

      # 3) Boston if MASS available (sometimes): medv regression
      if (require(MASS, quietly = TRUE)) {
        data(Boston)
        datasets$boston_medv <- list(
          X = Boston[, setdiff(colnames(Boston), "medv")],
          y = Boston$medv,
          description = sprintf("Boston->medv (%d samples, %d features)", nrow(Boston), ncol(Boston) - 1)
        )
      }

      # 4) Synthetic linear (your earlier benchmark spirit)
      set.seed(0)
      n <- 20000; p <- 10
      Xs <- matrix(rnorm(n * p), nrow = n, ncol = p)
      ys <- 2 * Xs[, 1] + rnorm(n)
      datasets$synthetic_linear <- list(
        X = Xs, y = ys,
        description = sprintf("Synthetic linear (%d samples, %d features)", nrow(Xs), ncol(Xs))
      )

      # 5) Synthetic non-linear (your earlier demo)
      set.seed(0)
      Xn <- cbind(runif(800, -2, 2), runif(800, -2, 2))
      yn <- ifelse(Xn[, 1] > 0, 1, 0) + 0.4 * sin(2 * Xn[, 2]) + rnorm(800, sd = 0.2)
      datasets$synthetic_nonlinear <- list(
        X = Xn, y = yn,
        description = sprintf("Synthetic non-linear (%d samples, %d features)", nrow(Xn), ncol(Xn))
      )

      cat(sprintf("Loaded %d datasets\n", length(datasets)))
      datasets
    },

    split_data = function(X, y, test_size = 0.3, seed = 42) {
      set.seed(seed)
      n <- nrow(as.matrix(X))
      train_idx <- sample(seq_len(n), size = floor((1 - test_size) * n))
      test_idx <- setdiff(seq_len(n), train_idx)
      list(
        X_train = X[train_idx, , drop = FALSE],
        X_test = X[test_idx, , drop = FALSE],
        y_train = y[train_idx],
        y_test = y[test_idx]
      )
    },

    test_basic_functionality = function(datasets) {
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("BASIC FUNCTIONALITY TEST (REGRESSION)\n")
      cat(strrep("=", 80), "\n")

      test_results <- list()

      for (name in names(datasets)) {
        d <- datasets[[name]]
        cat(sprintf("\n%s: %s\n", toupper(name), d$description))
        cat(strrep("-", 60), "\n")

        tryCatch({
          sp <- split_data(d$X, d$y, test_size = 0.3, seed = 42)

          model <- model_class$new(R = 15, M = 30, Q = 6, min_gain = 1e-3, store_history = TRUE)

          t0 <- Sys.time()
          model$fit(sp$X_train, sp$y_train)
          fit_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

          t1 <- Sys.time()
          pred <- model$predict(sp$X_test)
          predict_time <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

          rmse <- sqrt(mean((sp$y_test - pred)^2))
          mae <- mean(abs(sp$y_test - pred))
          r2 <- 1 - sum((sp$y_test - pred)^2) / sum((sp$y_test - mean(sp$y_test))^2)

          test_results[[name]] <- list(
            rmse = rmse, mae = mae, r2 = r2,
            fit_time = fit_time,
            predict_time = predict_time,
            n_splits = length(model$history_),
            success = TRUE
          )

          cat(sprintf("✓ RMSE:            %.4f\n", rmse))
          cat(sprintf("✓ MAE:             %.4f\n", mae))
          cat(sprintf("✓ R²:              %.4f\n", r2))
          cat(sprintf("✓ Fit Time:        %.4f s\n", fit_time))
          cat(sprintf("✓ Predict Time:    %.4f s\n", predict_time))
          cat(sprintf("✓ Splits stored:   %d\n", length(model$history_)))

        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          test_results[[name]] <- list(success = FALSE, error = e$message)
        })
      }

      results$basic_functionality <<- test_results
      test_results
    },

    test_cross_validation = function(datasets, cv = 5) {
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

          folds <- createFolds(y, k = cv, list = TRUE)

          rmse_scores <- numeric(cv)
          r2_scores <- numeric(cv)

          for (fold in seq_len(cv)) {
            test_idx <- folds[[fold]]
            train_idx <- setdiff(seq_len(n), test_idx)

            X_train <- X[train_idx, , drop = FALSE]
            X_test <- X[test_idx, , drop = FALSE]
            y_train <- y[train_idx]
            y_test <- y[test_idx]

            model <- model_class$new(R = 15, M = 30, Q = 6, min_gain = 1e-3, store_history = FALSE)
            model$fit(X_train, y_train)
            pred <- model$predict(X_test)

            rmse_scores[fold] <- sqrt(mean((y_test - pred)^2))
            r2_scores[fold] <- 1 - sum((y_test - pred)^2) / sum((y_test - mean(y_test))^2)

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
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("PARAMETER SENSITIVITY TEST (AID)\n")
      cat(strrep("=", 80), "\n")

      param_configs <- list(
        list(R = 5,  M = 10, Q = 3, min_gain = 0.0),
        list(R = 10, M = 20, Q = 4, min_gain = 1e-4),
        list(R = 15, M = 30, Q = 6, min_gain = 1e-3),
        list(R = 30, M = 60, Q = 8, min_gain = 1e-3),
        list(R = 50, M = 100, Q = 10, min_gain = 1e-2)
      )

      param_results <- list()

      for (name in names(datasets)) {
        d <- datasets[[name]]
        cat(sprintf("\n%s\n", toupper(name)))
        cat(strrep("-", 60), "\n")

        sp <- split_data(d$X, d$y, test_size = 0.3, seed = 42)
        X_train <- as.matrix(sp$X_train)
        X_test <- as.matrix(sp$X_test)
        y_train <- as.numeric(sp$y_train)
        y_test <- as.numeric(sp$y_test)

        config_results <- list()

        for (i in seq_along(param_configs)) {
          params <- param_configs[[i]]
          tryCatch({
            model <- do.call(model_class$new, c(params, list(store_history = FALSE)))
            model$fit(X_train, y_train)
            pred <- model$predict(X_test)

            rmse <- sqrt(mean((y_test - pred)^2))
            r2 <- 1 - sum((y_test - pred)^2) / sum((y_test - mean(y_test))^2)

            config_results[[i]] <- list(params = params, rmse = rmse, r2 = r2)

            cat(sprintf("  Config %d: R=%d M=%d Q=%d min_gain=%g -> RMSE=%.4f | R²=%.4f\n",
                        i, params$R, params$M, params$Q, params$min_gain, rmse, r2))
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
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("COMPARISON WITH RPART (REGRESSION TREE)\n")
      cat(strrep("=", 80), "\n")

      if (!require(rpart, quietly = TRUE)) {
        cat("rpart package not available\n")
        return(NULL)
      }

      comp_results <- list()

      for (name in names(datasets)) {
        d <- datasets[[name]]
        cat(sprintf("\n%s\n", toupper(name)))
        cat(strrep("-", 60), "\n")

        tryCatch({
          sp <- split_data(d$X, d$y, test_size = 0.3, seed = 42)

          # AID
          t0 <- Sys.time()
          aid_model <- model_class$new(R = 15, M = 30, Q = 6, min_gain = 1e-3, store_history = TRUE)
          aid_model$fit(sp$X_train, sp$y_train)
          aid_fit <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

          t1 <- Sys.time()
          aid_pred <- aid_model$predict(sp$X_test)
          aid_pred_t <- as.numeric(difftime(Sys.time(), t1, units = "secs"))

          aid_rmse <- sqrt(mean((as.numeric(sp$y_test) - aid_pred)^2))
          aid_r2 <- 1 - sum((sp$y_test - aid_pred)^2) / sum((sp$y_test - mean(sp$y_test))^2)

          # rpart
          df_train <- data.frame(sp$X_train, y = as.numeric(sp$y_train))
          df_test <- data.frame(sp$X_test)

          t2 <- Sys.time()
          rp <- rpart::rpart(y ~ ., data = df_train,
                             control = rpart::rpart.control(minsplit = 30, minbucket = 15, cp = 0))
          rp_fit <- as.numeric(difftime(Sys.time(), t2, units = "secs"))

          t3 <- Sys.time()
          rp_pred <- predict(rp, df_test)
          rp_pred_t <- as.numeric(difftime(Sys.time(), t3, units = "secs"))

          rp_rmse <- sqrt(mean((as.numeric(sp$y_test) - rp_pred)^2))
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
          model <- model_class$new(R = 2, M = 2, Q = 3, min_gain = 0, store_history = FALSE)
          model$fit(td$X, td$y)
          pred <- model$predict(td$X)
          rmse <- sqrt(mean((td$y - pred)^2))

          cat(sprintf("✓ SUCCESS: RMSE = %.4f\n", rmse))
          edge_results[[test_name]] <- list(success = TRUE, rmse = rmse)

        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          edge_results[[test_name]] <- list(success = FALSE, error = e$message)
        })
      }

      results$edge_cases <<- edge_results
      edge_results
    },

    visualize_results = function() {
      cat("\n", strrep("=", 80), "\n", sep = "")
      cat("GENERATING VISUALIZATIONS\n")
      cat(strrep("=", 80), "\n")

      plots <- list()

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
            geom_bar(stat = "identity", alpha = 0.85) +
            labs(title = "AID — RMSE Across Datasets", x = "Dataset", y = "RMSE") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))

          p2 <- ggplot(df, aes(x = dataset, y = r2)) +
            geom_bar(stat = "identity", alpha = 0.85) +
            labs(title = "AID — R² Across Datasets", x = "Dataset", y = "R²") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))

          p3 <- ggplot(df, aes(x = dataset, y = fit_time)) +
            geom_bar(stat = "identity", alpha = 0.85) +
            labs(title = "AID — Fit Time", x = "Dataset", y = "Seconds") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))

          p4 <- ggplot(df, aes(x = dataset, y = predict_time)) +
            geom_bar(stat = "identity", alpha = 0.85) +
            labs(title = "AID — Predict Time", x = "Dataset", y = "Seconds") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))

          plots <- list(p1, p2, p3, p4)

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
            report <- c(report, sprintf("  RMSE:         %.4f", r$rmse))
            report <- c(report, sprintf("  MAE:          %.4f", r$mae))
            report <- c(report, sprintf("  R²:           %.4f", r$r2))
            report <- c(report, sprintf("  Fit Time:     %.4f s", r$fit_time))
            report <- c(report, sprintf("  Predict Time: %.4f s", r$predict_time))
            report <- c(report, sprintf("  Splits stored:%d", r$n_splits))
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

# ============================================================================
# MAIN: run complete AID test suite
# ============================================================================
run_aid_complete_test_suite <- function() {
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

  tester
}

# ============================================================================
# EXAMPLE USAGE (interactive)
# ============================================================================
if (interactive()) {
  set.seed(0)

  # Your earlier benchmark example
  n <- 20000; p <- 10
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  y <- 2 * X[, 1] + rnorm(n)

  model <- AIDRegressor$new(R = 15, M = 30, Q = 6, min_gain = 1e-3, store_history = TRUE)
  t <- system.time(model$fit(X, y))
  print(t)
  cat(model$summary(), "\n")

  # Simple non-linear example
  X2 <- cbind(runif(800, -2, 2), runif(800, -2, 2))
  y2 <- ifelse(X2[, 1] > 0, 1, 0) + 0.4 * sin(2 * X2[, 2]) + rnorm(800, sd = 0.2)

  m2 <- AIDRegressor$new(R = 10, M = 20, Q = 4, min_gain = 1e-3, store_history = TRUE)
  m2$fit(X2, y2)
  pred <- m2$predict(X2)
  cat("RMSE:", sqrt(mean((y2 - pred)^2)), "\n")

  plot(y2, pred, pch = 16, cex = 0.6,
       main = "AID (R) — synthétique (train)",
       xlab = "y (true)", ylab = "y (pred)")
  abline(0, 1, col = "red")

  # Full test suite
  # tester <- run_aid_complete_test_suite()
}

