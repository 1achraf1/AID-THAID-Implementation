# ============================================================================
# AID (Automatic Interaction Detection) Regressor in R — SSE / least squares
# ============================================================================
# Key fixes vs your previous version:
# - Correct split routing: left/right decided with split_value (midpoint), not with X_sorted[best_idx]
# - Faster optimal split scan using cumulative sums (O(n) after sorting)
# - Stop condition includes feasibility of two leaves of size >= min_samples_leaf
# - Optional deterministic feature subsampling (max_features) via set.seed(random_state) at fit()
# ============================================================================

AIDNode <- setRefClass("AIDNode",
  fields = list(
    prediction = "numeric",
    sse = "numeric",
    n_samples = "numeric",
    split_feature_idx = "numeric",
    split_value = "numeric",
    split_gain = "numeric",
    f_statistic = "numeric",
    left = "ANY",
    right = "ANY",
    is_leaf = "logical",
    depth = "numeric"
  ),
  methods = list(
    initialize = function() {
      prediction <<- 0.0
      sse <<- 0.0
      n_samples <<- 0
      split_feature_idx <<- NA_integer_
      split_value <<- NA_real_
      split_gain <<- 0.0
      f_statistic <<- NA_real_
      left <<- NULL
      right <<- NULL
      is_leaf <<- TRUE
      depth <<- 0
    }
  )
)

AIDRegressor <- setRefClass("AIDRegressor",
  fields = list(
    min_samples_split = "numeric",
    min_samples_leaf = "numeric",
    max_depth = "numeric",
    min_split_gain = "numeric",
    max_features = "numeric",
    store_history = "logical",
    max_leaves = "numeric",
    random_state = "numeric",

    root_ = "ANY",
    n_features_ = "numeric",
    feature_names_ = "character",
    X_ = "matrix",
    y_ = "numeric",
    history_ = "list",
    n_leaves_ = "numeric"
  ),

  methods = list(
    initialize = function(min_samples_split = 10,
                          min_samples_leaf = 5,
                          max_depth = 5,
                          min_split_gain = 0.0,
                          max_features = NA,
                          store_history = FALSE,
                          max_leaves = Inf,
                          random_state = 0) {
      min_samples_split <<- as.numeric(min_samples_split)
      min_samples_leaf <<- as.numeric(min_samples_leaf)
      max_depth <<- as.numeric(max_depth)
      min_split_gain <<- as.numeric(min_split_gain)
      max_features <<- max_features
      store_history <<- store_history
      max_leaves <<- as.numeric(max_leaves)
      random_state <<- as.numeric(random_state)

      root_ <<- NULL
      n_features_ <<- 0
      feature_names_ <<- character(0)
      X_ <<- matrix(numeric(0), nrow = 0, ncol = 0)
      y_ <<- numeric(0)
      history_ <<- list()
      n_leaves_ <<- 0
    },

    validate_input = function(X, y) {
      if (is.data.frame(X)) {
        feature_names_ <<- colnames(X)
        X <- as.matrix(X)
      } else {
        X <- as.matrix(X)
        feature_names_ <<- paste0("X", seq_len(ncol(X)))
      }

      storage.mode(X) <- "double"
      y <- as.numeric(y)

      if (nrow(X) != length(y)) {
        stop(sprintf("Shape mismatch: X has %d samples, y has %d", nrow(X), length(y)))
      }

      if (any(is.na(X)) || any(is.infinite(X))) stop("X contains NA or Inf")
      if (any(is.na(y)) || any(is.infinite(y))) stop("y contains NA or Inf")

      list(X = X, y = y)
    },

    fit = function(X, y) {
      result <- validate_input(X, y)
      X <- result$X
      y <- result$y

      X_ <<- X
      y_ <<- y
      n_features_ <<- ncol(X_)

      if (is.na(max_features)) {
        max_features <<- n_features_
      } else {
        max_features <<- min(as.numeric(max_features), n_features_)
      }

      set.seed(as.integer(random_state))

      sorted_indices_ <<- lapply(seq_len(n_features_), function(j) {
        order(X_[, j], method = "radix")
      })

      history_ <<- list()
      n_leaves_ <<- 0

      indices <- seq_len(length(y_))
      root_ <<- build_tree(indices, depth = 0)

      invisible(.self)
    },

    find_numeric_split = function(indices, feature_idx, parent_sse) {
    
      global_ord <- sorted_indices_[[feature_idx]]
      
      in_node <- logical(nrow(X_))
      in_node[indices] <- TRUE
      
      idx_sorted_global <- global_ord[in_node[global_ord]]
      
      
      x_sorted <- X_[idx_sorted_global, feature_idx]
      y_sorted <- y_[idx_sorted_global]
      
      n <- length(y_sorted)
      if (n < 2 * min_samples_leaf) return(NULL)

      diff_mask <- x_sorted[-n] != x_sorted[-1]
      
      if (!any(diff_mask)) return(NULL)
      split_pos <- which(diff_mask)  

      left_n <- split_pos
      right_n <- n - split_pos
      valid <- (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
      split_pos <- split_pos[valid]
      if (length(split_pos) == 0) return(NULL)

      # cumulative sums for fast SSE
      csum <- cumsum(y_sorted)
      csum2 <- cumsum(y_sorted * y_sorted)
      total_sum <- csum[n]
      total_sum2 <- csum2[n]

      left_sum <- csum[split_pos]
      left_sum2 <- csum2[split_pos]
      right_sum <- total_sum - left_sum
      right_sum2 <- total_sum2 - left_sum2

      left_n_f <- as.numeric(split_pos)
      right_n_f <- as.numeric(n - split_pos)

      left_sse <- left_sum2 - (left_sum * left_sum) / left_n_f
      right_sse <- right_sum2 - (right_sum * right_sum) / right_n_f

      within_sse <- left_sse + right_sse
      gains <- parent_sse - within_sse

      best_i <- which.max(gains)
      best_gain <- gains[best_i]
      if (is.na(best_gain) || best_gain <= 0) return(NULL)

      k <- split_pos[best_i]
      split_value <- (x_sorted[k] + x_sorted[k + 1]) / 2.0

      denom <- within_sse[best_i] / max(n - 2, 1)
      f_stat <- if (denom > 0) best_gain / denom else Inf

      left_idx <- idx_sorted_global[1:k]
      right_idx <- idx_sorted_global[(k + 1):n]

      list(
        split_value = as.numeric(split_value),
        gain = as.numeric(best_gain),
        f_stat = as.numeric(f_stat),
        left_idx = left_idx,
        right_idx = right_idx
      )
    },

    should_stop = function(node, depth) {
      if (depth >= max_depth) return(TRUE)
      if (node$n_samples < min_samples_split) return(TRUE)
      if (n_leaves_ >= max_leaves) return(TRUE)
      if (node$n_samples < 2 * min_samples_leaf) return(TRUE)
      FALSE
    },

    build_tree = function(indices, depth) {
      node <- AIDNode$new()
      node$depth <- depth
      node$n_samples <- length(indices)

      y_subset <- y_[indices]
      node$prediction <- mean(y_subset)
      # SSE using centered sum-of-squares
      node$sse <- sum((y_subset - node$prediction)^2)

      if (should_stop(node, depth)) {
        n_leaves_ <<- n_leaves_ + 1
        return(node)
      }

      best_split <- find_best_split(indices, node$sse)

      if (is.null(best_split) || best_split$gain <= min_split_gain) {
        n_leaves_ <<- n_leaves_ + 1
        return(node)
      }

      left_idx <- best_split$left_idx
      right_idx <- best_split$right_idx

      if (length(left_idx) < min_samples_leaf || length(right_idx) < min_samples_leaf) {
        n_leaves_ <<- n_leaves_ + 1
        return(node)
      }

      node$is_leaf <- FALSE
      node$split_feature_idx <- best_split$feature_idx
      node$split_value <- best_split$split_value
      node$split_gain <- best_split$gain
      node$f_statistic <- best_split$f_stat

      if (store_history) {
        history_[[length(history_) + 1]] <<- list(
          depth = depth,
          feature_idx = best_split$feature_idx,
          split_value = best_split$split_value,
          gain = best_split$gain,
          f_statistic = best_split$f_stat
        )
      }

      node$left <- build_tree(left_idx, depth + 1)
      node$right <- build_tree(right_idx, depth + 1)
      node
    },

    find_best_split = function(indices, parent_sse) {
      if (length(indices) < 2 * min_samples_leaf) return(NULL)

      # choose subset of features (optional)
      features_to_try <- if (max_features < n_features_) {
        sample(seq_len(n_features_), size = as.integer(max_features), replace = FALSE)
      } else {
        seq_len(n_features_)
      }

      best_gain <- -Inf
      best_result <- NULL

      for (feature_idx in features_to_try) {
        result <- find_numeric_split(indices, feature_idx, parent_sse)
        if (!is.null(result) && result$gain > best_gain) {
          best_gain <- result$gain
          best_result <- c(list(feature_idx = feature_idx), result)
        }
      }

      best_result
    },

    find_numeric_split = function(indices, feature_idx, parent_sse) {
      # sort indices by feature value
      x <- X_[indices, feature_idx]
      ord <- order(x, method = "radix")
      x_sorted <- x[ord]
      y_sorted <- y_[indices][ord]
      idx_sorted_global <- indices[ord]

      n <- length(y_sorted)
      if (n < 2 * min_samples_leaf) return(NULL)

      # candidate split positions where x changes (between k and k+1)
      diff_mask <- x_sorted[-n] != x_sorted[-1]
      if (!any(diff_mask)) return(NULL)
      split_pos <- which(diff_mask)  # k in 1..n-1

      # leaf size constraints
      left_n <- split_pos
      right_n <- n - split_pos
      valid <- (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
      split_pos <- split_pos[valid]
      if (length(split_pos) == 0) return(NULL)

      # cumulative sums for fast SSE
      csum <- cumsum(y_sorted)
      csum2 <- cumsum(y_sorted * y_sorted)
      total_sum <- csum[n]
      total_sum2 <- csum2[n]

      left_sum <- csum[split_pos]
      left_sum2 <- csum2[split_pos]
      right_sum <- total_sum - left_sum
      right_sum2 <- total_sum2 - left_sum2

      left_n_f <- as.numeric(split_pos)
      right_n_f <- as.numeric(n - split_pos)

      left_sse <- left_sum2 - (left_sum * left_sum) / left_n_f
      right_sse <- right_sum2 - (right_sum * right_sum) / right_n_f

      within_sse <- left_sse + right_sse
      gains <- parent_sse - within_sse

      best_i <- which.max(gains)
      best_gain <- gains[best_i]
      if (is.na(best_gain) || best_gain <= 0) return(NULL)

      k <- split_pos[best_i]
      split_value <- (x_sorted[k] + x_sorted[k + 1]) / 2.0

      denom <- within_sse[best_i] / max(n - 2, 1)
      f_stat <- if (denom > 0) best_gain / denom else Inf

      left_idx <- idx_sorted_global[1:k]
      right_idx <- idx_sorted_global[(k + 1):n]

      list(
        split_value = as.numeric(split_value),
        gain = as.numeric(best_gain),
        f_stat = as.numeric(f_stat),
        left_idx = left_idx,
        right_idx = right_idx
      )
    },

    predict = function(X) {
      if (is.null(root_)) stop("Model not fitted")
      if (is.data.frame(X)) X <- as.matrix(X)
      X <- as.matrix(X)
      storage.mode(X) <- "double"

      preds <- numeric(nrow(X))
      preds <- predict_recursive(root_, X, seq_len(nrow(X)), preds)
      preds
    },

    predict_recursive = function(node, X, row_idx, results) {
      if (length(row_idx) == 0) return(results)

      if (node$is_leaf) {
        results[row_idx] <- node$prediction
        return(results)
      }

      col <- node$split_feature_idx
      thr <- node$split_value

      xcol <- X[row_idx, col]
      goes_left <- xcol <= thr

      results <- predict_recursive(node$left, X, row_idx[goes_left], results)
      results <- predict_recursive(node$right, X, row_idx[!goes_left], results)
      results
    },

    mse = function(X, y) {
      y <- as.numeric(y)
      y_pred <- predict(X)
      mean((y - y_pred)^2)
    },

    score = function(X, y) {
      y <- as.numeric(y)
      y_pred <- predict(X)
      ss_res <- sum((y - y_pred)^2)
      ss_tot <- sum((y - mean(y))^2)
      if (ss_tot > 0) 1 - (ss_res / ss_tot) else 0.0
    },

    print_tree = function(max_depth_print = NA) {
      if (is.null(root_)) {
        cat("Model not fitted\n")
        return(invisible(NULL))
      }

      print_node <- function(node) {
        if (!is.na(max_depth_print) && node$depth > max_depth_print) return()

        indent <- paste(rep("  ", node$depth), collapse = "")
        if (node$is_leaf) {
          cat(sprintf("%sLeaf: pred=%.4f, sse=%.4f, n=%d\n", indent, node$prediction, node$sse, node$n_samples))
        } else {
          fname <- feature_names_[node$split_feature_idx]
          cat(sprintf("%s%s <= %.6f (gain=%.6f, F=%.2f, n=%d)\n",
                      indent, fname, node$split_value, node$split_gain, node$f_statistic, node$n_samples))
          print_node(node$left)
          print_node(node$right)
        }
      }

      print_node(root_)
      invisible(NULL)
    },

    summary = function() {
      cat("AID Regressor\n")
      cat(sprintf("min_samples_split=%d | min_samples_leaf=%d | max_depth=%d\n",
                  as.integer(min_samples_split), as.integer(min_samples_leaf), as.integer(max_depth)))
      cat(sprintf("min_split_gain=%.6g | max_features=%s | random_state=%d\n",
                  min_split_gain,
                  ifelse(max_features >= n_features_, "all", as.character(as.integer(max_features))),
                  as.integer(random_state)))
      cat(sprintf("Number of leaves: %d | Stored splits: %d\n", as.integer(n_leaves_), length(history_)))
      invisible(NULL)
    }
  )
)
