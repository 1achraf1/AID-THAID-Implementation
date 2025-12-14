# ============================================================================
# THAID (Theta Automatic Interaction Detection) Classifier in R
# ============================================================================
# 
# A decision tree classifier that uses theta as the
# splitting criterion. Supports both numeric and categorical features.


THAIDNode <- setRefClass("THAIDNode",
  fields = list(
    prediction = "numeric",
    theta = "numeric",
    n_samples = "numeric",
    class_counts = "ANY",
    split_feature_idx = "numeric",
    split_value = "ANY",
    split_categories = "ANY",
    is_numeric = "logical",
    left = "ANY",
    right = "ANY",
    is_leaf = "logical"
  ),
  methods = list(
    initialize = function() {
      prediction <<- -1
      theta <<- 0.0
      n_samples <<- 0
      class_counts <<- NULL
      split_feature_idx <<- -1
      split_value <<- NULL
      split_categories <<- NULL
      is_numeric <<- FALSE
      left <<- NULL
      right <<- NULL
      is_leaf <<- TRUE
    }
  )
)

THAID <- setRefClass("THAID",
  fields = list(
    min_samples_split = "numeric",
    min_samples_leaf = "numeric",
    max_depth = "ANY",
    max_categories = "numeric",
    root_ = "ANY",
    n_features_ = "numeric",
    n_classes_ = "numeric",
    classes_ = "ANY",
    feature_types_ = "ANY",
    feature_names_ = "ANY",
    X_ = "ANY",
    y_ = "ANY",
    sorted_indices_ = "ANY"
  ),
  
  methods = list(
    initialize = function(min_samples_split = 20, min_samples_leaf = 1, 
                         max_depth = NULL, max_categories = 10) {
      "Initialize THAID classifier
      
      Parameters:
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        max_depth: Maximum depth of the tree (NULL for unlimited)
        max_categories: Maximum categories for exhaustive search"
      
      min_samples_split <<- min_samples_split
      min_samples_leaf <<- min_samples_leaf
      max_depth <<- max_depth
      max_categories <<- max_categories
      root_ <<- NULL
      n_features_ <<- 0
      n_classes_ <<- 0
      classes_ <<- NULL
      feature_types_ <<- NULL
      feature_names_ <<- NULL
      X_ <<- NULL
      y_ <<- NULL
      sorted_indices_ <<- NULL
    },
    
    fit = function(X, y) {
      "Fit the THAID classifier to training data
      
      Parameters:
        X: Training features (data.frame or matrix)
        y: Training labels (vector or factor)
      
      Returns:
        self (invisibly)"
      
      result <- validate_input(X, y)
      X <- result$X
      y <- result$y
      
      classes_ <<- sort(unique(y))
      y_encoded <- match(y, classes_) - 1
      n_classes_ <<- length(classes_)
      n_features_ <<- ncol(X)
      
      detect_feature_types(X)
      X_ <<- X
      y_ <<- y_encoded
      presort_numeric_features()
      
      indices <- 1:length(y)
      root_ <<- build_tree(indices, depth = 0)
      
      invisible(.self)
    },
    
    validate_input = function(X, y) {
      "Validate and prepare input data"
      
      if (is.data.frame(X)) {
        feature_names_ <<- colnames(X)
        X <- as.matrix(X)
      } else {
        X <- as.matrix(X)
        feature_names_ <<- paste0("X", 1:ncol(X))
      }
      
      y <- as.vector(y)
      
      if (nrow(X) != length(y)) {
        stop(sprintf("Shape mismatch: X has %d samples, y has %d", nrow(X), length(y)))
      }
      
      if (any(is.na(X)) || any(is.infinite(X))) {
        stop("X contains NA or Inf")
      }
      
      list(X = X, y = y)
    },
    
    detect_feature_types = function(X) {
      "Detect whether each feature is numeric (0) or categorical (1)"
      
      feature_types_ <<- integer(n_features_)
      
      for (i in 1:n_features_) {
        col <- X[, i]
        
        if (!is.numeric(col)) {
          feature_types_[i] <<- 1
          next
        }
        
        unique_vals <- unique(col)
        if (length(unique_vals) <= 10 && all(col == round(col))) {
          feature_types_[i] <<- 1
        }
      }
    },
    
    presort_numeric_features = function() {
      "Pre-sort numeric features for efficient splitting"
      
      sorted_indices_ <<- vector("list", n_features_)
      
      for (i in 1:n_features_) {
        if (feature_types_[i] == 0) {
          sorted_indices_[[i]] <<- order(X_[, i])
        } else {
          sorted_indices_[[i]] <<- NULL
        }
      }
    },
    
    build_tree = function(indices, depth) {
      "Recursively build the decision tree"
      
      node <- THAIDNode$new()
      node$n_samples <- length(indices)
      
      y_subset <- y_[indices]
      node$class_counts <- tabulate(y_subset + 1, nbins = n_classes_)
      
      node$prediction <- which.max(node$class_counts) - 1
      node$theta <- node$class_counts[node$prediction + 1] / node$n_samples
      
      if (should_stop(node, depth)) {
        return(node)
      }
      
      best_split <- find_best_split(indices)
      
      if (is.null(best_split)) {
        return(node)
      }
      
      feature_idx <- best_split$feature_idx
      split_info <- best_split$split_info
      left_idx <- best_split$left_idx
      right_idx <- best_split$right_idx
      
      if (length(left_idx) < min_samples_leaf || length(right_idx) < min_samples_leaf) {
        return(node)
      }
      
      node$is_leaf <- FALSE
      node$split_feature_idx <- feature_idx
      
      if (feature_types_[feature_idx] == 0) {
        node$is_numeric <- TRUE
        node$split_value <- split_info
      } else {
        node$is_numeric <- FALSE
        node$split_categories <- split_info
      }
      
      node$left <- build_tree(left_idx, depth + 1)
      node$right <- build_tree(right_idx, depth + 1)
      
      node
    },
    
    should_stop = function(node, depth) {
      "Check if tree building should stop at this node"
      
      if (sum(node$class_counts > 0) <= 1) return(TRUE)
      if (node$n_samples < min_samples_split) return(TRUE)
      if (!is.null(max_depth) && depth >= max_depth) return(TRUE)
      FALSE
    },
    
    find_best_split = function(indices) {
      "Find the best split across all features"
      
      best_theta <- -1.0
      best_result <- NULL
      
      for (feature_idx in 1:n_features_) {
        if (feature_types_[feature_idx] == 0) {
          result <- find_numeric_split(indices, feature_idx)
        } else {
          result <- find_categorical_split(indices, feature_idx)
        }
        
        if (!is.null(result) && result$theta > best_theta) {
          best_theta <- result$theta
          best_result <- c(list(feature_idx = feature_idx), result)
        }
      }
      
      best_result
    },
    
    find_numeric_split = function(indices, feature_idx) {
      "Find best split for a numeric feature"
      
      sorted_full <- sorted_indices_[[feature_idx]]
      mask <- sorted_full %in% indices
      relevant <- sorted_full[mask]
      
      if (length(relevant) < 2) return(NULL)
      
      X_sorted <- X_[relevant, feature_idx]
      y_sorted <- y_[relevant]
      n_total <- length(relevant)
      
      change_indices <- which(X_sorted[-length(X_sorted)] != X_sorted[-1])
      if (length(change_indices) == 0) return(NULL)
      
      right_counts <- tabulate(y_sorted + 1, nbins = n_classes_)
      left_counts <- integer(n_classes_)
      
      best_theta <- -1.0
      best_split_val <- NULL
      best_split_idx <- NULL
      
      current_idx <- 1
      for (split_idx in change_indices) {
        for (i in current_idx:(split_idx)) {
          cls <- y_sorted[i] + 1
          left_counts[cls] <- left_counts[cls] + 1
          right_counts[cls] <- right_counts[cls] - 1
        }
        
        current_idx <- split_idx + 1
        theta <- (max(left_counts) + max(right_counts)) / n_total
        
        if (theta > best_theta) {
          best_theta <- theta
          best_split_idx <- split_idx
          best_split_val <- (X_sorted[split_idx] + X_sorted[split_idx + 1]) / 2.0
        }
      }
      
      if (is.null(best_split_val)) return(NULL)
      
      left_mask <- X_sorted <= X_sorted[best_split_idx]
      left_indices <- indices[indices %in% relevant[left_mask]]
      right_indices <- indices[indices %in% relevant[!left_mask]]
      
      list(split_info = best_split_val, theta = best_theta, 
           left_idx = left_indices, right_idx = right_indices)
    },
    
    find_categorical_split = function(indices, feature_idx) {
      "Find best split for a categorical feature"
      
      X_col <- X_[indices, feature_idx]
      unique_vals <- unique(X_col)
      
      if (length(unique_vals) < 2) return(NULL)
      
      if (length(unique_vals) <= max_categories) {
        exhaustive_categorical(indices, feature_idx, unique_vals)
      } else {
        heuristic_categorical(indices, feature_idx, unique_vals)
      }
    },
    
    exhaustive_categorical = function(indices, feature_idx, unique_vals) {
      "Exhaustive search for categorical splits"
      
      X_col <- X_[indices, feature_idx]
      y_subset <- y_[indices]
      n_total <- length(indices)
      
      best_theta <- -1.0
      best_mask <- NULL
      
      max_size <- floor(length(unique_vals) / 2) + 1
      
      for (size in 1:(max_size - 1)) {
        combos <- combn(unique_vals, size, simplify = FALSE)
        
        for (combo in combos) {
          mask <- X_col %in% combo
          
          if (!any(mask) || all(mask)) next
          
          y_left <- y_subset[mask]
          y_right <- y_subset[!mask]
          
          counts_left <- tabulate(y_left + 1, nbins = n_classes_)
          counts_right <- tabulate(y_right + 1, nbins = n_classes_)
          
          theta <- (max(counts_left) + max(counts_right)) / n_total
          
          if (theta > best_theta) {
            best_theta <- theta
            best_mask <- mask
          }
        }
      }
      
      if (is.null(best_mask)) return(NULL)
      
      list(split_info = best_mask, theta = best_theta, 
           left_idx = indices[best_mask], right_idx = indices[!best_mask])
    },
    
    heuristic_categorical = function(indices, feature_idx, unique_vals) {
      "Heuristic search for categorical splits (when many categories)"
      
      X_col <- X_[indices, feature_idx]
      y_subset <- y_[indices]
      n_total <- length(indices)
      
      majority_class <- which.max(tabulate(y_subset + 1, nbins = n_classes_)) - 1
      
      scores <- sapply(unique_vals, function(cat) {
        mean(y_subset[X_col == cat] == majority_class)
      })
      sorted_cats <- unique_vals[order(-scores)]
      
      best_theta <- -1.0
      best_mask <- NULL
      
      for (i in 1:(length(sorted_cats) - 1)) {
        mask <- X_col %in% sorted_cats[1:i]
        
        if (!any(mask) || all(mask)) next
        
        y_left <- y_subset[mask]
        y_right <- y_subset[!mask]
        
        counts_left <- tabulate(y_left + 1, nbins = n_classes_)
        counts_right <- tabulate(y_right + 1, nbins = n_classes_)
        
        theta <- (max(counts_left) + max(counts_right)) / n_total
        
        if (theta > best_theta) {
          best_theta <- theta
          best_mask <- mask
        }
      }
      
      if (is.null(best_mask)) return(NULL)
      
      list(split_info = best_mask, theta = best_theta, 
           left_idx = indices[best_mask], right_idx = indices[!best_mask])
    },
    
    predict = function(X) {
      "Predict class labels for samples
      
      Parameters:
        X: Features to predict (data.frame or matrix)
      
      Returns:
        Vector of predicted class labels"
      
      if (is.null(root_)) stop("Model not fitted")
      
      if (is.data.frame(X)) X <- as.matrix(X)
      
      predictions <- integer(nrow(X))
      predictions <- predict_recursive(root_, X, 1:nrow(X), predictions)
      
      classes_[predictions + 1]
    },
    
    predict_recursive = function(node, X, indices, results) {
      "Recursively predict for samples"
      
      if (length(indices) == 0) return(results)
      
      if (node$is_leaf) {
        results[indices] <- node$prediction
        return(results)
      }
      
      X_feature <- X[indices, node$split_feature_idx]
      
      if (node$is_numeric) {
        goes_left <- X_feature <= node$split_value
      } else {
        goes_left <- X_feature %in% node$split_categories
      }
      
      results <- predict_recursive(node$left, X, indices[goes_left], results)
      results <- predict_recursive(node$right, X, indices[!goes_left], results)
      
      return(results)
    },
    
    predict_proba = function(X) {
      "Predict class probabilities for samples
      
      Parameters:
        X: Features to predict (data.frame or matrix)
      
      Returns:
        Matrix of class probabilities"
      
      if (is.null(root_)) stop("Model not fitted")
      
      if (is.data.frame(X)) X <- as.matrix(X)
      
      probas <- matrix(0, nrow = nrow(X), ncol = n_classes_)
      probas <- predict_proba_recursive(root_, X, 1:nrow(X), probas)
      
      probas
    },
    
    predict_proba_recursive = function(node, X, indices, results) {
      "Recursively predict probabilities"
      
      if (length(indices) == 0) return(results)
      
      if (node$is_leaf) {
        results[indices, ] <- matrix(node$class_counts / node$n_samples, 
                                      nrow = length(indices), 
                                      ncol = n_classes_, byrow = TRUE)
        return(results)
      }
      
      X_feature <- X[indices, node$split_feature_idx]
      
      if (node$is_numeric) {
        goes_left <- X_feature <= node$split_value
      } else {
        goes_left <- X_feature %in% node$split_categories
      }
      
      results <- predict_proba_recursive(node$left, X, indices[goes_left], results)
      results <- predict_proba_recursive(node$right, X, indices[!goes_left], results)
      
      return(results)
    },
    
    score = function(X, y) {
      "Calculate accuracy score
      
      Parameters:
        X: Features
        y: True labels
      
      Returns:
        Accuracy score"
      
      y <- as.vector(y)
      mean(predict(X) == y)
    },
    
    print_tree = function(max_depth = NULL) {
      "Print the decision tree structure
      
      Parameters:
        max_depth: Maximum depth to print (NULL for all)"
      
      if (is.null(root_)) {
        cat("Model not fitted\n")
        return()
      }
      
      print_node <- function(node, depth = 0) {
        if (!is.null(max_depth) && depth > max_depth) return()
        
        indent <- paste(rep("  ", depth), collapse = "")
        
        if (node$is_leaf) {
          cat(sprintf("%sLeaf: class=%s, theta=%.3f, n=%d\n",
                     indent, classes_[node$prediction + 1], 
                     node$theta, node$n_samples))
        } else {
          fname <- feature_names_[node$split_feature_idx]
          
          if (node$is_numeric) {
            cat(sprintf("%s%s <= %.3f (theta=%.3f, n=%d)\n",
                       indent, fname, node$split_value, 
                       node$theta, node$n_samples))
          } else {
            cats_display <- if (length(node$split_categories) > 5) {
              paste(head(node$split_categories, 5), collapse = ", ")
            } else {
              paste(node$split_categories, collapse = ", ")
            }
            cat(sprintf("%s%s in [%s] (theta=%.3f, n=%d)\n",
                       indent, fname, cats_display, 
                       node$theta, node$n_samples))
          }
          
          print_node(node$left, depth + 1)
          print_node(node$right, depth + 1)
        }
      }
      
      print_node(root_)
    }
  )
)

# Example usage:
# library(datasets)
# data(iris)
# model <- THAID$new(min_samples_split = 20, min_samples_leaf = 5, max_depth = 3)
# model$fit(iris[, 1:4], iris$Species)
# model$print_tree()
# predictions <- model$predict(iris[, 1:4])
# accuracy <- model$score(iris[, 1:4], iris$Species)
# probas <- model$predict_proba(iris[, 1:4])
