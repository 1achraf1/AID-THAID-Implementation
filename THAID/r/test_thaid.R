# ============================================================================
# THAID Testing Suite
# Comprehensive testing framework for THAID classifier
# ============================================================================

# Source the main THAID implementation
# source("thaid.R")

# Required libraries
library(datasets)
library(caret)
library(ggplot2)
library(gridExtra)

THAIDTester <- setRefClass("THAIDTester",
  fields = list(
    model_class = "ANY",
    results = "list"
  ),
  
  methods = list(
    initialize = function(model_class) {
      "Initialize the tester with a model class"
      model_class <<- model_class
      results <<- list()
    },
    
    load_datasets = function() {
      "Load various datasets for testing"
      
      datasets <- list()
      
      # 1. Iris Dataset
      data(iris)
      datasets$iris <- list(
        X = iris[, 1:4],
        y = iris$Species,
        description = sprintf("Iris (%d samples, %d features, %d classes)", 
                            nrow(iris), 4, length(unique(iris$Species)))
      )
      
      # 2. mtcars Dataset (convert to classification)
      data(mtcars)
      datasets$mtcars <- list(
        X = mtcars[, c("mpg", "disp", "hp", "wt", "qsec")],
        y = factor(mtcars$am, labels = c("automatic", "manual")),
        description = sprintf("mtcars (%d samples, %d features, %d classes)", 
                            nrow(mtcars), 5, 2)
      )
      
      # 3. Titanic Dataset (aggregated)
      data(Titanic)
      titanic_df <- as.data.frame(Titanic)
      titanic_expanded <- titanic_df[rep(seq_len(nrow(titanic_df)), titanic_df$Freq), ]
      titanic_expanded <- titanic_expanded[, -5]  # Remove Freq column
      
      if (nrow(titanic_expanded) > 0) {
        X_titanic <- data.frame(
          Class = as.numeric(titanic_expanded$Class),
          Sex = as.numeric(titanic_expanded$Sex),
          Age = as.numeric(titanic_expanded$Age)
        )
        
        datasets$titanic <- list(
          X = X_titanic,
          y = titanic_expanded$Survived,
          description = sprintf("Titanic (%d samples, %d features, %d classes)", 
                              nrow(X_titanic), 3, length(unique(titanic_expanded$Survived)))
        )
      }
      
      # 4. Breast Cancer Wisconsin (if available)
      if (require(mlbench, quietly = TRUE)) {
        data(BreastCancer)
        bc <- BreastCancer[complete.cases(BreastCancer), ]
        
        X_bc <- bc[, 2:10]
        for (i in 1:ncol(X_bc)) {
          X_bc[, i] <- as.numeric(as.character(X_bc[, i]))
        }
        
        datasets$breast_cancer <- list(
          X = X_bc,
          y = bc$Class,
          description = sprintf("Breast Cancer (%d samples, %d features, %d classes)", 
                              nrow(X_bc), ncol(X_bc), length(unique(bc$Class)))
        )
      }
      
      # 5. Wine Dataset (if available)
      if (require(rattle.data, quietly = TRUE)) {
        data(wine)
        datasets$wine <- list(
          X = wine[, -1],
          y = factor(wine$Type),
          description = sprintf("Wine (%d samples, %d features, %d classes)", 
                              nrow(wine), ncol(wine) - 1, length(unique(wine$Type)))
        )
      }
      
      cat(sprintf("Loaded %d datasets\n", length(datasets)))
      return(datasets)
    },
    
    split_data = function(X, y, test_size = 0.3, seed = 42) {
      "Split data into train and test sets"
      
      set.seed(seed)
      n <- nrow(X)
      train_idx <- sample(1:n, size = floor((1 - test_size) * n))
      test_idx <- setdiff(1:n, train_idx)
      
      list(
        X_train = X[train_idx, , drop = FALSE],
        X_test = X[test_idx, , drop = FALSE],
        y_train = y[train_idx],
        y_test = y[test_idx]
      )
    },
    
    test_basic_functionality = function(datasets) {
      "Test basic fit/predict functionality"
      
      cat("\n")
      cat(strrep("=", 80), "\n")
      cat("BASIC FUNCTIONALITY TEST\n")
      cat(strrep("=", 80), "\n")
      
      test_results <- list()
      
      for (name in names(datasets)) {
        data <- datasets[[name]]
        cat(sprintf("\n%s: %s\n", toupper(name), data$description))
        cat(strrep("-", 60), "\n")
        
        tryCatch({
          split <- split_data(data$X, data$y, test_size = 0.3, seed = 42)
          
          model <- model_class$new()
          start_time <- Sys.time()
          model$fit(split$X_train, split$y_train)
          fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
          
          start_time <- Sys.time()
          y_pred <- model$predict(split$X_test)
          predict_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
          
          train_acc <- model$score(split$X_train, split$y_train)
          test_acc <- mean(y_pred == split$y_test)
          
          test_results[[name]] <- list(
            train_accuracy = train_acc,
            test_accuracy = test_acc,
            fit_time = fit_time,
            predict_time = predict_time,
            success = TRUE
          )
          
          cat(sprintf("✓ Training Accuracy: %.4f\n", train_acc))
          cat(sprintf("✓ Testing Accuracy:  %.4f\n", test_acc))
          cat(sprintf("✓ Fit Time:          %.4f s\n", fit_time))
          cat(sprintf("✓ Predict Time:      %.4f s\n", predict_time))
          
        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          test_results[[name]] <- list(success = FALSE, error = e$message)
        })
      }
      
      results$basic_functionality <<- test_results
      return(test_results)
    },
    
    test_cross_validation = function(datasets, cv = 5) {
      "Test with cross-validation for robustness"
      
      cat("\n")
      cat(strrep("=", 80), "\n")
      cat("CROSS-VALIDATION TEST\n")
      cat(strrep("=", 80), "\n")
      
      cv_results <- list()
      
      for (name in names(datasets)) {
        data <- datasets[[name]]
        cat(sprintf("\n%s: %s\n", toupper(name), data$description))
        cat(strrep("-", 60), "\n")
        
        tryCatch({
          set.seed(42)
          n <- nrow(data$X)
          folds <- createFolds(data$y, k = cv, list = TRUE)
          
          cv_scores <- numeric(cv)
          
          for (fold in 1:cv) {
            test_idx <- folds[[fold]]
            train_idx <- setdiff(1:n, test_idx)
            
            X_train <- data$X[train_idx, , drop = FALSE]
            X_test <- data$X[test_idx, , drop = FALSE]
            y_train <- data$y[train_idx]
            y_test <- data$y[test_idx]
            
            model <- model_class$new()
            model$fit(X_train, y_train)
            score <- model$score(X_test, y_test)
            cv_scores[fold] <- score
            
            cat(sprintf("  Fold %d: %.4f\n", fold, score))
          }
          
          mean_score <- mean(cv_scores)
          std_score <- sd(cv_scores)
          
          cv_results[[name]] <- list(
            cv_scores = cv_scores,
            mean_accuracy = mean_score,
            std_accuracy = std_score,
            success = TRUE
          )
          
          cat(sprintf("\n✓ Mean CV Accuracy: %.4f (+/- %.4f)\n", mean_score, std_score))
          
        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          cv_results[[name]] <- list(success = FALSE, error = e$message)
        })
      }
      
      results$cross_validation <<- cv_results
      return(cv_results)
    },
    
    test_parameter_sensitivity = function(datasets) {
      "Test sensitivity to hyperparameters"
      
      cat("\n")
      cat(strrep("=", 80), "\n")
      cat("PARAMETER SENSITIVITY TEST\n")
      cat(strrep("=", 80), "\n")
      
      param_configs <- list(
        list(min_samples_split = 10, min_samples_leaf = 1, max_depth = NULL),
        list(min_samples_split = 20, min_samples_leaf = 5, max_depth = NULL),
        list(min_samples_split = 20, min_samples_leaf = 1, max_depth = 5),
        list(min_samples_split = 50, min_samples_leaf = 10, max_depth = 3)
      )
      
      param_results <- list()
      
      for (name in names(datasets)) {
        data <- datasets[[name]]
        cat(sprintf("\n%s\n", toupper(name)))
        cat(strrep("-", 60), "\n")
        
        split <- split_data(data$X, data$y, test_size = 0.3, seed = 42)
        config_results <- list()
        
        for (i in seq_along(param_configs)) {
          params <- param_configs[[i]]
          
          tryCatch({
            model <- do.call(model_class$new, params)
            model$fit(split$X_train, split$y_train)
            test_acc <- model$score(split$X_test, split$y_test)
            
            config_results[[i]] <- list(
              params = params,
              accuracy = test_acc
            )
            
            cat(sprintf("  Config %d: min_split=%d, min_leaf=%d, max_depth=%s -> %.4f\n",
                       i, params$min_samples_split, params$min_samples_leaf,
                       ifelse(is.null(params$max_depth), "NULL", params$max_depth),
                       test_acc))
            
          }, error = function(e) {
            cat(sprintf("  Config %d: ERROR - %s\n", i, e$message))
          })
        }
        
        param_results[[name]] <- config_results
      }
      
      results$parameter_sensitivity <<- param_results
      return(param_results)
    },
    
    test_comparison_with_rpart = function(datasets) {
      "Compare THAID with rpart decision tree"
      
      cat("\n")
      cat(strrep("=", 80), "\n")
      cat("COMPARISON WITH RPART DECISION TREE\n")
      cat(strrep("=", 80), "\n")
      
      if (!require(rpart, quietly = TRUE)) {
        cat("rpart package not available\n")
        return(NULL)
      }
      
      comp_results <- list()
      
      for (name in names(datasets)) {
        data <- datasets[[name]]
        cat(sprintf("\n%s\n", toupper(name)))
        cat(strrep("-", 60), "\n")
        
        tryCatch({
          split <- split_data(data$X, data$y, test_size = 0.3, seed = 42)
          
          # THAID
          start_time <- Sys.time()
          thaid_model <- model_class$new(min_samples_split = 20, min_samples_leaf = 1)
          thaid_model$fit(split$X_train, split$y_train)
          thaid_fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
          
          start_time <- Sys.time()
          thaid_pred <- thaid_model$predict(split$X_test)
          thaid_predict_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
          thaid_acc <- mean(thaid_pred == split$y_test)
          
          # rpart
          train_data <- cbind(split$X_train, y = split$y_train)
          start_time <- Sys.time()
          rpart_model <- rpart(y ~ ., data = train_data, 
                              control = rpart.control(minsplit = 20, minbucket = 1))
          rpart_fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
          
          start_time <- Sys.time()
          rpart_pred <- predict(rpart_model, split$X_test, type = "class")
          rpart_predict_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
          rpart_acc <- mean(rpart_pred == split$y_test)
          
          comp_results[[name]] <- list(
            thaid_accuracy = thaid_acc,
            rpart_accuracy = rpart_acc,
            thaid_fit_time = thaid_fit_time,
            rpart_fit_time = rpart_fit_time,
            thaid_predict_time = thaid_predict_time,
            rpart_predict_time = rpart_predict_time,
            success = TRUE
          )
          
          cat("THAID:\n")
          cat(sprintf("  Accuracy:      %.4f\n", thaid_acc))
          cat(sprintf("  Fit Time:      %.4f s\n", thaid_fit_time))
          cat(sprintf("  Predict Time:  %.4f s\n", thaid_predict_time))
          
          cat("\nrpart DecisionTree:\n")
          cat(sprintf("  Accuracy:      %.4f\n", rpart_acc))
          cat(sprintf("  Fit Time:      %.4f s\n", rpart_fit_time))
          cat(sprintf("  Predict Time:  %.4f s\n", rpart_predict_time))
          
          cat("\nComparison:\n")
          cat(sprintf("  Accuracy Diff: %+.4f\n", thaid_acc - rpart_acc))
          cat(sprintf("  Speed Ratio:   %.2fx\n", rpart_fit_time / thaid_fit_time))
          
        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          comp_results[[name]] <- list(success = FALSE, error = e$message)
        })
      }
      
      results$rpart_comparison <<- comp_results
      return(comp_results)
    },
    
    test_edge_cases = function(datasets) {
      "Test edge cases and robustness"
      
      cat("\n")
      cat(strrep("=", 80), "\n")
      cat("EDGE CASES TEST\n")
      cat(strrep("=", 80), "\n")
      
      edge_results <- list()
      
      name <- names(datasets)[1]
      data <- datasets[[name]]
      
      X <- data$X
      y <- data$y
      
      test_cases <- list(
        single_sample = list(X = X[1, , drop = FALSE], y = y[1]),
        two_samples = list(X = X[1:2, , drop = FALSE], y = y[1:2]),
        small_sample = list(X = X[1:10, , drop = FALSE], y = y[1:10]),
        single_feature = list(X = X[, 1, drop = FALSE], y = y)
      )
      
      for (test_name in names(test_cases)) {
        cat(sprintf("\n%s\n", toupper(test_name)))
        cat(strrep("-", 60), "\n")
        
        test_data <- test_cases[[test_name]]
        
        tryCatch({
          model <- model_class$new(min_samples_split = 2, min_samples_leaf = 1, max_depth = 3)
          model$fit(test_data$X, test_data$y)
          pred <- model$predict(test_data$X)
          acc <- mean(pred == test_data$y)
          
          cat(sprintf("✓ SUCCESS: Accuracy = %.4f\n", acc))
          edge_results[[test_name]] <- list(success = TRUE, accuracy = acc)
          
        }, error = function(e) {
          cat(sprintf("✗ ERROR: %s\n", e$message))
          edge_results[[test_name]] <- list(success = FALSE, error = e$message)
        })
      }
      
      results$edge_cases <<- edge_results
      return(edge_results)
    },
    
    visualize_results = function() {
      "Create visualizations of test results"
      
      cat("\n")
      cat(strrep("=", 80), "\n")
      cat("GENERATING VISUALIZATIONS\n")
      cat(strrep("=", 80), "\n")
      
      plots <- list()
      
      if ("basic_functionality" %in% names(results)) {
        data <- results$basic_functionality
        successful <- names(data)[sapply(data, function(x) x$success)]
        
        if (length(successful) > 0) {
          df <- data.frame(
            dataset = rep(successful, 2),
            type = rep(c("Train", "Test"), each = length(successful)),
            accuracy = c(
              sapply(successful, function(n) data[[n]]$train_accuracy),
              sapply(successful, function(n) data[[n]]$test_accuracy)
            )
          )
          
          p1 <- ggplot(df, aes(x = dataset, y = accuracy, fill = type)) +
            geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
            labs(title = "THAID Performance Across Datasets",
                 x = "Dataset", y = "Accuracy") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))
          
          plots[[1]] <- p1
          
          df_time <- data.frame(
            dataset = rep(successful, 2),
            type = rep(c("Fit", "Predict"), each = length(successful)),
            time = c(
              sapply(successful, function(n) data[[n]]$fit_time),
              sapply(successful, function(n) data[[n]]$predict_time)
            )
          )
          
          p2 <- ggplot(df_time, aes(x = dataset, y = time, fill = type)) +
            geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
            labs(title = "THAID Execution Time",
                 x = "Dataset", y = "Time (seconds)") +
            theme_minimal() +
            theme(axis.text.x = element_text(angle = 45, hjust = 1))
          
          plots[[2]] <- p2
        }
      }
      
      if ("cross_validation" %in% names(results)) {
        data <- results$cross_validation
        successful <- names(data)[sapply(data, function(x) x$success)]
        
        if (length(successful) > 0) {
          df_cv <- do.call(rbind, lapply(successful, function(n) {
            data.frame(
              dataset = n,
              fold = 1:length(data[[n]]$cv_scores),
              accuracy = data[[n]]$cv_scores
            )
          }))
          
          p3 <- ggplot(df_cv, aes(x = fold, y = accuracy, color = dataset, group = dataset)) +
            geom_line(size = 1.2) +
            geom_point(size = 3) +
            labs(title = "Cross-Validation Scores Across Datasets",
                 x = "Fold", y = "Accuracy") +
            theme_minimal()
          
          plots[[3]] <- p3
        }
      }
      
      if (length(plots) > 0) {
        tryCatch({
          if (length(plots) >= 2) {
            png("thaid_performance.png", width = 1400, height = 500, res = 100)
            grid.arrange(plots[[1]], plots[[2]], ncol = 2)
            dev.off()
            cat("✓ Saved: thaid_performance.png\n")
          }
          
          if (length(plots) >= 3) {
            png("thaid_cv_scores.png", width = 1000, height = 600, res = 100)
            print(plots[[3]])
            dev.off()
            cat("✓ Saved: thaid_cv_scores.png\n")
          }
        }, error = function(e) {
          cat(sprintf("Warning: Could not save plots: %s\n", e$message))
        })
      }
    },
    
    generate_report = function() {
      "Generate comprehensive test report"
      
      cat("\n")
      cat(strrep("=", 80), "\n")
      cat("COMPREHENSIVE TEST REPORT\n")
      cat(strrep("=", 80), "\n")
      
      report <- c()
      report <- c(report, "\nTHAID ALGORITHM TEST REPORT")
      report <- c(report, strrep("=", 80))
      report <- c(report, sprintf("\nGenerated: %s\n", Sys.time()))
      
      if ("basic_functionality" %in% names(results)) {
        report <- c(report, "\n1. BASIC FUNCTIONALITY")
        report <- c(report, strrep("-", 80))
        data <- results$basic_functionality
        success_count <- sum(sapply(data, function(x) x$success))
        report <- c(report, sprintf("Datasets Tested: %d", length(data)))
        report <- c(report, sprintf("Successful: %d/%d", success_count, length(data)))
        
        for (name in names(data)) {
          result <- data[[name]]
          if (result$success) {
            report <- c(report, sprintf("\n%s:", name))
            report <- c(report, sprintf("  Train Accuracy: %.4f", result$train_accuracy))
            report <- c(report, sprintf("  Test Accuracy:  %.4f", result$test_accuracy))
            report <- c(report, sprintf("  Fit Time:       %.4f s", result$fit_time))
          }
        }
      }
      
      if ("cross_validation" %in% names(results)) {
        report <- c(report, "\n\n2. CROSS-VALIDATION RESULTS")
        report <- c(report, strrep("-", 80))
        data <- results$cross_validation
        
        for (name in names(data)) {
          result <- data[[name]]
          if (result$success) {
            report <- c(report, sprintf("\n%s:", name))
            report <- c(report, sprintf("  Mean Accuracy: %.4f", result$mean_accuracy))
            report <- c(report, sprintf("  Std Accuracy:  %.4f", result$std_accuracy))
          }
        }
      }
      
      if ("rpart_comparison" %in% names(results)) {
        report <- c(report, "\n\n3. COMPARISON WITH RPART")
        report <- c(report, strrep("-", 80))
        data <- results$rpart_comparison
        
        for (name in names(data)) {
          result <- data[[name]]
          if (result$success) {
            report <- c(report, sprintf("\n%s:", name))
            report <- c(report, sprintf("  THAID Accuracy: %.4f", result$thaid_accuracy))
            report <- c(report, sprintf("  rpart Accuracy: %.4f", result$rpart_accuracy))
            report <- c(report, sprintf("  Difference:     %+.4f", 
                                       result$thaid_accuracy - result$rpart_accuracy))
          }
        }
      }
      
      report_text <- paste(report, collapse = "\n")
      cat(report_text)
      cat("\n")
      
      tryCatch({
        writeLines(report_text, "thaid_test_report.txt")
        cat("\n✓ Saved: thaid_test_report.txt\n")
      }, error = function(e) {
        cat(sprintf("Warning: Could not save report: %s\n", e$message))
      })
      
      return(report_text)
    }
  )
)

# ============================================================================
# MAIN FUNCTION TO RUN COMPLETE TEST SUITE
# ============================================================================

run_complete_test_suite <- function() {
  "Run all tests on THAID classifier"
  
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat("THAID ALGORITHM - COMPLETE TEST SUITE\n")
  cat(strrep("=", 80), "\n")
  
  # Initialize tester
  tester <- THAIDTester$new(THAID)
  
  # Load datasets
  cat("\nLoading datasets...\n")
  datasets <- tester$load_datasets()
  cat(sprintf("✓ Loaded %d datasets\n", length(datasets)))
  
  # Run all tests
  tester$test_basic_functionality(datasets)
  tester$test_cross_validation(datasets, cv = 5)
  tester$test_parameter_sensitivity(datasets)
  tester$test_comparison_with_rpart(datasets)
  tester$test_edge_cases(datasets)
  
  # Generate visualizations and report
  tester$visualize_results()
  tester$generate_report()
  
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat("TEST SUITE COMPLETED\n")
  cat(strrep("=", 80), "\n")
  
  return(tester)
}

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if (interactive()) {
  # Simple example with Iris
  cat("\n")
  cat(strrep("=", 80), "\n")
  cat("SIMPLE EXAMPLE: IRIS DATASET\n")
  cat(strrep("=", 80), "\n\n")
  
  data(iris)
  model <- THAID$new(min_samples_split = 20, min_samples_leaf = 5, max_depth = 3)
  model$fit(iris[, 1:4], iris$Species)
  
  cat("Tree Structure:\n")
  cat(strrep("-", 80), "\n")
  model$print_tree()
  
  cat("\n\nPredictions:\n")
  cat(strrep("-", 80), "\n")
  predictions <- model$predict(iris[, 1:4])
  accuracy <- model$score(iris[, 1:4], iris$Species)
  cat(sprintf("Accuracy: %.4f\n", accuracy))
  
  probas <- model$predict_proba(iris[, 1:4])
  cat(sprintf("\nFirst 5 probability predictions:\n"))
  print(head(probas, 5))

  #to do the full test
  #tester <- run_complete_test_suite()
}
