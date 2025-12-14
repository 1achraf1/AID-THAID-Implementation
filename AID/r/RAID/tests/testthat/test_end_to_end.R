library(testthat)
library(RAID)

test_that("model exports to list", {
  set.seed(1)
  X <- matrix(rnorm(300), ncol = 3)
  y <- ifelse(X[, 2] > 0, 1, 0) + 0.1 * X[, 3]
  model <- aid_regressor(X, y, R = 5, M = 12, Q = 4, min_gain = 1e-3)
  tree_list <- RAID:::node_to_list(model$root)
  expect_true(is.list(tree_list))
  expect_true("mean" %in% names(tree_list))
})
