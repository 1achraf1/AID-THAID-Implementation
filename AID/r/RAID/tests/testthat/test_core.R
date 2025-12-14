library(testthat)
library(RAID)

test_that("fit and predict run", {
  set.seed(0)
  X <- matrix(rnorm(200), ncol = 2)
  y <- ifelse(X[, 1] > 0, 1, 0) + 0.2 * X[, 2]
  model <- aid_regressor(X, y, R = 5, M = 10, Q = 3)
  preds <- predict_aid(model, X[1:3, , drop = FALSE])
  expect_length(preds, 3)
  expect_true(all(is.finite(preds)))
})
