library(testthat)

test_that("best split found on step function", {
  X <- matrix(c(0, 1, 2, 3), ncol = 1)
  y <- c(0, 0, 1, 1)
  parent_sse <- RAID:::sse_from_stats(sum(y), sum(y * y), length(y))
  cand <- RAID:::find_best_split(X, y, parent_sse = parent_sse, min_child = 1)
  expect_false(is.null(cand))
  expect_gt(cand$gain, 0)
  expect_true(cand$threshold > 1 && cand$threshold < 2)
})
