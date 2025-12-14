#' Convertit les données en matrice numérique et vérifie les dimensions
ensure_matrix <- function(X, y) {
  if (is.data.frame(X)) {
    X <- as.matrix(X)
  }
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  if (nrow(X) != length(y)) stop("X et y doivent avoir le même nombre de lignes.")
  list(X = X, y = y)
}

#' Calcul du SSE à partir des sommes agrégées
sse_from_stats <- function(sum_y, sum_y2, n) {
  if (n <= 0) return(0)
  sum_y2 - (sum_y * sum_y) / n
}

#' Recherche vectorisée du meilleur split
find_best_split <- function(X, y, parent_sse, min_child) {
  n <- nrow(X); p <- ncol(X)
  best <- NULL

  for (j in seq_len(p)) {
    x <- X[, j]
    ord <- order(x)
    x_sorted <- x[ord]
    y_sorted <- y[ord]

    diff_mask <- diff(x_sorted) != 0
    if (!any(diff_mask)) next

    split_pos <- which(diff_mask)
    split_pos <- split_pos[split_pos + 1 >= min_child & n - (split_pos + 1) >= min_child]
    if (length(split_pos) == 0) next

    csum_y <- cumsum(y_sorted)
    csum_y2 <- cumsum(y_sorted * y_sorted)
    total_sum <- csum_y[length(csum_y)]
    total_sum2 <- csum_y2[length(csum_y2)]

    left_n <- split_pos + 1
    right_n <- n - left_n

    left_sum <- csum_y[split_pos]
    left_sum2 <- csum_y2[split_pos]
    right_sum <- total_sum - left_sum
    right_sum2 <- total_sum2 - left_sum2

    left_sse <- left_sum2 - (left_sum * left_sum) / left_n
    right_sse <- right_sum2 - (right_sum * right_sum) / right_n
    gains <- parent_sse - (left_sse + right_sse)

    within <- left_sse + right_sse
    f_stats <- ifelse(within > 0, gains / (within / (n - 2)), 0)

    best_idx <- which.max(gains)
    if (gains[best_idx] <= 0) next

    pos <- split_pos[best_idx]
    threshold <- (x_sorted[pos] + x_sorted[pos + 1]) / 2

    cand <- list(
      feature = j,
      threshold = threshold,
      gain = gains[best_idx],
      f_stat = f_stats[best_idx],
      left_idx = ord[seq_len(pos)],
      right_idx = ord[(pos + 1):n]
    )
    if (is.null(best) || cand$gain > best$gain) best <- cand
  }
  best
}
