#' AID pour régression (Morgan & Sonquist, 1963)
#'
#' @param X matrice ou data.frame des variables prédictives.
#' @param y vecteur numérique de la variable cible.
#' @param R taille minimale de chaque enfant.
#' @param M taille minimale pour tenter un split.
#' @param Q profondeur maximale (racine = 0).
#' @param min_gain réduction minimale de variance pour accepter un split.
#' @return un objet de classe \code{aid_model}
#' @export
aid_regressor <- function(X, y, R = 5, M = 10, Q = 5, min_gain = 0, max_leaves = Inf) {
  data <- ensure_matrix(X, y)
  X <- data$X; y <- data$y

  total_sum <- sum(y)
  total_sum2 <- sum(y * y)
  total_sse <- sse_from_stats(total_sum, total_sum2, length(y))

  root <- new_node(depth = 0, n = length(y), sse = total_sse, mean = mean(y))
  history <- list()
  leaves <- 0

  can_split <- function(node, depth) {
    if (depth >= Q) return(FALSE)
    if (node$n < M) return(FALSE)
    if (leaves >= max_leaves) return(FALSE)
    TRUE
  }

  grow <- function(node, X_sub, y_sub, depth) {
    if (!can_split(node, depth)) {
      leaves <<- leaves + 1
      return(node)
    }

    cand <- find_best_split(X_sub, y_sub, parent_sse = node$sse, min_child = R)
    if (is.null(cand) || cand$gain <= min_gain) {
      leaves <<- leaves + 1
      return(node)
    }
    if (length(cand$left_idx) < R || length(cand$right_idx) < R) {
      leaves <<- leaves + 1
      return(node)
    }

    left_y <- y_sub[cand$left_idx]
    right_y <- y_sub[cand$right_idx]
    left_sse <- sse_from_stats(sum(left_y), sum(left_y * left_y), length(left_y))
    right_sse <- sse_from_stats(sum(right_y), sum(right_y * right_y), length(right_y))

    node$feature <- cand$feature
    node$threshold <- cand$threshold
    node$gain <- cand$gain
    node$f_stat <- cand$f_stat

    history[[length(history) + 1]] <<- list(
      depth = depth,
      feature = cand$feature,
      threshold = cand$threshold,
      gain = cand$gain,
      f_stat = cand$f_stat,
      n_left = length(cand$left_idx),
      n_right = length(cand$right_idx)
    )

    node$left <- new_node(depth = depth + 1, n = length(left_y), sse = left_sse,
                          mean = mean(left_y), parent_value = node$mean)
    node$right <- new_node(depth = depth + 1, n = length(right_y), sse = right_sse,
                           mean = mean(right_y), parent_value = node$mean)

    node$left <- grow(node$left, X_sub[cand$left_idx, , drop = FALSE], left_y, depth + 1)
    node$right <- grow(node$right, X_sub[cand$right_idx, , drop = FALSE], right_y, depth + 1)
    node
  }

  root <- grow(root, X, y, depth = 0)
  structure(
    list(root = root, history = history, params = list(R = R, M = M, Q = Q, min_gain = min_gain)),
    class = "aid_model"
  )
}

#' Prédiction pour un modèle AID
#' @export
predict_aid <- function(object, newdata, ...) {
  X <- as.matrix(newdata)
  apply(X, 1, function(row) predict_node(object$root, row))
}
