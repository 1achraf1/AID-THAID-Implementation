# ============================================================
# AIDRegressor — structured like THAID, algorithm = AID
# Morgan & Sonquist (1963) — SSE reduction
# ============================================================

AIDNode <- setRefClass(
  "AIDNode",
  fields = list(
    depth = "numeric",
    n = "numeric",
    sse = "numeric",
    mean = "numeric",
    feature = "numeric",
    threshold = "numeric",
    gain = "numeric",
    f_stat = "numeric",
    left = "ANY",
    right = "ANY"
  ),
  methods = list(
    is_leaf = function() {
      is.null(left) || is.null(right) || is.na(feature)
    }
  )
)

AIDRegressor <- setRefClass(
  "AIDRegressor",
  fields = list(
    R = "numeric",
    M = "numeric",
    Q = "numeric",
    min_gain = "numeric",
    store_history = "logical",
    max_leaves = "numeric",

    root_ = "ANY",
    history_ = "list",
    n_leaves_ = "numeric"
  ),

  methods = list(

    initialize = function(R=5, M=10, Q=5, min_gain=0,
                          store_history=FALSE, max_leaves=Inf) {
      .self$R <- R
      .self$M <- M
      .self$Q <- Q
      .self$min_gain <- min_gain
      .self$store_history <- store_history
      .self$max_leaves <- max_leaves

      root_ <<- NULL
      history_ <<- list()
      n_leaves_ <<- 0
    },

    fit = function(X, y) {
      if (is.data.frame(X)) X <- as.matrix(X)
      storage.mode(X) <- "double"
      y <- as.numeric(y)

      total_sse <- sum((y - mean(y))^2)

      root_ <<- AIDNode$new(
        depth = 0,
        n = length(y),
        sse = total_sse,
        mean = mean(y),
        feature = NA,
        threshold = NA,
        gain = 0,
        f_stat = NA,
        left = NULL,
        right = NULL
      )

      history_ <<- list()
      n_leaves_ <<- 0

      root_ <<- grow_node(root_, X, y)
      invisible(.self)
    },

    can_split = function(node) {
      if (node$depth >= Q) return(FALSE)
      if (node$n < M) return(FALSE)
      if (n_leaves_ >= max_leaves) return(FALSE)
      TRUE
    },

    grow_node = function(node, X, y) {

      if (!can_split(node)) {
        n_leaves_ <<- n_leaves_ + 1
        return(node)
      }

      cand <- find_best_split(X, y, node$sse, R)

      if (is.null(cand) || cand$gain <= min_gain) {
        n_leaves_ <<- n_leaves_ + 1
        return(node)
      }

      left_y <- y[cand$left_idx]
      right_y <- y[cand$right_idx]

      node$feature <- cand$feature
      node$threshold <- cand$threshold
      node$gain <- cand$gain
      node$f_stat <- cand$f_stat

      if (store_history) {
        history_[[length(history_) + 1]] <<- list(
          depth = node$depth,
          feature = cand$feature,
          threshold = cand$threshold,
          gain = cand$gain,
          f_stat = cand$f_stat
        )
      }

      node$left <- AIDNode$new(
        depth = node$depth + 1,
        n = length(left_y),
        sse = sum((left_y - mean(left_y))^2),
        mean = mean(left_y)
      )

      node$right <- AIDNode$new(
        depth = node$depth + 1,
        n = length(right_y),
        sse = sum((right_y - mean(right_y))^2),
        mean = mean(right_y)
      )

      node$left <- grow_node(node$left, X[cand$left_idx, , drop=FALSE], left_y)
      node$right <- grow_node(node$right, X[cand$right_idx, , drop=FALSE], right_y)

      node
    },

    predict = function(X) {
      if (is.data.frame(X)) X <- as.matrix(X)
      apply(X, 1, function(row) predict_one(root_, row))
    },

    print_tree = function(max_depth=NULL) {
      print_node <- function(node) {
        if (!is.null(max_depth) && node$depth > max_depth) return()
        indent <- paste(rep("  ", node$depth), collapse = "")

        if (node$is_leaf()) {
          cat(sprintf("%sLeaf(mean=%.3f, n=%d)\n",
                      indent, node$mean, node$n))
        } else {
          cat(sprintf("%sX%d <= %.3f (gain=%.3f)\n",
                      indent, node$feature, node$threshold, node$gain))
          print_node(node$left)
          print_node(node$right)
        }
      }
      print_node(root_)
    },

    summary = function() {
      cat("AIDRegressor\n")
      cat(sprintf("R=%d | M=%d | Q=%d | min_gain=%.4f\n", R, M, Q, min_gain))
      cat(sprintf("Leaves: %d | Stored splits: %d\n",
                  n_leaves_, length(history_)))
    }
  )
)

# ================= helpers =================

predict_one <- function(node, row) {
  while (TRUE) {
    if (node$is_leaf()) return(node$mean)
    if (row[node$feature] <= node$threshold) node <- node$left
    else node <- node$right
  }
}

