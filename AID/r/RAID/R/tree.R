#' Construction d'un nœud AID (structure S3)
new_node <- function(depth, n, sse, mean, parent_value = NA_real_,
                     feature = NA_integer_, threshold = NA_real_,
                     gain = 0, f_stat = NA_real_, left = NULL, right = NULL) {
  structure(
    list(
      depth = depth,
      n = n,
      sse = sse,
      mean = mean,
      parent_value = parent_value,
      feature = feature,
      threshold = threshold,
      gain = gain,
      f_stat = f_stat,
      left = left,
      right = right
    ),
    class = "aid_node"
  )
}

is_leaf <- function(node) {
  is.null(node$left) && is.null(node$right)
}

predict_node <- function(node, row) {
  if (is_leaf(node)) return(node$mean)
  branch <- if (row[node$feature] <= node$threshold) node$left else node$right
  if (is.null(branch)) return(node$mean)
  predict_node(branch, row)
}

node_to_list <- function(node) {
  if (is.null(node)) return(NULL)
  list(
    depth = node$depth,
    n = node$n,
    sse = node$sse,
    mean = node$mean,
    feature = node$feature,
    threshold = node$threshold,
    gain = node$gain,
    f_stat = node$f_stat,
    left = node_to_list(node$left),
    right = node_to_list(node$right)
  )
}
