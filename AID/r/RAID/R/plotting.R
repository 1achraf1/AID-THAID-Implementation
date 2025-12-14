#' Visualisation de l'arbre AID
#' @export
#' @importFrom ggplot2 geom_segment

plot_aid_tree <- function(model) {
  node_positions <- list()
  leaf_counter <- 0

  assign_positions <- function(node, depth = 0) {
    if (is_leaf(node)) {
      leaf_counter <<- leaf_counter + 1
      x <- leaf_counter
      node_positions[[length(node_positions) + 1]] <<- data.frame(
        id = paste0("n", leaf_counter),
        depth = depth,
        x = x,
        y = -depth,
        label = if (is_leaf(node)) sprintf("leaf\nn=%d\nmean=%.2f", node$n, node$mean) else ""
      )
      return(x)
    }
    lx <- assign_positions(node$left, depth + 1)
    rx <- assign_positions(node$right, depth + 1)
    x <- (lx + rx) / 2
    node_positions[[length(node_positions) + 1]] <<- data.frame(
      id = paste0("n", leaf_counter + 1),
      depth = depth,
      x = x,
      y = -depth,
      label = sprintf("x%d <= %.2f\nn=%d\nmean=%.2f", node$feature, node$threshold, node$n, node$mean)
    )
    x
  }

  assign_positions(model$root)
  nodes_df <- do.call(rbind, node_positions)

  edges <- data.frame(x = numeric(), xend = numeric(), y = numeric(), yend = numeric())
  build_edges <- function(node) {
    if (is_leaf(node)) return()
    parent_pos <- nodes_df[which(nodes_df$label == sprintf("x%d <= %.2f\nn=%d\nmean=%.2f", node$feature, node$threshold, node$n, node$mean))[1], ]
    left_pos <- nodes_df[which(nodes_df$label == if (is_leaf(node$left)) sprintf("leaf\nn=%d\nmean=%.2f", node$left$n, node$left$mean) else sprintf("x%d <= %.2f\nn=%d\nmean=%.2f", node$left$feature, node$left$threshold, node$left$n, node$left$mean))[1], ]
    right_pos <- nodes_df[which(nodes_df$label == if (is_leaf(node$right)) sprintf("leaf\nn=%d\nmean=%.2f", node$right$n, node$right$mean) else sprintf("x%d <= %.2f\nn=%d\nmean=%.2f", node$right$feature, node$right$threshold, node$right$n, node$right$mean))[1], ]
    edges <<- rbind(edges, data.frame(x = parent_pos$x, xend = left_pos$x, y = parent_pos$y, yend = left_pos$y))
    edges <<- rbind(edges, data.frame(x = parent_pos$x, xend = right_pos$x, y = parent_pos$y, yend = right_pos$y))
    build_edges(node$left); build_edges(node$right)
  }
  build_edges(model$root)

  ggplot(nodes_df, aes(x = x, y = y)) +
    geom_segment(data = edges, aes(x = x, xend = xend, y = y, yend = yend), color = "grey60") +
    geom_point(size = 4, color = "#2a9d8f") +
    geom_point(size = 3, color = "#1d3557") +
    geom_text(aes(label = label), vjust = -0.8, size = 3) +
    theme_minimal() +
    labs(title = "Arbre AID", x = NULL, y = NULL)
}

#' Visualiser le meilleur split pour un nœud
#' @export
plot_aid_split <- function(X, y, node) {
  df <- data.frame(x = X[, node$feature], y = y)
  ggplot(df, aes(x = x, y = y)) +
    geom_point(alpha = 0.6, color = "#1d3557") +
    geom_vline(xintercept = node$threshold, linetype = "dashed", color = "#e76f51") +
    theme_minimal() +
    labs(title = "Meilleur split", x = sprintf("x%d", node$feature), y = "y")
}
