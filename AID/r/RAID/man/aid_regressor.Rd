\name{aid_regressor}
\alias{aid_regressor}
\title{Automatic Interaction Detection for Regression}
\description{
Implémentation fidèle de l’algorithme Automatic Interaction Detection (AID)
proposé par Morgan et Sonquist (1963) pour la régression.
}
\usage{
aid_regressor(X, y, R = 5, M = 10, Q = 5, min_gain = 0, max_leaves = Inf)
}
\arguments{
  \item{X}{Matrice ou data.frame des variables explicatives.}
  \item{y}{Vecteur numérique de la variable cible.}
  \item{R}{Taille minimale de chaque sous-classe fille.}
  \item{M}{Taille minimale d’un nœud pour tenter un split.}
  \item{Q}{Profondeur maximale de l’arbre.}
  \item{min_gain}{Gain minimal de SSE pour accepter un split.}
  \item{max_leaves}{Nombre maximal de feuilles.}
}
\value{
Un objet de classe \code{aid_model}.
}
\references{
Morgan, J. N., & Sonquist, J. A. (1963).
Problems in the analysis of survey data, and a proposal.
Journal of the American Statistical Association.
}
