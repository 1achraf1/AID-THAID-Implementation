pyAID: Automatic Interaction Detection in Python
===============================================

Ce package propose une implémentation fidèle et optimisée de l’algorithme AID (Morgan & Sonquist, 1963) pour la régression. L’objectif est pédagogique : illustrer l’approche historique de partitionnement binaire basée sur la réduction de variance, avec les paramètres canoniques R, M et Q.

Caractéristiques
----------------
- AIDRegressor conforme à la logique originale (réduction de variance, F-stat informatif).
- Paramètres R (taille minimale des enfants), M (taille minimale pour tenter un split), Q (profondeur maximale).
- Implémentation vectorisée NumPy et calculs agrégés pour limiter les copies.
- Export JSON, visualisations légères (structure d’arbre et splits).
- Tests unitaires avec pytest et exemples reproductibles (iris, données synthétiques).

Installation locale
-------------------
```bash
cd pyAID
pip install -e .
```

Utilisation rapide
------------------
```python
import numpy as np
from pyAID import AIDRegressor

X = np.random.normal(size=(200, 2))
y = (X[:, 0] > 0).astype(float) + 0.2 * X[:, 1]

model = AIDRegressor(R=5, M=12, Q=4, min_gain=1e-3, store_history=True)
model.fit(X, y)
preds = model.predict(X[:3])
print(model.summary())
```

Organisation du code
--------------------
- `pyAID/aid.py` : classe principale `AIDRegressor`.
- `pyAID/utils.py` : conversions, recherche de split vectorisée.
- `pyAID/tree.py` : structure de nœud et export.
- `pyAID/plotting.py` : visualisation minimaliste.
- `tests/` : tests unitaires avec pytest.
- `examples/` : notebooks illustratifs (iris, données numériques synthétiques).

Hypothèses et choix
-------------------
- L’algorithme suit strictement la sélection du meilleur split par réduction de SSE et F-stat calculé à des fins de diagnostic (pas de CART/CHAID).
- Données supposées numériques ; les variables catégorielles peuvent être encodées en amont.
- Arrêts : profondeur maximale (Q), taille minimale parent (M) et enfants (R), gain minimal optionnel.


# Analyse de performance et justification de l’optimisation

L’implémentation a été évaluée empiriquement afin de vérifier que les choix de programmation respectent la complexité algorithmique attendue de l’algorithme AID, sans en modifier le principe.

---

## Configuration expérimentale

- **Taille des données :** \(n = 20\,000\) observations
- **Variables :** \(p = 10\) variables explicatives continues
- **Profondeur maximale :** \(Q = 6\)
- **Paramètres :** \(R = 15\), \(M = 30\)
- **Environnement :** Python 3.9, NumPy, Anaconda
- **Mesure du temps :** `python -m timeit` pour éviter les biais liés aux imports ou au profiling

---

## Temps d’exécution mesuré

- **Temps moyen :** ≈ 314 ms
- **Protocole de mesure :** 5 itérations, 3 répétitions, valeur minimale retenue

```bash
python -m timeit -n 5 -r 3 "
import numpy as np
from pyAID import AIDRegressor
rng = np.random.default_rng(0)
X = rng.normal(size=(20000, 10))
y = 2*X[:,0] + rng.normal(size=20000)
AIDRegressor(Q=6, M=30, R=15, min_gain=1e-3).fit(X, y)
"
```

Ce temps est cohérent avec la complexité théorique d’AID, qui nécessite :
- **Tri par variable,**
- **Évaluation exhaustive de tous les seuils admissibles,**
- **À chaque nœud de l’arbre.**

---

## Techniques d’optimisation utilisées

- **Un seul tri par variable et par nœud :** utilisation de `argsort`, puis évaluation de tous les seuils en une passe.
- **Sommes cumulées vectorisées :** calcul du SSE gauche/droite en \(O(1)\) par seuil, sans boucles Python.
  
  ```python
  csum_y  = np.cumsum(y_sorted)
  csum_y2 = np.cumsum(y_sorted * y_sorted)
  ```

- **Calcul du SSE à partir de statistiques agrégées :** évite le recalcul explicite des résidus.
  
  ```python
  # SSE = sum(y^2) - (sum(y))^2 / n
  SSE = sum_y2 - (sum_y * sum_y) / n
  ```

- **Aucune copie inutile des données :** `ensure_numpy` retourne des vues contiguës, et les sous-ensembles sont indexés une seule fois au moment du split.
- **Arrêts précoces stricts :** contraintes \(R\), \(M\), \(Q\), et **gain minimal** optionnel réduisent significativement le nombre de splits évalués.

---


## Complexité algorithmique de l’algorithme AID

---

L’algorithme AID (Automatic Interaction Detection) repose sur une recherche exhaustive du meilleur seuil de partition pour chaque variable explicative, en maximisant la réduction de variance (SSE).

### Complexité temporelle

Considérons :

* (n) : nombre d’observations,
* (p) : nombre de variables explicatives,
* (Q) : profondeur maximale de l’arbre.

#### Coût d’un nœud

Pour un nœud contenant (n_k) observations :

1. **Tri par variable**
   Chaque variable est triée une fois :
   [
   O(n_k \log n_k)
   ]

2. **Évaluation des seuils admissibles**
   Grâce aux sommes cumulées, tous les seuils sont évalués en une seule passe :
   [
   O(n_k)
   ]

Le coût total par variable est donc :
[
O(n_k \log n_k)
]

et pour (p) variables :
[
O(p , n_k \log n_k)
]

#### Coût total de l’arbre

Dans le pire des cas (arbre équilibré), la somme des tailles des nœuds à une profondeur donnée est (n).
Sur (Q) niveaux, la complexité temporelle est donc bornée par :

[
\boxed{
O\bigl(Q , p , n \log n\bigr)
}
]

Cette borne correspond à une recherche exhaustive des splits, conforme à la définition originale d’AID.

### Complexité mémoire

* Les données ne sont pas dupliquées : seules des vues et des index sont manipulés.
* L’arbre stocke un nombre de nœuds borné par :
  [
  O(2^{Q})
  ]
* Chaque nœud contient uniquement des statistiques agrégées (SSE, moyenne, seuil).

La complexité mémoire est donc :
[
\boxed{
O(n + 2^{Q})
}
]

### Remarques importantes

* Les contraintes (R) (taille minimale des enfants) et (M) (taille minimale du parent) réduisent fortement le nombre de splits réellement évalués en pratique.
* La profondeur maximale (Q) contrôle explicitement la complexité et empêche toute croissance incontrôlée de l’arbre.
* Cette complexité est plus élevée que celle d’arbres modernes optimisés (ex. CART avec heuristiques), mais correspond fidèlement à l’approche historique d’AID.

---

## Profilage CPU

- **Observation principale :** la majorité du temps est concentrée dans la fonction de recherche du meilleur split, cœur algorithmique d’AID.
- **Surcoûts :** aucun surcoût significatif dans la gestion de l’arbre, l’export ou la prédiction.

---

## Mémoire

- **Empreinte modérée :** absence de structures intermédiaires volumineuses.
- **Pas de stockage redondant :** données non dupliquées.
- **Structure compacte :** arbre minimal avec nœuds binaires légers.



Ressources
----------
- Morgan, J. N., & Sonquist, J. A. (1963). *Problems in the analysis of survey data, and a proposal*. Journal of the American Statistical Association, 58(302), 415‑434.
