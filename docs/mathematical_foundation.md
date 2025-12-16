# Mathematical Foundations of AID and THAID

## Table of Contents
- [Introduction](#introduction)
- [Recursive Partitioning Framework](#recursive-partitioning-framework)
- [AID: Automatic Interaction Detection](#aid-automatic-interaction-detection)
- [THAID: Theta Automatic Interaction Detection](#thaid-theta-automatic-interaction-detection)
- [Computational Complexity](#computational-complexity)
- [Stopping Criteria](#stopping-criteria)
- [References](#references)

---

## Introduction

This document provides the mathematical foundations for the AID (Automatic Interaction Detection) and THAID (Theta Automatic Interaction Detection) algorithms. These foundational decision tree methods, developed in the 1960s and 1970s, pioneered the concept of recursive partitioning for multivariate analysis.

**Key Contributions:**
- **AID (Morgan & Sonquist, 1963)**: Automated the detection of interactions between predictor variables in regression problems
- **THAID (Morgan & Messenger, 1973)**: Extended recursive partitioning to categorical dependent variables using novel splitting criteria

---

## Recursive Partitioning Framework

### Feature Space Partitioning

The fundamental operation in decision tree algorithms is the recursive partitioning of the feature space. 

**Definition:** Let **X** ∈ ℝ^p be the p-dimensional feature space. The algorithm partitions this space into M distinct, non-overlapping regions (hyperrectangles):

```
R₁, R₂, ..., Rₘ
```

such that:

```
⋃ᵢ₌₁ᴹ Rᵢ = X   and   Rᵢ ∩ Rⱼ = ∅  for i ≠ j
```

### Prediction Function

For any input vector **x**, the model predicts a response ŷ based on the region into which **x** falls:

```
f(x) = ∑ᵢ₌₁ᴹ cᵢ · 𝟙(x ∈ Rᵢ)
```

where:
- **Rᵢ**: The i-th region (leaf node)
- **cᵢ**: The constant prediction for region Rᵢ
- **𝟙(·)**: Indicator function (1 if condition is true, 0 otherwise)

### Recursive Splitting Process

1. **Start** with the entire dataset as a single node
2. **Evaluate** all possible splits on all features
3. **Select** the split that optimizes the criterion (variance reduction for AID, theta/delta for THAID)
4. **Recurse** on child nodes until stopping criteria are met
5. **Assign** predictions to leaf nodes

---

## AID: Automatic Interaction Detection

AID is designed for **regression problems** where the target variable Y is continuous. The objective is to partition the feature space such that variance within each region is minimized.

### Total Sum of Squares (TSS)

For a parent node *t* containing Nₜ samples with target values {y₁, y₂, ..., yₙ}, the Total Sum of Squares measures the total variance:

```
TSSₜ = ∑ᵢ∈ₜ (yᵢ - ȳₜ)²
```

where **ȳₜ** is the mean target value in node t:

```
ȳₜ = (1/Nₜ) ∑ᵢ∈ₜ yᵢ
```

### Within-Group Sum of Squares (WSS)

When a candidate split divides node *t* into:
- **Left child** t_L with N_L samples
- **Right child** t_R with N_R samples

The Within-Group Sum of Squares is:

```
WSS(t_L, t_R) = ∑ᵢ∈ₜ_L (yᵢ - ȳₜ_L)² + ∑ⱼ∈ₜ_R (yⱼ - ȳₜ_R)²
```

### Between-Group Sum of Squares (BSS)

The reduction in variance achieved by the split:

```
BSS(t_L, t_R) = TSSₜ - WSS(t_L, t_R)
```

Expanding this:

```
BSS(t_L, t_R) = N_L · (ȳₜ_L - ȳₜ)² + N_R · (ȳₜ_R - ȳₜ)²
```

### AID Splitting Criterion

**Objective:** Find the split (feature j, threshold s) that **maximizes** the variance reduction:

```
(j*, s*) = argmax_{j,s} Δ_AID(j, s)
```

where:

```
Δ_AID(j, s) = TSSₜ - WSS(t_L, t_R)
              = BSS(t_L, t_R)
```

**Equivalently**, we can maximize the ratio:

```
R² = BSS/TSS = 1 - WSS/TSS
```

### Prediction in Leaf Nodes

For AID, the prediction at each leaf node is the **mean** of the target values:

```
ĉᵢ = (1/Nᵢ) ∑ⱼ∈Rᵢ yⱼ
```

### Example: Numerical Feature Split

Consider a feature *x₁* with sorted unique values [1.2, 2.5, 3.8, 4.1, 5.0]. The algorithm tests splits between consecutive values:

- Split 1: x₁ ≤ 1.85 → {left: [1.2], right: [2.5, 3.8, 4.1, 5.0]}
- Split 2: x₁ ≤ 3.15 → {left: [1.2, 2.5], right: [3.8, 4.1, 5.0]}
- Split 3: x₁ ≤ 3.95 → {left: [1.2, 2.5, 3.8], right: [4.1, 5.0]}
- Split 4: x₁ ≤ 4.55 → {left: [1.2, 2.5, 3.8, 4.1], right: [5.0]}

For each split, compute BSS and select the one with maximum variance reduction.

---

## THAID: Theta Automatic Interaction Detection

THAID is designed for **classification problems** where the target variable Y is categorical. Unlike entropy or Gini impurity used in modern trees (C4.5, CART), THAID offers two distinct optimization strategies.

### Criterion 1: The Theta Statistic (θ)

The Theta criterion aims to **maximize modal classification accuracy** (zero-one accuracy).

#### Definition

For a split dividing parent node *t* into left (t_L) and right (t_R) children:

```
S_θ = (C_max(t_L) + C_max(t_R)) / N_total
```

where:
- **C_max(t)**: Count of the most frequent (modal) class in node t
- **N_total**: Total number of samples in the parent node

#### Interpretation

The theta statistic measures the proportion of samples that would be correctly classified if we predict the majority class in each child node. Higher values indicate purer nodes.

#### Example

Suppose we have 100 samples with binary classes {A, B}:
- Parent: 60A, 40B
- Split creates:
  - Left child: 50A, 10B → C_max = 50
  - Right child: 10A, 30B → C_max = 30

```
S_θ = (50 + 30)/100 = 0.80
```

This split correctly classifies 80% of samples using majority class prediction.

### Criterion 2: The Delta Statistic (δ)

The Delta criterion measures the **distributional difference** between child nodes using L₁ distance.

#### Definition

Let K be the number of classes. For each child node, compute the probability distribution over classes:

```
p_L,k = N_L,k / N_L    (probability of class k in left child)
p_R,k = N_R,k / N_R    (probability of class k in right child)
```

The Delta statistic is the Manhattan distance between these distributions:

```
S_δ = ∑ₖ₌₁ᴷ |p_L,k - p_R,k|
```

#### Interpretation

A higher δ value indicates that the left and right nodes have very different class compositions, suggesting a strong interaction between the split feature and the target variable.

#### Properties

- **Range**: δ ∈ [0, 2]
- **Minimum** (δ = 0): Children have identical class distributions
- **Maximum** (δ = 2): Children have completely disjoint class distributions

#### Example

Consider 3 classes {A, B, C} with 100 total samples:
- Left child: 30A, 10B, 10C → p_L = [0.6, 0.2, 0.2]
- Right child: 10A, 20B, 20C → p_R = [0.2, 0.4, 0.4]

```
S_δ = |0.6 - 0.2| + |0.2 - 0.4| + |0.2 - 0.4|
    = 0.4 + 0.2 + 0.2
    = 0.8
```

### Comparison: Theta vs Delta

| Aspect | Theta (θ) | Delta (δ) |
|--------|-----------|-----------|
| **Focus** | Majority class dominance | Distributional separation |
| **Sensitive to** | Purity of modal class | Overall class distribution differences |
| **Best for** | Imbalanced datasets | Balanced, multi-class problems |
| **Computational cost** | Lower | Slightly higher |

### Handling Categorical Features

#### Low Cardinality (C ≤ max_categories)

For categorical features with few unique values, THAID performs **exhaustive search** of all possible binary partitions:

```
Number of splits = 2^(C-1) - 1
```

Example: Feature with categories {Red, Blue, Green}
- Split 1: {Red} vs {Blue, Green}
- Split 2: {Blue} vs {Red, Green}
- Split 3: {Green} vs {Red, Blue}
- Split 4: {Red, Blue} vs {Green}
- etc.

#### High Cardinality (C > max_categories)

For high-cardinality categorical features, exhaustive search is computationally prohibitive. THAID uses a **heuristic approach**:

**Algorithm:**
1. Identify the majority class y_maj of the current parent node
2. For each category c, calculate the conditional probability:
   ```
   P(c) = P(Y = y_maj | X = c) = count(Y=y_maj, X=c) / count(X=c)
   ```
3. Sort categories in **descending order** by P(c)
4. Treat sorted categories as ordinal and perform linear split search

**Rationale:** Categories with high P(c) are more strongly associated with the majority class, so grouping similar categories reduces computational cost while maintaining split quality.

### Prediction in Leaf Nodes

For THAID, the prediction at each leaf node is the **mode** (most frequent class):

```
ĉᵢ = argmax_k count(y = k | x ∈ Rᵢ)
```

---

## Computational Complexity

### AID Complexity

For a dataset with:
- **N** samples
- **p** features
- **d** tree depth

**Per split evaluation:**
- Sorting continuous features: O(N log N)
- Evaluating all splits: O(N)
- Total per feature: O(N log N)

**Complete tree:**
```
O(p · N · log(N) · d)
```

### THAID Complexity

**Numerical features:** Same as AID: O(N log N) per feature

**Categorical features:**
- Low cardinality (C ≤ max_cat): O(2^C · N)
- High cardinality (C > max_cat): O(C log C + C · N) using heuristic

**Worst case with many high-cardinality categorical features:**
```
O(p · 2^C_max · N · d)
```

---

## Stopping Criteria

Both AID and THAID implement multiple stopping criteria to prevent overfitting:

### 1. Minimum Samples to Split
```
N_t < min_samples_split
```
If a node has fewer than min_samples_split samples, it becomes a leaf.

### 2. Maximum Depth
```
depth(t) ≥ max_depth
```
Prevents the tree from growing too deep.

### 3. Minimum Improvement Threshold

**AID:**
```
BSS(t_L, t_R) / TSSₜ < min_improvement
```
Stop if the variance reduction is below threshold.

**THAID:**
```
S_criterion < min_score
```
Stop if the splitting criterion (theta or delta) is below threshold.

### 4. Pure Node
```
all(yᵢ = y₁) for i in node
```
For classification, stop if all samples in a node have the same class.

### 5. Single Feature Value
```
unique(X_feature) = 1
```
Stop if all samples have the same value for all remaining features.

---

## References

### Primary Sources

1. **Morgan, J. N., & Sonquist, J. A. (1963)**  
   *Problems in the analysis of survey data, and a proposal.*  
   Journal of the American Statistical Association, 58(302), 415-434.

2. **Morgan, J. N., & Messenger, R. C. (1973)**  
   *THAID: A sequential analysis program for the analysis of nominal scale dependent variables.*  
   Survey Research Center, Institute for Social Research, University of Michigan.

### Contextual References

3. **Breiman, L., Friedman, J., Stone, C. J., & Olshen, R. A. (1984)**  
   *Classification and regression trees.*  
   CRC press.

4. **Quinlan, J. R. (1986)**  
   *Induction of decision trees.*  
   Machine Learning, 1(1), 81-106.

5. **Hastie, T., Tibshirani, R., & Friedman, J. (2009)**  
   *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.).  
   Springer Science & Business Media.

---

## Appendix: Notation Summary

| Symbol | Description |
|--------|-------------|
| **X** | Feature matrix (N × p) |
| **y** | Target vector (N × 1) |
| N | Number of samples |
| p | Number of features |
| K | Number of classes (THAID only) |
| t | Current node |
| t_L, t_R | Left and right child nodes |
| TSS | Total Sum of Squares |
| WSS | Within-Group Sum of Squares |
| BSS | Between-Group Sum of Squares |
| θ | Theta statistic (modal accuracy) |
| δ | Delta statistic (distributional difference) |
| ȳ | Mean of target values |
| R² | Coefficient of determination |
| 𝟙(·) | Indicator function |

---

*This document is part of the AID-THAID-Implementation project.*  
*GitHub: https://github.com/1achraf1/AID-THAID-Implementation*
