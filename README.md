### AID & THAID Implementation

A from-scratch reproduction of the original Sonquist & Morgan decision tree algorithms. This project implements the core logic of AID (Automatic Interaction Detection) and THAID (Theta Automatic Interaction Detection) without relying on external tree libraries like sklearn, rpart, or caret.

The goal is to demonstrate the fundamental mechanics of data discovery and interaction detection as they were originally conceived in the 1960s and 70s, serving as a precursor to modern algorithms like CART and CHAID.

📂 Repository Structure

.
├── AID/                # Automatic Interaction Detection (Continuous Target)
│   ├── AID.R           # R implementation of AID
│   └── AID.py          # Python implementation of AID
├── THAID/              # Theta Automatic Interaction Detection (Categorical Target)
│   ├── THAID.R         # R implementation of THAID
│   └── THAID.py        # Python implementation of THAID
├── data/               # Sample datasets for testing
├── .gitignore
└── README.md


### 🧠 Algorithms Implemented

## 1. AID (Automatic Interaction Detection)

Origin: Sonquist, J. A., & Morgan, J. N. (1964).

Purpose: Designed for continuous dependent variables.

Method: Recursively splits data into mutually exclusive groups to maximize the reduction in unexplained variance.

Criterion: Maximizes the Between Sum of Squares (BSS) / Total Sum of Squares (TSS) ratio.

## 2. THAID (Theta Automatic Interaction Detection)

Origin: Morgan, J. N., & Messenger, R. C. (1973).

Purpose: Designed for categorical (nominal/ordinal) dependent variables.

Method: Searches for splits that maximize the probability of correctly predicting the modal category.

Criterion: Maximizes the Theta (or Delta) statistic.

🚀 Usage

Prerequisites

R: Version 4.0+

Python: Version 3.8+ (requires numpy, pandas)

R Example (AID)

# Source the function
source("AID/AID.R")

# Load data
data <- read.csv("data/housing.csv")

# Run AID
# target: "price", predictors: c("sqft", "location")
model <- AID(data, target_var = "price", min_split = 20, max_depth = 4)

# Print results
print(model)


Python Example (THAID)

import pandas as pd
from THAID.THAID import ThaidTree

# Load data
df = pd.read_csv("data/titanic.csv")

# Initialize and fit
# Target: "Survived" (Categorical)
tree = ThaidTree(target="Survived", min_samples=10, max_depth=3)
tree.fit(df)

# Visualize
tree.print_tree()


⚙️ Key Implementation Details

No "Black Box" Libraries: All splitting logic, variance calculations, and tree traversal are written in raw code.

Category Sorting: Implements the efficiency hack where nominal categories are sorted by their mean dependent value (AID) to reduce split complexity from $2^{N-1}$ to $N-1$.

Recursion: Uses recursive functions to build the tree structure node by node.

📚 References

Sonquist, J. A., & Morgan, J. N. (1964). The Detection of Interaction Effects. Survey Research Center, Institute for Social Research, University of Michigan.

Morgan, J. N., & Messenger, R. C. (1973). THAID: A Sequential Analysis Program for the Analysis of Nominal Scale Dependent Variables. It's
