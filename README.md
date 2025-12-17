# AID & THAID Algorithms Implementation

![Language](https://img.shields.io/badge/Languages-Python%20%7C%20R-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> A comprehensive implementation of **Automatic Interaction Detection (AID)** and **Theta Automatic Interaction Detection (THAID)** algorithms—foundational statistical segmentation techniques now accessible in both Python and R.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Algorithms](#-algorithms)
  - [AID: Automatic Interaction Detection](#aid-automatic-interaction-detection)
  - [THAID: Theta Automatic Interaction Detection](#thaid-theta-automatic-interaction-detection)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
  - [Python Implementation](#python-implementation)
  - [R Implementation](#r-implementation)
- [Documentation & Theory](#-documentation--theory)
- [Examples & Demos](#-examples--demos)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

This project revives and documents two foundational interaction detection techniques that laid the groundwork for modern machine learning algorithms. **AID** and **THAID** are precursors to contemporary decision tree methods like CART and CHAID, offering powerful tools for data segmentation and exploratory analysis.

### Why AID & THAID?

- **Historical Significance**: Understand the roots of modern decision trees
- **Interpretability**: Highly transparent segmentation logic
- **Educational Value**: Perfect for learning statistical segmentation
- **Practical Applications**: Excellent for market segmentation, risk analysis, and customer profiling

---

## ✨ Key Features

- 🐍 **Dual Implementation**: Complete Python and R versions
- 📊 **Comprehensive Documentation**: Detailed mathematical foundations
- 📓 **Interactive Notebooks**: Step-by-step Jupyter demonstrations
- ✅ **Tested & Validated**: Includes unit tests and validation suites
- 🎯 **Production-Ready**: Clean, modular, and well-documented code
- 🔧 **Flexible**: Customizable stopping rules and splitting criteria

---

## 🧠 Algorithms

### AID: Automatic Interaction Detection

**Use Case**: When your target variable is **quantitative** (continuous, interval, or ratio scale)

**Objective**: Identify segments that maximize explained variance in the target variable

**Methodology**:
- Iteratively splits data to minimize within-group sum of squared errors (SSE)
- Equivalent to maximizing between-group variance
- Produces homogeneous segments with similar target values

**Splitting Criterion**:
$$\text{SSE} = \sum_{i=1}^{n} (y_i - \bar{y})^2$$

**Example Applications**:
- Price segmentation
- Sales forecasting by customer groups
- Resource allocation optimization

---

### THAID: Theta Automatic Interaction Detection

**Use Case**: When your target variable is **qualitative** (categorical, nominal, or ordinal)

**Objective**: Find segments that best discriminate between categories of the target variable

**Methodology**:
- Uses the Theta (θ) statistic focused on modal categories
- Maximizes the difference in category distributions across segments
- Creates groups with distinct categorical profiles

**Splitting Criterion**:
$$\theta = \text{Measure of modal category concentration}$$

**Example Applications**:
- Customer classification (high/medium/low value)
- Risk categorization
- Behavioral segmentation

---

## 📂 Repository Structure

```
AID-THAID-Implementation/
│
├── AID/                          # Automatic Interaction Detection
│   ├── python/                   # Python implementation
│   │   └── aid_implementation.py
│   └── r/                        # R implementation
│       └── aid_implementation.R
│
├── THAID/                        # Theta Automatic Interaction Detection
│   ├── python/                   # Python implementation
│   │   └── thaid_implementation.py
│   └── r/                        # R implementation
│       └── thaid_implementation.R
│
├── docs/                         # Documentation
│   ├── mathematical_foundation.md
│   ├── algorithm_comparison.md
│   └── best_practices.md
│
├── notebooks/                    # Interactive demonstrations
│   ├── aid_demo.ipynb
│   └── thaid_demo.ipynb
│
├── tests/                        # Testing & validation
│   ├── thaid_test.ipynb
│   └── validation_suite.ipynb
│
├── examples/                     # Sample datasets
│   └── sample_data.csv
│
├── LICENSE
└── README.md
```

---

## 📚 Documentation & Theory

Dive deep into the mathematics and theory behind these algorithms:

| Document | Description |
|----------|-------------|
| **[Mathematical Foundation](docs/mathematical_foundation.md)** | Detailed formulas, splitting criteria, and stopping rules |
| **Algorithm Comparison** | When to use AID vs THAID vs modern methods |
| **Best Practices** | Tips for parameter tuning and interpretation |

### Key Concepts Covered:

- **Sum of Squared Errors (SSE)** minimization in AID
- **Theta (θ) statistic** calculation in THAID
- **Stopping criteria**: minimum sample size, maximum depth, significance tests
- **Pruning strategies** to prevent overfitting
- **Comparison with CART, CHAID, and Random Forests**

---

## 📓 Examples & Demos

Explore interactive Jupyter notebooks with real-world examples:

- **[AID Demo Notebook](notebooks/aid_demo.ipynb)**: Step-by-step walkthrough with housing price data
- **[THAID Demo Notebook](notebooks/thaid_demo.ipynb)**: Customer segmentation case study
- **[Test Suite](tests/thaid_test.ipynb)**: Validation and edge case testing

---

## 🤝 Contributing

Contributions are warmly welcomed! Here's how you can help:

### Ways to Contribute:

- 🐛 **Bug Reports**: Found an issue? Open a GitHub issue
- ✨ **Feature Requests**: Have an idea? We'd love to hear it
- 📝 **Documentation**: Improve clarity and add examples
- 🎨 **Visualizations**: Add tree plotting capabilities
- ⚡ **Optimizations**: Enhance performance and efficiency
- 🧪 **Testing**: Expand test coverage

### Contribution Workflow:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

Please ensure your code follows the existing style and includes appropriate tests.

---

## 🙏 Acknowledgments

- Original AID algorithm developed by **Morgan & Sonquist (1963)**
- THAID algorithm introduced by **Morgan & Messenger (1973)**

---

## 📧 Contact & Support

- **Authors**: [1achraf1](https://github.com/1achraf1) [ZIADEA](https://github.com/ZIADEA) [Asmaeelhakioui](https://github.com/Asmaeelhakioui)
- **Issues**: [GitHub Issues](https://github.com/1achraf1/AID-THAID-Implementation/issues)
- **Discussions**: [GitHub Discussions](https://github.com/1achraf1/AID-THAID-Implementation/discussions)

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

</div>
