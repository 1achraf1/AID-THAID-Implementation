
from thaid import THAID
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer



# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class THAIDTester:
    """Comprehensive testing  for the THAID algorithm against real-world data."""
    
    def __init__(self, base_model_params=None):
        """
        Initialize the tester.
        
        Parameters
        ----------
        base_model_params : dict
            Dictionary of parameters to initialize the THAID model with.
        """
        self.base_params = base_model_params if base_model_params else {
            'min_samples_split': 20, 
            'min_samples_leaf': 1, 
            'max_depth': None
        }
        self.results = {}
        
    def _get_fresh_model(self, **kwargs):
        """Helper to get a new instance of THAID to prevent state leakage between tests."""
        params = self.base_params.copy()
        params.update(kwargs)
        return THAID(**params)

    def load_datasets(self):
        """Load various benchmark datasets."""
        print("Loading datasets...")
        datasets = {}
        
        # 1. Iris Dataset (Simple numeric)
        iris = load_iris()
        datasets['iris'] = {
            'X': pd.DataFrame(iris.data, columns=iris.feature_names),
            'y': pd.Series(iris.target),
            'description': 'Iris (150 samples, 4 features, 3 classes)'
        }
        
        # Wine Dataset (Multiclass numeric)
        wine = load_wine()
        datasets['wine'] = {
            'X': pd.DataFrame(wine.data, columns=wine.feature_names),
            'y': pd.Series(wine.target),
            'description': 'Wine (178 samples, 13 features, 3 classes)'
        }
        
        # Breast Cancer (Binary high-dimensional)
        cancer = load_breast_cancer()
        datasets['breast_cancer'] = {
            'X': pd.DataFrame(cancer.data, columns=cancer.feature_names),
            'y': pd.Series(cancer.target),
            'description': 'Breast Cancer (569 samples, 30 features, 2 classes)'
        }
        
        # Titanic (Categorical heavy)
        try:
            titanic = sns.load_dataset('titanic')
            # Select relevant features and drop minimal NaNs for stability
            titanic = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'survived']]
            titanic = titanic.dropna()
            
            X_titanic = titanic.drop(columns=['survived'])
            y_titanic = titanic['survived']
            
            # Simple encoding for this test
            le = LabelEncoder()
            X_titanic['sex'] = le.fit_transform(X_titanic['sex'])
            X_titanic['embarked'] = le.fit_transform(X_titanic['embarked'])
            
            datasets['titanic'] = {
                'X': X_titanic,
                'y': y_titanic,
                'description': f'Titanic ({len(X_titanic)} samples, 7 features, 2 classes)'
            }
        except Exception as e:
            print(f"Warning: Could not load Titanic dataset ({e})")

        # Penguins (Mixed types)
        try:
            penguins = sns.load_dataset('penguins')
            penguins = penguins.dropna()
            
            X_penguins = penguins.drop(columns=['species'])
            y_penguins = LabelEncoder().fit_transform(penguins['species'])
            
            # Encode categorical cols
            le = LabelEncoder()
            X_penguins['sex'] = le.fit_transform(X_penguins['sex'])
            X_penguins['island'] = le.fit_transform(X_penguins['island'])
            
            datasets['penguins'] = {
                'X': X_penguins,
                'y': pd.Series(y_penguins),
                'description': f'Penguins ({len(X_penguins)} samples, 6 features, 3 classes)'
            }
        except Exception as e:
            print(f"Warning: Could not load Penguins dataset ({e})")
            
        print(f"✓ Loaded {len(datasets)} datasets successfully.")
        return datasets
    
    def test_basic_functionality(self, datasets):
        """Test basic fit/predict pipeline."""
        print("\n" + "="*60)
        print("1. BASIC FUNCTIONALITY TEST")
        print("="*60)
        
        results = {}
        
        for name, data in datasets.items():
            print(f"\ndataset: {name.upper()}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.3, random_state=42, stratify=data['y']
            )
            
            try:
                model = self._get_fresh_model()
                
                # Fit
                t0 = time.time()
                model.fit(X_train, y_train)
                fit_time = time.time() - t0
                
                # Predict
                t0 = time.time()
                y_pred = model.predict(X_test)
                pred_time = time.time() - t0
                
                train_acc = model.score(X_train, y_train)
                test_acc = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'fit_time': fit_time,
                    'pred_time': pred_time,
                    'success': True
                }
                
                print(f"  ✓ Train Acc: {train_acc:.4f}")
                print(f"  ✓ Test Acc:  {test_acc:.4f}")
                print(f"  ✓ Time:      {fit_time*1000:.2f}ms (fit) / {pred_time*1000:.2f}ms (pred)")
                
            except Exception as e:
                print(f"  ✗ FAILED: {str(e)}")
                results[name] = {'success': False, 'error': str(e)}
                
        self.results['basic'] = results
        return results

    def test_cross_validation(self, datasets, cv=5):
        """Test stability using Stratified K-Fold."""
        print("\n" + "="*60)
        print("2. CROSS-VALIDATION ROBUSTNESS")
        print("="*60)
        
        results = {}
        
        for name, data in datasets.items():
            print(f"\nDataset: {name.upper()}")
            
            try:
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                scores = []
                
                # We need to reset index to handle loc/iloc correctly during CV splitting
                X = data['X'].reset_index(drop=True)
                y = data['y'].reset_index(drop=True)
                
                for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                    model = self._get_fresh_model()
                    
                    X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    model.fit(X_fold_train, y_fold_train)
                    acc = model.score(X_fold_val, y_fold_val)
                    scores.append(acc)
                    # print(f"  Fold {fold}: {acc:.4f}")
                    
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                results[name] = {
                    'cv_scores': scores,
                    'mean': mean_score,
                    'std': std_score,
                    'success': True
                }
                print(f"  ✓ Mean Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
                
            except Exception as e:
                print(f"  ✗ FAILED: {str(e)}")
                import traceback
                traceback.print_exc()
                results[name] = {'success': False, 'error': str(e)}
                
        self.results['cv'] = results
        return results

    def test_comparison_sklearn(self, datasets):
        """Benchmark against Scikit-Learn's DecisionTreeClassifier."""
        print("\n" + "="*60)
        print("3. BENCHMARK VS SKLEARN")
        print("="*60)
        
        results = {}
        
        for name, data in datasets.items():
            print(f"\nDataset: {name.upper()}")
            
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.3, random_state=42, stratify=data['y']
            )
            
            try:
                # THAID
                thaid = self._get_fresh_model()
                t0 = time.time()
                thaid.fit(X_train, y_train)
                t_thaid = time.time() - t0
                acc_thaid = thaid.score(X_test, y_test)
                
                # Sklearn 
                sk_tree = DecisionTreeClassifier(
                    min_samples_split=self.base_params['min_samples_split'],
                    min_samples_leaf=self.base_params['min_samples_leaf'],
                    max_depth=self.base_params['max_depth'],
                    random_state=42
                )
                t0 = time.time()
                sk_tree.fit(X_train, y_train)
                t_sk = time.time() - t0
                acc_sk = sk_tree.score(X_test, y_test)
                
                diff = acc_thaid - acc_sk
                
                results[name] = {
                    'thaid_acc': acc_thaid,
                    'sk_acc': acc_sk,
                    'thaid_time': t_thaid,
                    'sk_time': t_sk,
                    'success': True
                }
                
                print(f"  THAID Accuracy:   {acc_thaid:.4f} ({t_thaid*1000:.2f}ms)")
                print(f"  Sklearn Accuracy: {acc_sk:.4f} ({t_sk*1000:.2f}ms)")
                print(f"  Difference:       {diff:+.4f}")
                
            except Exception as e:
                print(f"  ✗ FAILED: {str(e)}")
                
        self.results['benchmark'] = results
        return results

    def generate_visualizations(self):
        """Generate plots based on collected results."""
        if not self.results:
            print("No results to visualize.")
            return

        print("\n" + "="*60)
        print("GENERATING PLOTS...")
        print("="*60)
        
        # Benchmark Plot
        if 'benchmark' in self.results:
            data = self.results['benchmark']
            names = [k for k in data.keys() if data[k].get('success')]
            
            thaid_scores = [data[k]['thaid_acc'] for k in names]
            sk_scores = [data[k]['sk_acc'] for k in names]
            
            x = np.arange(len(names))
            width = 0.35
            
            plt.figure(figsize=(10, 6))
            plt.bar(x - width/2, thaid_scores, width, label='THAID', color='#4c72b0')
            plt.bar(x + width/2, sk_scores, width, label='Sklearn CART', color='#dd8452')
            
            plt.ylabel('Accuracy')
            plt.title('THAID vs Sklearn Performance')
            plt.xticks(x, names, rotation=45)
            plt.ylim(0, 1.1)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('thaid_benchmark.png')
            print("✓ Saved 'thaid_benchmark.png'")
            plt.close()

        # Fit Time Comparison
        if 'benchmark' in self.results:
            data = self.results['benchmark']
            names = [k for k in data.keys() if data[k].get('success')]
            
            # Using log scale often helps visualize speed differences better
            thaid_time = [data[k]['thaid_time'] for k in names]
            sk_time = [data[k]['sk_time'] for k in names]
            
            plt.figure(figsize=(10, 6))
            plt.plot(names, thaid_time, marker='o', label='THAID', linewidth=2)
            plt.plot(names, sk_time, marker='x', label='Sklearn', linewidth=2, linestyle='--')
            
            plt.ylabel('Time (seconds)')
            plt.title('Training Time Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('thaid_speed.png')
            print("✓ Saved 'thaid_speed.png'")
            plt.close()

def run_tests():
    # Initialize Tester
    tester = THAIDTester(base_model_params={
        'min_samples_split': 20,
        'min_samples_leaf': 5,
        'max_depth': 6
    })
    
    #  Load Data
    datasets = tester.load_datasets()
    
    # Run Tests
    tester.test_basic_functionality(datasets)
    tester.test_cross_validation(datasets)
    tester.test_comparison_sklearn(datasets)
    
    # Viz
    tester.generate_visualizations()
    
    print("\nTests Completed.")

if __name__ == "__main__":
    run_tests()
