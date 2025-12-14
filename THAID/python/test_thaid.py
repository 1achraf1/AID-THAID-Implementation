import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
from thaid import THAID

class THAIDTester:
    """Comprehensive testing for THAID algorithm."""
    
    def __init__(self, thaid_model):
        self.model = thaid_model
        self.results = {}
        
    def load_datasets(self):
        """Load various famous datasets for testing."""
        datasets = {}
        
        # Iris Dataset (multiclass, numeric features)
        iris = load_iris()
        datasets['iris'] = {
            'X': pd.DataFrame(iris.data, columns=iris.feature_names),
            'y': pd.Series(iris.target),
            'description': 'Iris (150 samples, 4 features, 3 classes)'
        }
        
        # Wine Dataset (multiclass, numeric features)
        wine = load_wine()
        datasets['wine'] = {
            'X': pd.DataFrame(wine.data, columns=wine.feature_names),
            'y': pd.Series(wine.target),
            'description': 'Wine (178 samples, 13 features, 3 classes)'
        }
        
        # Breast Cancer Dataset (binary, numeric features)
        cancer = load_breast_cancer()
        datasets['breast_cancer'] = {
            'X': pd.DataFrame(cancer.data, columns=cancer.feature_names),
            'y': pd.Series(cancer.target),
            'description': 'Breast Cancer (569 samples, 30 features, 2 classes)'
        }
        
        # Titanic Dataset (mixed features - if available via seaborn)
        try:
            titanic = sns.load_dataset('titanic')
            titanic = titanic.dropna(subset=['age', 'embarked', 'fare'])
            
            X_titanic = titanic[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
            # Encode categorical variables
            le_sex = LabelEncoder()
            le_embarked = LabelEncoder()
            X_titanic['sex'] = le_sex.fit_transform(X_titanic['sex'])
            X_titanic['embarked'] = le_embarked.fit_transform(X_titanic['embarked'])
            
            datasets['titanic'] = {
                'X': X_titanic,
                'y': titanic['survived'],
                'description': f'Titanic ({len(X_titanic)} samples, 7 features, 2 classes)'
            }
        except Exception as e:
            print(f"Could not load Titanic dataset: {e}")
        
        # Penguins Dataset (mixed features)
        try:
            penguins = sns.load_dataset('penguins')
            penguins = penguins.dropna()
            
            X_penguins = penguins[['bill_length_mm', 'bill_depth_mm', 
                                   'flipper_length_mm', 'body_mass_g', 'sex', 'island']]
            # Encode categorical variables
            le_sex = LabelEncoder()
            le_island = LabelEncoder()
            X_penguins['sex'] = le_sex.fit_transform(X_penguins['sex'])
            X_penguins['island'] = le_island.fit_transform(X_penguins['island'])
            
            y_penguins = LabelEncoder().fit_transform(penguins['species'])
            
            datasets['penguins'] = {
                'X': X_penguins,
                'y': pd.Series(y_penguins),
                'description': f'Penguins ({len(X_penguins)} samples, 6 features, 3 classes)'
            }
        except Exception as e:
            print(f"Could not load Penguins dataset: {e}")
        
        return datasets
    
    def test_basic_functionality(self, datasets):
        """Test basic fit/predict functionality."""
        print("="*80)
        print("BASIC FUNCTIONALITY TEST")
        print("="*80)
        
        results = {}
        
        for name, data in datasets.items():
            print(f"\n{name.upper()}: {data['description']}")
            print("-" * 60)
            
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.3, random_state=42, stratify=data['y']
            )
            
            try:
                # Fit model
                start_time = time.time()
                self.model.fit(X_train, y_train)
                fit_time = time.time() - start_time
                
                # Predict
                start_time = time.time()
                y_pred = self.model.predict(X_test)
                predict_time = time.time() - start_time
                
                # Metrics
                train_acc = self.model.score(X_train, y_train)
                test_acc = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'fit_time': fit_time,
                    'predict_time': predict_time,
                    'success': True
                }
                
                print(f"✓ Training Accuracy: {train_acc:.4f}")
                print(f"✓ Testing Accuracy:  {test_acc:.4f}")
                print(f"✓ Fit Time:          {fit_time:.4f}s")
                print(f"✓ Predict Time:      {predict_time:.4f}s")
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                results[name] = {'success': False, 'error': str(e)}
        
        self.results['basic_functionality'] = results
        return results
    
    def test_cross_validation(self, datasets, cv=5):
        """Test with cross-validation for robustness."""
        print("\n" + "="*80)
        print("CROSS-VALIDATION TEST")
        print("="*80)
        
        results = {}
        
        for name, data in datasets.items():
            print(f"\n{name.upper()}: {data['description']}")
            print("-" * 60)
            
            try:
                cv_scores = []
                skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                
                for fold, (train_idx, test_idx) in enumerate(skf.split(data['X'], data['y']), 1):
                    X_train = data['X'].iloc[train_idx]
                    X_test = data['X'].iloc[test_idx]
                    y_train = data['y'].iloc[train_idx]
                    y_test = data['y'].iloc[test_idx]
                    
                    self.model.fit(X_train, y_train)
                    score = self.model.score(X_test, y_test)
                    cv_scores.append(score)
                    print(f"  Fold {fold}: {score:.4f}")
                
                mean_score = np.mean(cv_scores)
                std_score = np.std(cv_scores)
                
                results[name] = {
                    'cv_scores': cv_scores,
                    'mean_accuracy': mean_score,
                    'std_accuracy': std_score,
                    'success': True
                }
                
                print(f"\n✓ Mean CV Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                results[name] = {'success': False, 'error': str(e)}
        
        self.results['cross_validation'] = results
        return results
    
    def test_parameter_sensitivity(self, datasets):
        """Test sensitivity to hyperparameters."""
        print("\n" + "="*80)
        print("PARAMETER SENSITIVITY TEST")
        print("="*80)
        
        # Test different parameter combinations
        param_configs = [
            {'min_samples_split': 10, 'min_samples_leaf': 1, 'max_depth': None},
            {'min_samples_split': 20, 'min_samples_leaf': 5, 'max_depth': None},
            {'min_samples_split': 20, 'min_samples_leaf': 1, 'max_depth': 5},
            {'min_samples_split': 50, 'min_samples_leaf': 10, 'max_depth': 3},
            {'min_samples_split': 20, 'min_samples_leaf': 1, 'max_depth': 10},
            {'min_samples_split': 20, 'min_samples_leaf': 1, 'max_depth': 5},
            {'min_samples_split': 20, 'min_samples_leaf': 1, 'max_depth': 3},
            
        ]
        
        results = {}
        
        for name, data in datasets.items():
            print(f"\n{name.upper()}")
            print("-" * 60)
            
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.3, random_state=42, stratify=data['y']
            )
            
            config_results = []
            
            for i, params in enumerate(param_configs, 1):
                try:
                    model = THAID(**params)
                    model.fit(X_train, y_train)
                    test_acc = model.score(X_test, y_test)
                    
                    config_results.append({
                        'params': params,
                        'accuracy': test_acc
                    })
                    
                    print(f"  Config {i} {params}: {test_acc:.4f}")
                    
                except Exception as e:
                    print(f"  Config {i} {params}: ERROR - {str(e)}")
            
            results[name] = config_results
        
        self.results['parameter_sensitivity'] = results
        return results
    
    def test_comparison_with_sklearn(self, datasets):
        """Compare THAID with sklearn DecisionTreeClassifier."""
        print("\n" + "="*80)
        print("COMPARISON WITH SKLEARN DECISION TREE")
        print("="*80)
        
        results = {}
        
        for name, data in datasets.items():
            print(f"\n{name.upper()}")
            print("-" * 60)
            
            X_train, X_test, y_train, y_test = train_test_split(
                data['X'], data['y'], test_size=0.3, random_state=42, stratify=data['y']
            )
            
            try:
                # THAID
                start_time = time.time()
                thaid_model = THAID(min_samples_split=20, min_samples_leaf=1)
                thaid_model.fit(X_train, y_train)
                thaid_fit_time = time.time() - start_time
                
                start_time = time.time()
                thaid_pred = thaid_model.predict(X_test)
                thaid_predict_time = time.time() - start_time
                
                thaid_acc = accuracy_score(y_test, thaid_pred)
                
                # Sklearn
                start_time = time.time()
                sklearn_model = DecisionTreeClassifier(min_samples_split=20, min_samples_leaf=1, random_state=42)
                sklearn_model.fit(X_train, y_train)
                sklearn_fit_time = time.time() - start_time
                
                start_time = time.time()
                sklearn_pred = sklearn_model.predict(X_test)
                sklearn_predict_time = time.time() - start_time
                
                sklearn_acc = accuracy_score(y_test, sklearn_pred)
                
                results[name] = {
                    'thaid_accuracy': thaid_acc,
                    'sklearn_accuracy': sklearn_acc,
                    'thaid_fit_time': thaid_fit_time,
                    'sklearn_fit_time': sklearn_fit_time,
                    'thaid_predict_time': thaid_predict_time,
                    'sklearn_predict_time': sklearn_predict_time,
                    'success': True
                }
                
                print(f"THAID:")
                print(f"  Accuracy:      {thaid_acc:.4f}")
                print(f"  Fit Time:      {thaid_fit_time:.4f}s")
                print(f"  Predict Time:  {thaid_predict_time:.4f}s")
                
                print(f"\nSklearn DecisionTree:")
                print(f"  Accuracy:      {sklearn_acc:.4f}")
                print(f"  Fit Time:      {sklearn_fit_time:.4f}s")
                print(f"  Predict Time:  {sklearn_predict_time:.4f}s")
                
                print(f"\nComparison:")
                print(f"  Accuracy Diff: {thaid_acc - sklearn_acc:+.4f}")
                print(f"  Speed Ratio:   {sklearn_fit_time/thaid_fit_time:.2f}x")
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                results[name] = {'success': False, 'error': str(e)}
        
        self.results['sklearn_comparison'] = results
        return results
    
    def test_edge_cases(self, datasets):
        """Test edge cases and robustness."""
        print("\n" + "="*80)
        print("EDGE CASES TEST")
        print("="*80)
        
        results = {}
        
        # Use first dataset for edge case testing
        name = list(datasets.keys())[0]
        data = datasets[name]
        
        X, y = data['X'], data['y']
        
        test_cases = {
            'single_sample': (X.iloc[:1], y.iloc[:1]),
            'two_samples': (X.iloc[:2], y.iloc[:2]),
            'imbalanced': (X, y),  # Will use as-is
            'single_feature': (X.iloc[:, :1], y),
        }
        
        for test_name, (X_test, y_test) in test_cases.items():
            print(f"\n{test_name.upper()}")
            print("-" * 60)
            
            try:
                model = THAID(min_samples_split=2, min_samples_leaf=1, max_depth=3)
                model.fit(X_test, y_test)
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                
                print(f"✓ SUCCESS: Accuracy = {acc:.4f}")
                results[test_name] = {'success': True, 'accuracy': acc}
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)}")
                results[test_name] = {'success': False, 'error': str(e)}
        
        self.results['edge_cases'] = results
        return results
    
    def visualize_results(self):
        """Create visualizations of test results."""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # 1. Accuracy comparison plot
        if 'basic_functionality' in self.results:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            data = self.results['basic_functionality']
            datasets_names = [k for k, v in data.items() if v.get('success', False)]
            train_accs = [data[k]['train_accuracy'] for k in datasets_names]
            test_accs = [data[k]['test_accuracy'] for k in datasets_names]
            
            x = np.arange(len(datasets_names))
            width = 0.35
            
            axes[0].bar(x - width/2, train_accs, width, label='Train', alpha=0.8)
            axes[0].bar(x + width/2, test_accs, width, label='Test', alpha=0.8)
            axes[0].set_xlabel('Dataset')
            axes[0].set_ylabel('Accuracy')
            axes[0].set_title('THAID Performance Across Datasets')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(datasets_names, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            
            # 2. Timing comparison
            fit_times = [data[k]['fit_time'] for k in datasets_names]
            predict_times = [data[k]['predict_time'] for k in datasets_names]
            
            axes[1].bar(x - width/2, fit_times, width, label='Fit Time', alpha=0.8)
            axes[1].bar(x + width/2, predict_times, width, label='Predict Time', alpha=0.8)
            axes[1].set_xlabel('Dataset')
            axes[1].set_ylabel('Time (seconds)')
            axes[1].set_title('THAID Execution Time')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(datasets_names, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('thaid_performance.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: thaid_performance.png")
            plt.close()
        
        # 3. Cross-validation results
        if 'cross_validation' in self.results:
            data = self.results['cross_validation']
            datasets_names = [k for k, v in data.items() if v.get('success', False)]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for i, name in enumerate(datasets_names):
                cv_scores = data[name]['cv_scores']
                ax.plot(range(1, len(cv_scores) + 1), cv_scores, 
                       marker='o', label=name, linewidth=2, markersize=8)
            
            ax.set_xlabel('Fold', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title('Cross-Validation Scores Across Datasets', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('thaid_cv_scores.png', dpi=300, bbox_inches='tight')
            print("✓ Saved: thaid_cv_scores.png")
            plt.close()
    
    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        report = []
        report.append("\nTHAID ALGORITHM TEST REPORT")
        report.append("=" * 80)
        report.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary statistics
        if 'basic_functionality' in self.results:
            report.append("\n1. BASIC FUNCTIONALITY")
            report.append("-" * 80)
            data = self.results['basic_functionality']
            success_count = sum(1 for v in data.values() if v.get('success', False))
            report.append(f"Datasets Tested: {len(data)}")
            report.append(f"Successful: {success_count}/{len(data)}")
            
            for name, result in data.items():
                if result.get('success', False):
                    report.append(f"\n{name}:")
                    report.append(f"  Train Accuracy: {result['train_accuracy']:.4f}")
                    report.append(f"  Test Accuracy:  {result['test_accuracy']:.4f}")
                    report.append(f"  Fit Time:       {result['fit_time']:.4f}s")
        
        if 'cross_validation' in self.results:
            report.append("\n\n2. CROSS-VALIDATION RESULTS")
            report.append("-" * 80)
            data = self.results['cross_validation']
            
            for name, result in data.items():
                if result.get('success', False):
                    report.append(f"\n{name}:")
                    report.append(f"  Mean Accuracy: {result['mean_accuracy']:.4f}")
                    report.append(f"  Std Accuracy:  {result['std_accuracy']:.4f}")
        
        if 'sklearn_comparison' in self.results:
            report.append("\n\n3. COMPARISON WITH SKLEARN")
            report.append("-" * 80)
            data = self.results['sklearn_comparison']
            
            for name, result in data.items():
                if result.get('success', False):
                    report.append(f"\n{name}:")
                    report.append(f"  THAID Accuracy:   {result['thaid_accuracy']:.4f}")
                    report.append(f"  Sklearn Accuracy: {result['sklearn_accuracy']:.4f}")
                    report.append(f"  Difference:       {result['thaid_accuracy'] - result['sklearn_accuracy']:+.4f}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save to file
        with open('thaid_test_report.txt', 'w') as f:
            f.write(report_text)
        print("\n✓ Saved: thaid_test_report.txt")
        
        return report_text


def run_complete_test():
    """Run all tests."""
    print("\n" + "="*80)
    print("THAID ALGORITHM - COMPLETE TEST SUITE")
    print("="*80)
    
    # Initialize model and tester
    model = THAID(min_samples_split=20, min_samples_leaf=1, max_depth=None)
    tester = THAIDTester(model)
    
    # Load datasets
    print("\nLoading datasets...")
    datasets = tester.load_datasets()
    print(f"✓ Loaded {len(datasets)} datasets")
    
    # Run all tests
    tester.test_basic_functionality(datasets)
    tester.test_cross_validation(datasets, cv=5)
    tester.test_parameter_sensitivity(datasets)
    tester.test_comparison_with_sklearn(datasets)
    tester.test_edge_cases(datasets)
    
    # Generate visualizations and report
    tester.visualize_results()
    tester.generate_report()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETED")
    print("="*80)
    
    return tester


if __name__ == "__main__":
    
    tester = run_complete_test()
