# Wine Quality Classification & Data Mining

An end-to-end data mining and machine learning project that classifies Portuguese red wine variants ("Vinho Verde") into high and low-quality categories using physicochemical attributes. 

This project explores the entire machine learning pipeline—from rigorous preprocessing and feature selection using Mutual Information to training, hyperparameter tuning, and comparing five classification models. The findings are documented in a co-authored technical paper included in this repository.

---

## 🚀 Key Highlights & Accomplishments
* **Preprocessing & Optimization**: Applied advanced data preprocessing techniques including equal-width binning discretization, standard scaling, and feature selection via Mutual Information (MI).
* **Multi-Model Pipeline**: Trained, optimized, and evaluated five diverse classifiers: **Decision Tree**, **Random Forest**, **K-Nearest Neighbors (KNN)**, **Logistic Regression**, and **Gaussian Naive Bayes**.
* **Hyperparameter Tuning**: Utilized visual elbow methods and out-of-bag (OOB) error analysis with the `Kneed` library to locate optimal settings for K-Nearest Neighbors ($k$) and Random Forest estimators ($n\_estimators$).
* **Comparative Evaluation**: Measured and contrasted models across key classification metrics—**Accuracy**, **Precision**, **Recall**, and **F1-score**—to identify trade-offs (e.g., KNN's recall vs. Logistic Regression's precision) and quantify the performance boost from feature selection.
* **Technical Writing**: Co-authored a complete, formal technical paper ([Wine Quality Classification Paper.docx](file:///c:/Users/aidan/Desktop/DSMLAI%20Projects/Wine%20Quality%20Classification%20Project/Wine%20Quality%20Classification%20Paper.docx)) detailing the research methodology, quantitative results, and theoretical interpretations.

---

## 📊 Dataset & Preprocessing

The analysis uses the classic **UCI Wine Quality Dataset** (red wine subset), consisting of **1,599 instances** with 11 physicochemical features (such as pH levels, volatile acidity, alcohol content, sulphates, etc.) and a target quality rating.

### Target Discretization
To mitigate data imbalance and construct a clear classification problem, the original wine quality ratings (ranging from 3 to 8) were binned into a binary classification target:
* **Low Quality (Class 0)**: Ratings 3–5 (744 samples)
* **High Quality (Class 1)**: Ratings 6–8 (855 samples)

### Scaling & Normalization
Features were scaled using **Standardization (Z-score normalization)** to ensure distance-based models (KNN) and gradient-based models (Logistic Regression) are not biased by disparate feature magnitudes.

---

## 🔍 Feature Selection via Mutual Information
Using **Mutual Information (MI) classification**, we evaluated how much information each physicochemical feature shares with the wine quality target. Features yielding a `0.0` MI score were removed to eliminate noise:
* **Dropped Features**: `free sulfur dioxide` and `pH`.
* **Retained Features**: `alcohol` (highest MI score), `volatile acidity`, `sulphates`, `citric acid`, `fixed acidity`, `chlorides`, `total sulfur dioxide`, `density`, and `residual sugar`.

---

## 🛠️ Models & Hyperparameter Tuning

All models were evaluated using **10-fold cross-validation** to guarantee generalization and prevent overfitting.

1. **Decision Tree**: Splitting nodes based on information gain (entropy). Served as our baseline classifier.
2. **Gaussian Naive Bayes (GNB)**: A probabilistic classifier assuming normal distribution of features and attribute independence.
3. **K-Nearest Neighbors (KNN)**: Distance-based classifier.
   * *Without Feature Selection*: Tuned to **$k = 15$** (visually selected using error rate curves to avoid a premature elbow detected by Kneed at $k=6$).
   * *With Feature Selection*: Tuned to **$k = 9$** (successfully located via Kneed library).
4. **Logistic Regression**: Linear binary classifier utilizing a sigmoid activation function and class-weight balancing.
5. **Random Forest**: Ensemble bootstrap aggregation (bagging) classifier.
   * *Without Feature Selection*: Tuned to **$n\_estimators = 10$** trees based on Out-of-Bag (OOB) error elbow curves.
   * *With Feature Selection*: Tuned to **$n\_estimators = 12$** trees.

---

## 📈 Performance Comparison

The tables below outline the average cross-validated performance of all five models before and after applying Mutual Information feature selection.

### 1. Baseline Performance (All Features)
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.6423 | 0.6733 | 0.6513 | 0.6586 |
| **Gaussian Naive Bayes** | 0.7186 | 0.7438 | 0.7260 | 0.7292 |
| **K-Nearest Neighbors ($k=15$)** | 0.7123 | 0.7255 | **0.7649** | 0.7376 |
| **Logistic Regression** | **0.7379** | **0.7817** | 0.7130 | 0.7375 |
| **Random Forest ($n=10$)** | 0.7098 | 0.7558 | 0.6944 | 0.7091 |

### 2. Post-Feature Selection Performance (Selected Features)
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree** | 0.6560 | 0.6782 | 0.6721 | 0.6732 |
| **Gaussian Naive Bayes** | 0.7211 | 0.7519 | 0.7225 | 0.7296 |
| **K-Nearest Neighbors ($k=9$)** | 0.7142 | 0.7214 | **0.7742** | **0.7401** |
| **Logistic Regression** | **0.7361** | **0.7748** | 0.7177 | 0.7383 |
| **Random Forest ($n=12$)** | 0.7248 | 0.7737 | 0.7014 | 0.7250 |

---

## 🔑 Crucial Trade-offs & Business Applications

* **Logistic Regression (High Precision/Accuracy)**: Achieved the highest accuracy (73.79%) and precision (78.17%). Because it minimizes false positives, it is highly suited for a **wine vendor select-display**. A retailer choosing wines to showcase on a premium front display needs to ensure every selected wine is truly high-quality to avoid customer dissatisfaction and protect brand reputation.
* **K-Nearest Neighbors (High Recall/F1)**: Outperformed other models in recall (reaching 77.42% with feature selection) and secured the highest overall F1-score (74.01%). Because it minimizes false negatives, KNN is ideal for building a **high-quality wine directory or database**, where the objective is to capture as many genuine high-quality wines as possible without missing them.
* **Random Forest**: Benefited the most from feature selection, experiencing substantial increases across accuracy, precision, and recall by filtering out noisy inputs (`free sulfur dioxide` and `pH`).
* **Decision Tree**: Consistently performed the worst, indicating severe overfitting and an inability to generalize to the cross-validated folds.

---

## ⚙️ Environment Setup & Notebook Execution

### Repository Structure
* [project.ipynb](file:///c:/Users/aidan/Desktop/DSMLAI%20Projects/Wine%20Quality%20Classification%20Project/project.ipynb): Core Jupyter Notebook with python implementation.
* [winequality-red.csv](file:///c:/Users/aidan/Desktop/DSMLAI%20Projects/Wine%20Quality%20Classification%20Project/winequality-red.csv): Dataset of physicochemical wine measurements.
* [Wine Quality Classification Paper.docx](file:///c:/Users/aidan/Desktop/DSMLAI%20Projects/Wine%20Quality%20Classification%20Project/Wine%20Quality%20Classification%20Paper.docx): Full co-authored technical report documenting findings.
* [README.md](file:///c:/Users/aidan/Desktop/DSMLAI%20Projects/Wine%20Quality%20Classification%20Project/README.md): Documentation page.

### Installation
Ensure you have Python 3.8+ installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn kneed jupyter
```

### Running the Notebook
To run the analysis locally, start the Jupyter environment and open the notebook:
```bash
jupyter notebook project.ipynb
```
Run each cell sequentially to reproduce the data preprocessing, model tuning plots, feature selection scores, and final evaluation metrics.

---

## 📜 Citations
* Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Wine Quality*. UCI Machine Learning Repository. https://doi.org/10.24432/C56S3T.
