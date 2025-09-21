# Multi-Strategy Ensemble Learning for Binary Data Classification

[Full Report](https://drive.google.com/file/d/1Cf21G0ubgu8sm_Y_j2p2E1bngO8KpD2U/view?usp=sharing)

[Kaggle Competition Leaderboard](https://www.kaggle.com/competitions/datascience-4-competition/leaderboard)  First Score (0.41791)

This project develops a robust classification pipeline for high-dimensional binary datasets. The dataset contained 64 binary features and a categorical target, with class imbalance handled using SMOTE, producing a balanced training set of 792 samples.

Features were scaled using StandardScaler to ensure consistency across training and testing. Hyperparameters were optimized using Optuna with stratified k-fold cross-validation (k=3 for ensemble, k=5 for individual models).

The final model is a soft-voting ensemble integrating multiple Bernoulli Naive Bayes classifiers, Logistic Regression, XGBoost, Balanced Random Forest, and three standard Random Forest classifiers. Ensemble hyperparameters were specifically tuned to maximize cross-validated accuracy (~0.437).

Exploratory methods like CTGAN, MLP-based feature extraction, and Random Forest-based feature selection were investigated but not included in the final pipeline. 

Critical attention was given to preprocessing consistency: the test set was scaled using the training scaler to prevent distribution mismatch. 

The ensemble approach balances diverse algorithmic strengths, improves robustness, and handles class imbalance effectively. Future directions include exploring advanced feature selection, stacking ensembles, alternative imbalance techniques, and probability calibration.

Key takeaways: multi-strategy ensembles are effective for high-dimensional binary data, SMOTE improves minority class performance, and consistent preprocessing is crucial for valid inference.
