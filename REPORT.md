# EEG-Project
We are testing how class weights ***and*** type of model affects that classification ability of the Valence, Arousal, Dominance, and Liking labels from the Deap-Dataset. We use three different types of ML models which are Random Forest, Multi-Layered Perceptron (MLP), and an eXtreme Gradient Boosting classifier (XGB). We use multi-label binary classification where we take the labels from the deap dataset and deduce a binary label-matrix by setting all values <5 to 0 and all values ≥5 to 1. 

## Experiment
1. Essential Question 
2. Background
3. Hypothesis
4. Procedure
5. Raw Data
6. Analysis
7. Error
7. Conclusion

## Essential Question
How do class weights and type of model affect the classification ability of Machine Learning models on the Valence, Arousal, Dominance, and Liking labels from the Deap-Dataset?

## Background 


## Hypothesis 
We are testing two hypotheses:
1. If we test 3 different types of models (xgb, mlp, random_forest) on the deap dataset for multi-label classification then the mlp will perform the best, xgb will come second, and random_forest will be last because mlp can capture temporal relationships, xgb has advanced grandient boosting allowing for better generalization, while random_forest is a relatively simple foresting algoritm designed for interpretability. 

2. If we use class weights on 3 different types of models (xgb, mlp, random_forest) on the deap dataset for multi-label classification then the weighted models will be perform better than their respective unweighted counterparts because class weights will help combat class imbalance after label binarization.   

## Procedure
Experiment setup:
- Dataset: DEAP (32 subjects × 40 trials = 1280 trials)
- Feature set: Hjorth + bandpower + skewness + kurtosis + Higuchi FD
- Models: XGBoost, RandomForest, MLP
- Weighting: per-label sample weighting (sklearn.compute_class_weight) vs unweighted
- Main Metrics: `f1`(`precision`, `recall`), `balanced_accuracy`,
- Other Metrics: `hamming_loss`, `kappa`, `confusion matrix`(`tp`, `tn`, `fp`, `fn`)

### Data Collection


## Data
[Google Drive Link](https://drive.google.com/file/d/18z3dpyH-sQxGPblzBjFJTmk49Cvcdj-_/view?usp=sharing) containing all trained models. 

## Analysis

## Error

## Conclusion