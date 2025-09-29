# EEG-Project
We are testing how class weights ***and*** type of model affects the classification ability of the Valence, Arousal, Dominance, and Liking labels from the Deap-Dataset. We use three different types of ML models which are Random Forest, Multi-Layered Perceptron (MLP), and an eXtreme Gradient Boosting classifier (XGB). We use multi-label binary classification where we take the labels from the deap dataset and deduce a binary label-matrix by setting all values <5 to 0 and all values ≥5 to 1. 

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
How do class weights and the type of model affect the classification ability of Machine Learning models on the Valence, Arousal, Dominance, and Liking labels from the Deap-Dataset?

### Hypothesis 
We are testing two hypotheses:
1. If we test 3 different types of models (`xgb`, `mlp`, `random_forest`) on the deap dataset for multi-label classification, then the mlp will perform the best, xgb will come second, and random_forest will be last because mlp can capture temporal relationships, xgb has advanced gradient boosting allowing for better generalization, while random_forest is a relatively simple foresting algorithm designed for interpretability. 

2. If we use class weights on 3 different types of models (`xgb`, `mlp`, `random_forest`) on the deap dataset for multi-label classification, then the weighted models will perform better than their respective unweighted counterparts because class weights will help combat class imbalance after label binarization.   

## Procedure
Experiment setup:
- Dataset: DEAP (32 subjects × 40 trials = 1280 trials)
- Feature set: Hjorth + bandpower + skewness + kurtosis + Higuchi FD
- Models: XGBoost, RandomForest, MLP
- Weighting: per-label sample weighting (`sklearn.compute_class_weight`) vs unweighted
- Main Metrics: `f1`(`precision`, `recall`), `balanced_accuracy`,
- Other Metrics: `hamming_loss`, `kappa`, `confusion matrix`(`tp`, `tn`, `fp`, `fn`)

### Data Collection
The data was collected using the following procedure: 
1. Run a specific model with a certain label type
2. Save model as either a `.keras` or `.joblib` file
3. Append `model_name`, `y_test`, `classification_reports`, and `y_pred_bins` to their respective lists
4. Create a dictionary with the four lists
5. Repeat step one-four for all 6 files (2 weighting settings * 3 model types) 
6. In the `analysis.py` file, get all dictionaries from the other files and put them into a mega-dictionary, consisting of all 3 model types for all 4 labels, for both weighted and unweighted groups. 
7. Calculate result metrics and sort important data from the mega-dictionary into a pandas dataframe for easy manipulation. 

## Data
[Google Drive Link](https://drive.google.com/file/d/18z3dpyH-sQxGPblzBjFJTmk49Cvcdj-_/view?usp=sharing) containing all trained models.

### Data Table
|Model_key               |Label    |F1    |Balanced Accuracy|
|------------------------|---------|------|-----------------|
|mlp_weighted            |Valence  |0.1379|0.5039           |
|mlp_weighted            |Arousal  |0.0   |0.4901           |
|mlp_weighted            |Dominance|0.2345|0.5268           |
|mlp_weighted            |Liking   |0.0741|0.5039           |
|mlp_unweighted          |Valence  |0.0   |0.4868           |
|mlp_unweighted          |Arousal  |0.0   |0.5              |
|mlp_unweighted          |Dominance|0.566 |0.5316           |
|mlp_unweighted          |Liking   |0.6748|0.5508           |
|xgb_weighted            |Valence  |0.2588|0.5257           |
|xgb_weighted            |Arousal  |0.4091|0.6235           |
|xgb_weighted            |Dominance|0.6777|0.5646           |
|xgb_weighted            |Liking   |0.7194|0.5898           |
|xgb_unweighted          |Valence  |0.1132|0.5046           |
|xgb_unweighted          |Arousal  |0.24  |0.5632           |
|xgb_unweighted          |Dominance|0.7094|0.5352           |
|xgb_unweighted          |Liking   |0.7655|0.5352           |
|random_forest_weighted  |Valence  |0.1538|0.5237           |
|random_forest_weighted  |Arousal  |0.1739|0.5455           |
|random_forest_weighted  |Dominance|0.7519|0.5956           |
|random_forest_weighted  |Liking   |0.7755|0.5391           |
|random_forest_unweighted|Valence  |0.2222|0.5487           |
|random_forest_unweighted|Arousal  |0.4   |0.6242           |
|random_forest_unweighted|Dominance|0.7388|0.57             |
|random_forest_unweighted|Liking   |0.7959|0.5742           |

### Graphs
![Descriptive Alt Text for Image](/Images/F1_Score_by_Label_Weighted_vs_Unweighted.png)
---
![Descriptive Alt Text for Image](/Images/F1_Score_Comparison_Across_Models.png)
---
![Descriptive Alt Text for Image](/Images/Average_F1_Score_Across_Labels.png)

<details>
  <summary>Click to view Balanced Accuracy Plots</summary>

![Descriptive Alt Text for Image](/Images/Balanced_Accuracy_by_Label_Weighted_vs_Unweighted.png)
---
![Descriptive Alt Text for Image](/Images/Balanced_Accuracy_Comparison_Across_Models.png)
---
![Descriptive Alt Text for Image](/Images/Average_Balanced_Accuracy_Across_Labels.png)

</details>

## Analysis
### Analysis of Weighting Effects on Classification

This analysis examines how applying **class weights** impacted the performance of the Multi-Layer Perceptron (`MLP`), Random Forest (`RF`), and XGBoost (`XGB`) models across four emotion labels (Valence, Arousal, Dominance, Liking).
* * * * *
### 1. Multi-Layer Perceptron (MLP)

#### F1 Score

Weighting **deteriorated the classification ability of the MLP significantly**.

-   It had a disastrous effect on **Dominance** and **Liking**, reducing their F1 Scores by **0.3315** and **0.6007**, respectively.

-   The overall classification performance was severely harmed by weighting, despite a marginal improvement in Valence.

#### Balanced Accuracy

Weighting had **minimal impact** on Balanced Accuracy, resulting in a score reduction in three out of four cases.
* * * * *

### 2. Random Forest (RF)

#### F1 Score

Weighting resulted in **slightly worse overall performance**.

-   The F1 scores for **Dominance** and **Liking** were largely unaffected.

-   However, **Arousal** and **Valence** F1 scores decreased by **0.2261** and **0.0684**, respectively.

#### Balanced Accuracy

Weighting generally caused **minor harm**.

-   It reduced the Balanced Accuracy for the Arousal, Liking, and Valence models.

-   The **Arousal** model was particularly affected, where weighting decreased the balanced accuracy by **0.0787**. Overall, weighting did more harm than good to the Balanced Accuracy.

* * * * *

### 3. XGBoost (XGB)

#### F1 Score

Weighting caused **significantly better performance** for XGBoost.

-   F1 scores for **Arousal** and **Valence** improved substantially by **0.1691** and **0.1456**, respectively.

-   Although Dominance and Liking saw slight F1 score decreases, the overall classification performance saw a marked benefit from weighting.

#### Balanced Accuracy

Weighting led to **consistent improvement** across the board.

-   Balanced Accuracy improved across all four emotion labels.

-   The average improvement was **0.04135±0.01649**.

* * * * *

### Overall Model Performance Ranking

The **Random Forest** model had the best classification scores, with the **XGBoost** model coming in a close second. The **MLP** model performed the worst by a significant and upsetting margin.

## Conclusion
Class weights only improved performance in XGBoost, while harming Random Forest and MLP. Ensemble models (RF and XGB) were far more effective than MLP on the DEAP features, with Random Forest best in raw performance and XGBoost best in leveraging weights to handle imbalance. Thus, both model type and the use of class weights strongly influence classification performance, but their effects are algorithm-dependent.
