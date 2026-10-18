# Diabetes Classification: Traditional ML vs Deep Learning

Summative Assignment (Model Training and Evaluation).

Author: Isimbi Nelly Assoumpta

## Links
- Google Drive (raw dataset zip, saved models, W&B logs): (https://drive.google.com/drive/folders/12PAfjZ8g_ynu3QIyrv6pwaOFz8rt78iS?usp=sharing)
- Video presentation: ((https://youtu.be/jdc9BHjgtxg))
- Report: submitted as a PDF on Canvas

## What this project does
I predict whether a person has diabetes from 21 survey answers about health and lifestyle. I compare traditional machine learning (scikit-learn) with deep learning (TensorFlow/Keras, using tf.data) in 9 experiments (plus one variant), log every run with Weights & Biases in offline mode, and evaluate the two best models on a held-out test set.

## Dataset
CDC Diabetes Health Indicators, built from the 2015 BRFSS survey (253,680 rows, 21 features, binary target `Diabetes_binary`). I downloaded it from Kaggle (search "Diabetes Health Indicators Dataset") and used the file `diabetes_binary_health_indicators_BRFSS2015.csv`.

- 24,206 duplicate rows were removed before splitting, leaving 229,474 rows with 15.3% positive cases.
- No missing values were found.
- Split (stratified, seed 42): 160,631 train / 34,421 validation / 34,422 test, with 15.3% positives in each part.
- BMI was clipped to the range 18.0 to 50.0 (1st and 99th percentiles of the training set), and features were standardized using the training set only.

The raw data is NOT in this repository. A zipped copy is in the Drive folder linked above.

## How to run the notebook
1. Open `Isimbi_diabetes_classification_summative.ipynb` in Google Colab.
2. Upload the zipped dataset to your own Google Drive and set `ZIP_PATH` in the second code cell to its location.
3. Choose Runtime > Run all (about 15 to 20 minutes). The notebook mounts Google Drive and saves models and logs there.
4. To run locally instead, install the libraries in `requirements.txt` and replace the Drive paths. Random seeds are fixed (seed 42).

## Notebook structure
1. Setup and data loading
2. Cleaning and splitting (duplicates removed before the split, BMI clipped, scaler fit on train only)
3. Data exploration
4. Experiment tracking helper (W&B offline)
5. Traditional ML: Logistic Regression, Random Forest, tuned HistGradientBoosting, feature engineering (E1 to E5)
6. Deep learning: Sequential MLP, regularized MLP with class weights, Functional API with a residual block (E6 to E8b)
7. Threshold tuning and final test evaluation (E9)
8. Error analysis
9. Learning curve
10. Requirements and full experiment table

## Experiment results
E1 to E8b are evaluated on the validation set at the default threshold of 0.5. E9a and E9b are evaluated on the test set at thresholds tuned on the validation set.

| Exp | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Time (s) |
|---|---|---|---|---|---|---|---|---|
| E1 | Logistic Regression baseline | 0.8532 | 0.5668 | 0.1710 | 0.2627 | 0.8138 | 0.4312 | 0.7 |
| E2 | Logistic Regression, balanced class weights | 0.7189 | 0.3233 | 0.7667 | 0.4548 | 0.8141 | 0.4302 | 1.6 |
| E3 | Random Forest (150 trees, depth 14) | 0.7309 | 0.3334 | 0.7597 | 0.4634 | 0.8177 | 0.4429 | 48.3 |
| E4 | HistGradientBoosting, tuned (random search) | 0.7091 | 0.3189 | 0.7945 | 0.4551 | 0.8196 | 0.4467 | 190.5 |
| E5 | HistGradientBoosting + 8 engineered features | 0.7098 | 0.3192 | 0.7922 | 0.4550 | 0.8200 | 0.4460 | 11.9 |
| E6 | MLP baseline (Sequential, 64-32-1) | 0.8550 | 0.5954 | 0.1619 | 0.2545 | 0.8192 | 0.4439 | 17.0 |
| E7 | MLP 128-64-32 + BatchNorm, Dropout, L2, class weights | 0.6982 | 0.3127 | 0.8127 | 0.4516 | 0.8195 | 0.4447 | 77.8 |
| E8 | Functional API, residual block, LR schedule | 0.7012 | 0.3145 | 0.8081 | 0.4527 | 0.8192 | 0.4443 | 75.5 |
| E8b | E8 with the 8 engineered features | 0.7037 | 0.3159 | 0.8041 | 0.4536 | 0.8194 | 0.4460 | 103.2 |
| E9a | **Test:** HistGB (E4), threshold 0.631 | 0.7883 | 0.3815 | 0.6182 | 0.4719 | 0.8159 | 0.4377 | n/a |
| E9b | **Test:** Neural net (E8b), threshold 0.646 | 0.7970 | 0.3923 | 0.5958 | 0.4731 | 0.8171 | 0.4389 | n/a |

Training time for E4 includes the whole random search (12 configurations x 3 folds), not a single fit. The best settings found were learning rate 0.03, max depth 5, max leaf nodes 63, min samples per leaf 20, and L2 regularization 5.0 (best cross-validated PR-AUC 0.4315).

### Default vs tuned threshold (test set)
| Model | Threshold | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|---|
| HistGB | 0.5 (default) | 0.315 | 0.791 | 0.451 | 0.705 |
| HistGB | 0.63 (tuned) | 0.382 | 0.618 | 0.472 | 0.788 |
| Neural net | 0.5 (default) | 0.313 | 0.801 | 0.450 | 0.700 |
| Neural net | 0.65 (tuned) | 0.392 | 0.596 | 0.473 | 0.797 |

### Confusion matrices (test set, tuned thresholds)
| Model | True negatives | False positives | False negatives | True positives |
|---|---|---|---|---|
| HistGB | 23,881 | 5,276 | 2,010 | 3,255 |
| Neural net | 24,297 | 4,860 | 2,128 | 3,137 |

## Main findings
- All models reach a similar ranking quality (ROC-AUC 0.814 to 0.820, PR-AUC 0.430 to 0.447 on validation). Gradient boosting and the neural network are equivalent on the test set (PR-AUC 0.438 vs 0.439).
- Accuracy is misleading here: E1 and E6 reach about 85% accuracy but find only 16% to 17% of the people with diabetes. Class weights raise recall to 0.77 to 0.81 but lower precision, without improving the ranking quality (AUC).
- Hyperparameter tuning, engineered features and architecture changes made differences of about 0.002 PR-AUC or less, which is within noise. This suggests the limit comes from the information in the survey features and not from model capacity.
- Recall varies by group: it is much higher for BMI 35 and above than for BMI under 25, and higher for people aged 65+ than for ages 18 to 44.

## Saved models and logs
Stored in the Drive folder: a joblib file for the best traditional model (HistGradientBoosting plus its threshold), an h5 file for the best deep learning model, a joblib file with the scaler and threshold for the neural network, `experiment_results.csv`, `requirements.txt`, and the Weights & Biases offline logs for all 11 runs.

## Limitations
The data is self-reported, comes from a single survey year (2015), has no laboratory values, and is imbalanced. The target means a doctor told the person they have diabetes, so undiagnosed cases are labelled negative.
