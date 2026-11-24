# Student Performance Report

## Objective and data

This study predicts the UCI Student Performance final grade, `G3`, using
recorded demographic, family, school, attendance, and study-habit variables.
The Math-course file was used because it is a single, clearly defined cohort.
The source documentation defines `G3` on a 0-20 scale. The earlier grades
`G1` and `G2` were removed before modeling to satisfy the project requirement
and to avoid target leakage.

The cleaned dataset contains **395 rows** and **35 input
features**. Exact duplicate rows were removed; missing input values are handled
inside the training pipeline.

## Preprocessing and modeling choices

The data was split into training and test sets before fitting any imputer,
scaler, or encoder. Numeric variables use median imputation and standardization.
Categorical variables use most-frequent imputation and one-hot encoding.
Unknown categories are ignored at prediction time.

Ridge regression provides a stable linear baseline for one-hot encoded data.
Random forest and boosting models provide nonlinear comparisons. The final
model is a simple average ensemble of a tuned random forest and histogram
gradient boosting model. Additional
predictor-only features summarize parent education, alcohol consumption,
support indicators, social time, and the skewed absence count. Hyperparameters
were selected with a shuffled five-fold `KFold` on training data only, using
negative RMSE. The final test set was not used for tuning.

## Results

| model                                | split   |   mae |   rmse |    r2 |
|:-------------------------------------|:--------|------:|-------:|------:|
| Ridge baseline                       | train   | 2.879 |  3.704 | 0.347 |
| Ridge baseline                       | test    | 3.436 |  4.205 | 0.138 |
| Random forest baseline               | train   | 1.097 |  1.474 | 0.897 |
| Random forest baseline               | test    | 2.994 |  3.803 | 0.295 |
| Gradient boosting baseline           | train   | 1.520 |  1.985 | 0.812 |
| Gradient boosting baseline           | test    | 3.162 |  3.964 | 0.234 |
| Histogram gradient boosting baseline | train   | 1.322 |  1.708 | 0.861 |
| Histogram gradient boosting baseline | test    | 3.037 |  3.761 | 0.310 |
| Tuned random forest                  | train   | 1.958 |  2.612 | 0.675 |
| Tuned random forest                  | test    | 3.098 |  3.836 | 0.282 |
| Ensemble model                       | train   | 1.736 |  2.275 | 0.754 |
| Ensemble model                       | test    | 3.104 |  3.788 | 0.300 |

The selected model is **Ensemble model**, chosen using cross-validation on the
training data. Its untouched test-set performance was:
**3.788 grade points**, with MAE **3.104** and
R-squared **0.300**. Because grades are noisy and the sample is modest,
these figures should be interpreted as approximate predictive performance rather
than proof that the model will generalize to every school or course.

## Patterns useful for support

The exploratory plots compare grades with study time, previous failures,
absences, school support, and higher-education aspiration.
The model's largest permutation-importance features in this run were:
failures, absences, sex, goout, famsup, parent_education_avg, age, reason, schoolsup, freetime.

These patterns can help staff ask better follow-up questions about attendance,
study conditions, prior difficulties, and support needs. They should not be used
to label students, deny opportunities, or claim that a single characteristic
causes a low grade.

## Limitations

- The data is observational and cannot establish causation.
- It comes from two Portuguese schools and one historical setting.
- The target is an ordinal-like grade, but this project treats it as a numeric
  regression target to match the requested final-grade prediction.
- Self-reported variables may contain measurement error.
- The dataset is relatively small and may not represent other schools,
  countries, subjects, or years.
- `G1` and `G2` are unavailable by design, so predictions are less informed than
  an end-of-term forecasting model that is allowed to use them.

## Team contributions

Replace the placeholders below with the actual member names and contributions
before submission:

- Member 1: [name] - [contribution]
- Member 2: [name] - [contribution]
- Member 3: [name] - [contribution]

## Reproducibility

Run `python student_performance_project.py` from this project directory after
installing `requirements.txt`. All generated metrics, plots, the cleaned data,
and the fitted model are saved under the documented output paths.
