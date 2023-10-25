# CDISC_Decision-trees
📊 CDISC Variable Mapping using Decision trees


# Classification Metrics Explanation

These metrics are commonly used to evaluate the performance of classification models. Below is a breakdown of each metric:

## 1. Precision
- **Definition**: Precision is the ratio of correctly predicted positive observations to the total predicted positives.
- **Formula**: 
  \[ Precision = \frac{True Positives}{True Positives + False Positives} \]
- **Interpretation**: Precision provides insights into the accuracy of positive predictions. A high precision suggests a low false positive rate. For instance, if the precision for a class is `0.9`, this means 90% of the instances that the model labeled as belonging to that class truly belong to that class, whereas 10% are misclassifications.

## 2. Recall (or Sensitivity or True Positive Rate)
- **Definition**: Recall quantifies the ratio of correctly predicted positive observations to all the observations in the actual class.
- **Formula**: 
  \[ Recall = \frac{True Positives}{True Positives + False Negatives} \]
- **Interpretation**: Recall conveys the extent to which our model identifies actual positives by labeling them as positive. A recall value of `1` implies the model correctly predicted all positives without any misses.

## 3. F1-Score
- **Definition**: The F1-Score represents the harmonic mean of precision and recall, serving as a singular metric that merges the effects of both precision and recall.
- **Formula**: 
  \[ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} \]
- **Interpretation**: The F1-score peaks at `1` (indicating perfect precision and recall) and can plummet to `0`. It proves especially handy when the class distribution is skewed. A superior F1-score signals a harmonious balance between precision and recall.

## 4. Support
- **Definition**: Support counts the actual occurrences of the class in the dataset.
- **Interpretation**: This metric elucidates the quantity of instances of each class in the evaluated dataset. For instance, a support value of `5` for a class indicates the test dataset contains 5 instances of that class. If support is particularly low for a given class, the precision, recall, and F1-score for that class may be less reliable due to their calculations being founded on minimal instances.

In classification reports, metrics are generally presented for every class as well as in the form of a weighted average (or macro average) spanning all classes.
