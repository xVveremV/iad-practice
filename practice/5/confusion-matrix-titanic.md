|     |    |
|:---:|:--:|
| 118 | 9  |
| 20  | 32 |

|    |    |
|:--:|:--:|
| TN | FP |
| FN | TP |

- TN (True Negatives) → top-left (118)
  Number of cases where the true label was 0 and the model predicted 0.
 
- FP (False Positives) → top-right (9)
  Number of cases where the true label was 0 but the model predicted 1.
  (“false alarm”)

- FN (False Negatives) → bottom-left (20)
  Number of cases where the true label was 1 but the model predicted 0.
  (“missed positive”)

- TP (True Positives) → bottom-right (32)
  Number of cases where the true label was 1 and the model predicted 1.