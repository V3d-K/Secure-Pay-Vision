Test set: 56,962 transactions, 98 fraud (0.1720%). Always-predict-legit accuracy: 0.9983

| run | accuracy | precision | recall | f1 | roc_auc | pr_auc | tp | fn | fp |
|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression \| none | 0.9991 | 0.8267 | 0.6327 | 0.7168 | 0.9605 | 0.7414 | 62 | 36 | 13 |
| Logistic Regression \| class_weight | 0.9755 | 0.0610 | 0.9184 | 0.1144 | 0.9721 | 0.7190 | 90 | 8 | 1386 |
| Logistic Regression \| undersample | 0.9708 | 0.0516 | 0.9184 | 0.0977 | 0.9752 | 0.4578 | 90 | 8 | 1655 |
| Logistic Regression \| smote | 0.9741 | 0.0578 | 0.9184 | 0.1088 | 0.9708 | 0.7245 | 90 | 8 | 1467 |
| Random Forest \| none | 0.9996 | 0.9419 | 0.8265 | 0.8804 | 0.9623 | 0.8730 | 81 | 17 | 5 |
| Random Forest \| class_weight | 0.9995 | 0.9610 | 0.7551 | 0.8457 | 0.9572 | 0.8589 | 74 | 24 | 3 |
| Random Forest \| undersample | 0.9661 | 0.0447 | 0.9184 | 0.0852 | 0.9749 | 0.6942 | 90 | 8 | 1925 |
| Random Forest \| smote | 0.9994 | 0.8265 | 0.8265 | 0.8265 | 0.9685 | 0.8760 | 81 | 17 | 17 |
