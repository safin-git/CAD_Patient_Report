# Cardio Care AI model card

The risk score starts from a machine-learning model trained on the Z-Alizadeh Sani dataset (extended version). The model predicts whether coronary artery disease (CAD) was confirmed on angiography. It does not predict heart attack recurrence.

The model uses sixteen inputs: age, sex, BMI, diabetes, hypertension, current smoking, family history of CAD, typical chest pain, breathlessness, systolic blood pressure, heart rate, LDL cholesterol, HDL cholesterol, triglycerides, hemoglobin and left ventricular ejection fraction.

Several algorithms are compared with cross-validation and the one with the highest ROC-AUC is kept. It is then checked once on a held-out test split. Missing numeric inputs are filled with the training median, and yes/no findings that are not reported count as absent.

Parameters that are not in the dataset, such as troponin, calcium score, QTc, HRV, stenosis and BNP, are scored by hand-written rules and added on top of the model probability. These rules are illustrative and have not been clinically validated.

The score is a screening estimate, not a diagnosis. A doctor must confirm any finding.
