# Calibrated item-weighting model vs. fixed-cutoff child screening rule

## Motivation
The paper "icare-adhd: A mobile application prototype for early child attention deficit hyperactivity disorder" (2018) describes a questionnaire-based mobile screening tool for children. Questionnaire screeners typically add up item answers and apply a fixed cutoff. This small stand-in experiment asks whether a learned, calibrated model can match labelled screening outcomes better than such a rule. It is not a re-implementation of the paper.

## Question
On a public child questionnaire screening dataset, can a calibrated, cross-validation-tuned model that learns item weights beat the standard fixed score-cutoff screening rule at matching labelled screening outcomes?

## Dataset
UCI "Autistic Spectrum Disorder Screening Data for Children" (ucimlrepo id 419): 10 yes/no items plus age, sex, jaundice and family history. This is **not** the paper's dataset (ADHD screening data with clinical labels is not freely available). Stratified 70/30 train/test split, seed 42.

## Baseline vs. improved
- **Baseline:** sum the 10 item answers and flag a child when the sum is 6 or more.
- **Improved:** class-balanced logistic regression and LightGBM, each tuned by cross-validation on the training set, sigmoid-calibrated, with probabilities averaged. The decision threshold is chosen from training out-of-fold predictions. Run with seeds 42, 43 and 44.

The main metric is positive-class F1 on the test set. Secondary metrics are sensitivity and ROC-AUC.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the public UCI child ASD screening dataset (88-case test set), a standard fixed score-cutoff rule (item sum ≥ 6) reached 88% positive-class F1, while my calibrated, CV-tuned LR+LightGBM model reached 98% (±0.5 points across 3 seeds) on one small split.

| Metric | Value |
|---|---|
| baseline_f1 | 0.8842 |
| improved_f1 | 0.9843 |
| improved_f1_std | 0.005351 |
| baseline_sensitivity | 1 |
| improved_sensitivity | 0.9921 |
| improved_sensitivity_std | 0.01122 |
| baseline_auc | 1 |
| improved_auc | 0.9998 |
| improved_auc_std | 0.000244 |
| baseline_score | 0.8842 |
| improved_score | 0.9843 |
| baseline_score_std | 0 |
| improved_score_std | 0.005351 |
| improvement | 0.1001 |
| mean_chosen_threshold | 0.5083 |

Run on 2026-10-23. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Small dataset, so the test set is small and differences are noisy; single split. Labels are ASD (not ADHD) and may be closely tied to the item score, so the cutoff rule can be hard to beat. The 'result' column (item sum) was excluded from model features. Baseline is deterministic (std 0); improved std is over seeds 42-44 on the same split. Hyper-parameters are tuned on train only, and the threshold is chosen from train out-of-fold predictions after tuning on the same train data (mild optimism confined to train).

## Reference

This project is motivated by the following paper:

> J Mitrpanont, B Bousai (2018). *icare-adhd: A mobile application prototype for early child attention deficit hyperactivity disorder*. seventh ICT. https://ieeexplore.ieee.org/abstract/document/8523973/

Paper author: Prof. Assoc. Prof. Dr. Jarernsri Mitrpanont, Mahidol University (jarernsri.mit@mahidol.ac.th)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
