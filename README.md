# Adversarial IDS Project

This project builds a machine-learning intrusion detection system, evaluates it under adversarial evasion attacks, applies feasibility constraints, tests black-box attacks, evaluates defenses, and generates a final security assessment report.

Main report:
reports\final_report.md

Key outputs:
results\metrics\baseline_metrics.json
results\metrics\feature_feasibility_table.csv
results\metrics\phase3_whitebox_full_summary.csv
results\metrics\phase4a_transfer_full_summary.csv
results\metrics\phase4b_hopskipjump_full_summary.csv
results\metrics\final_defense_comparison_table.csv

Main models:
models\baseline\pytorch_clean_baseline.pt
models\baseline\xgboost_clean_baseline.json
models\defense\pytorch_adversarial_training_defense.pt
