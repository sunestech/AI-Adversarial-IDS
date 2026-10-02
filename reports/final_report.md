# Adversarial Machine Learning Assessment of an IDS Classifier

Generated: 2026-09-28T23:36:46

## 1. Executive Summary

This project built and evaluated a machine-learning intrusion detection system under adversarial evasion conditions. The baseline IDS performed strongly on clean data, but unconstrained white-box attacks were able to evade the PyTorch model at very high rates. When perturbations were restricted to a conservative set of operationally plausible network-flow features, evasion dropped substantially, showing that feasibility constraints materially change the security interpretation.

At epsilon 0.100, unconstrained PGD evasion reached 99.67%, while constrained PGD evasion was 48.19%. The black-box transfer attack produced 41.67% PyTorch transfer evasion and 13.35% XGBoost transfer evasion at epsilon 0.100. HopSkipJump produced raw adversarial examples, but projected constrained evasion was 0.00% at epsilon 0.100. Adversarial training was the strongest tested defense, reducing constrained FGSM and PGD evasion to the low single digits while preserving high attack recall.

## 2. Threat Model

The attacker goal is evasion: real attack traffic is modified at inference time so that the IDS predicts it as benign. The project evaluated three levels of attacker knowledge:

- White-box: the attacker has access to model architecture, weights, and gradients.
- Black-box transfer: the attacker trains a separate surrogate model and transfers adversarial examples to the protected model.
- Decision-based black-box: the attacker only observes predicted labels from the target model.

The attacker is not allowed to change arbitrary features. Network-flow features must remain within scaled bounds and, for constrained attacks, only attacker-controllable features may be perturbed.

## 3. System Under Test

The protected IDS system includes two clean baseline models:

- PyTorch neural-network IDS used as the differentiable white-box target.
- XGBoost IDS used as a stronger non-neural comparison model.

The PyTorch baseline metrics were:

- Clean accuracy: 98.04%
- Attack precision: 99.07%
- Attack recall: 98.43%
- Attack F1: 98.75%
- False positive rate: 3.39%

The XGBoost baseline metrics were:

- Clean accuracy: 99.93%
- Attack precision: 99.93%
- Attack recall: 99.98%
- Attack F1: 99.96%
- False positive rate: 0.25%

## 4. Dataset and Preprocessing

The modeling pipeline used the supplied MachineLearningCSV flow records. The processed dataset was split into training, validation, and test partitions. The fitted preprocessing objects were frozen after baseline training.

Preprocessing controls:

- Median imputation fitted on training data only.
- MinMax scaling fitted on training data only.
- Test set preserved as the downstream reference set.
- Scaled feature range clipped to [0, 1].
- Baseline artifacts frozen with SHA256 manifests.

## 5. Attack Surface and Feasibility Analysis

The feasibility analysis classified 78 model input features. Only 5 were allowed to change during constrained attacks:

- Flow Duration
- Total Fwd Packets
- Total Length of Fwd Packets
- Init_Win_bytes_forward
- act_data_pkt_fwd

The remaining 73 features were frozen because they were derived, server-side, context-dependent, protocol-state dependent, or not included in the conservative attacker-controllable set.

This distinction is central to the assessment. Unconstrained attacks show mathematical vulnerability. Constrained attacks provide the more meaningful operational security result.

## 6. White-Box Attack Results

White-box attacks were run against the PyTorch IDS using FGSM and PGD across multiple epsilon values.

At epsilon 0.100:

- FGSM unconstrained evasion: 99.19%
- FGSM constrained evasion: 46.42%
- PGD unconstrained evasion: 99.67%
- PGD constrained evasion: 48.19%

The gap between unconstrained and constrained results shows that many mathematically successful adversarial perturbations depend on changing features outside the conservative attacker-controllable set.

## 7. Black-Box Attack Results

### Transfer Attack

The transfer attack trained an independent surrogate network, crafted constrained PGD examples against the surrogate, and submitted the same examples to the protected PyTorch and XGBoost models.

At epsilon 0.100:

- PyTorch transfer evasion: 41.67%
- XGBoost transfer evasion: 13.35%

This confirms that some adversarial risk transfers across models even without direct access to protected-model weights.

### HopSkipJump Decision-Based Attack

HopSkipJump generated raw decision-based adversarial examples. After projecting the examples back into the constrained feature space, projected evasion at epsilon 0.100 was:

- HopSkipJump projected evasion: 0.00%

This result indicates that the raw decision-based attack found mathematical adversarial examples, but the conservative feasibility projection removed the evasion effect in the tested configuration.

## 8. Defense Results

Two defenses were tested:

- Feature squeezing
- Adversarial training

The main comparison table is:

| scenario                     |   clean_accuracy |   clean_attack_recall |   false_positive_rate |   fgsm_constrained_evasion_eps_0_100 |   pgd_constrained_evasion_eps_0_100 |   transfer_pytorch_evasion_eps_0_100 |   transfer_xgboost_evasion_eps_0_100 |   hopskipjump_projected_evasion_eps_0_100 | notes                                                          |
|:-----------------------------|-----------------:|----------------------:|----------------------:|-------------------------------------:|------------------------------------:|-------------------------------------:|-------------------------------------:|------------------------------------------:|:---------------------------------------------------------------|
| Baseline PyTorch IDS         |         0.980366 |              0.984268 |             0.0339167 |                            0.464159  |                           0.481854  |                             0.416655 |                             0.133518 |                                         0 | Frozen baseline model                                          |
| Feature squeezing defense    |         0.979508 |              0.98288  |             0.0328333 |                            0.459233  |                           0.477043  |                           nan        |                           nan        |                                       nan | Feature quantization applied before prediction                 |
| Adversarial training defense |         0.977523 |              0.984086 |             0.0465    |                            0.0188513 |                           0.0188513 |                           nan        |                           nan        |                                       nan | Retrained with clean plus constrained PGD adversarial examples |

Adversarial training was the strongest defense in this experiment. It reduced constrained FGSM and PGD evasion at epsilon 0.100 while preserving high attack recall.

## 9. Residual Risk

The IDS remains vulnerable under some adversarial conditions. In particular, constrained white-box attacks and transfer attacks still produced nonzero evasion at larger perturbation budgets. A realistic adversary may also discover additional controllable feature pathways beyond the conservative five-feature threat model used here.

The largest residual concern is not the raw unconstrained evasion rate; it is the portion of evasion that survives feasibility constraints and transfers across model families.

## 10. Limitations

This assessment has several limitations:

- The feasibility model is conservative and may exclude some features a skilled attacker could influence.
- The dataset is static and may not reflect current network behavior.
- Flow-level features are abstractions of packet behavior.
- PCAP files were not available in the supplied PCAP archive; that archive contained checksum files only.
- HopSkipJump was evaluated on a bounded subset because query-based attacks are computationally expensive.
- Defenses were evaluated experimentally, not deployed in a real-time IDS pipeline.

## 11. Future Work

Recommended next steps:

- Extend the feasibility model with packet-level validation.
- Add real PCAP-derived replay or simulation testing.
- Evaluate additional black-box attacks such as ZOO.
- Test combined defenses: adversarial training plus feature squeezing.
- Build a Streamlit demo showing clean, adversarial, and defended predictions.
- Expand reporting by attack family, especially for rare classes.

## 12. Conclusion

The project demonstrates that adversarial evasion is a real risk for machine-learning IDS models, but the magnitude of the risk depends heavily on whether perturbations are operationally feasible. Unconstrained white-box attacks produced severe evasion. Constrained attacks were weaker but still meaningful at larger epsilon values. Black-box transfer attacks also produced measurable evasion. Defensive retraining substantially reduced constrained evasion and provides the strongest mitigation observed in this experiment.
