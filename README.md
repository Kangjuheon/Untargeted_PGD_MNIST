# PGD Untargeted Attack on MNIST

This project implements **PGD (Projected Gradient Descent)** untargeted adversarial attacks on a simple CNN trained with the **MNIST dataset**.

## What it does

- Trains a CNN on MNIST
- Applies **untargeted PGD attack** to the test set
- Compares **accuracy before and after the attack**

## PGD Parameters

- `eps = 0.3`
- `eps_step = 0.03`
- `k = 10` iterations

## How to run

```bash
pip install -r requirements.txt
python test.py
```
## Example output
```bash
Epoch 1, Loss: 0.1428
Epoch 2, Loss: 0.0413
Epoch 3, Loss: 0.0260

[Clean Accuracy] 98.85%
[PGD Untargeted Attack Accuracy] eps=0.3, k=10 → 0.08%
```

## Notes
- The model is trained from scratch each time.
- The attack is untargeted, aiming to cause any misclassification.
