# Zero-shot robustness summary (numpy backend)

Cells: mean final error / success rate (<5 cm), averaged over seeds and 100 fixed episodes.

## mass

| value | Nominal-trained (no DR) | Domain-randomized |
|---|---|---|
| 0.5 | 2.1 cm / 99% | 3.5 cm / 83% |
| 0.75 | 1.6 cm / 100% | 2.8 cm / 97% |
| 1.0 | 1.6 cm / 100% | 2.8 cm / 99% |
| 1.25 | 1.6 cm / 100% | 2.9 cm / 98% |
| 1.5 | 3.4 cm / 96% | 3.0 cm / 94% |
| 1.75 | 6.0 cm / 88% | 3.0 cm / 93% |
| 2.0 | 15.7 cm / 69% | 5.9 cm / 86% |

## friction

| value | Nominal-trained (no DR) | Domain-randomized |
|---|---|---|
| 0.25 | 4.5 cm / 95% | 2.8 cm / 98% |
| 0.5 | 2.1 cm / 98% | 2.8 cm / 98% |
| 1.0 | 1.6 cm / 100% | 2.8 cm / 99% |
| 2.0 | 1.6 cm / 100% | 2.8 cm / 98% |
| 3.0 | 1.7 cm / 99% | 2.7 cm / 99% |
| 4.0 | 2.3 cm / 98% | 2.7 cm / 99% |

## delay

| value | Nominal-trained (no DR) | Domain-randomized |
|---|---|---|
| 0 | 1.6 cm / 100% | 2.8 cm / 99% |
| 1 | 7.7 cm / 62% | 2.5 cm / 99% |
| 2 | 29.8 cm / 5% | 4.3 cm / 77% |
| 3 | 58.4 cm / 0% | 8.2 cm / 0% |
| 4 | 77.6 cm / 0% | 20.4 cm / 0% |
| 5 | 81.0 cm / 0% | 47.4 cm / 0% |
| 6 | 90.4 cm / 0% | 74.3 cm / 0% |
| 8 | 90.0 cm / 0% | 88.5 cm / 0% |
