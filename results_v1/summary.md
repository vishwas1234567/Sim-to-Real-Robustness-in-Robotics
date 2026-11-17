# Zero-shot robustness summary (numpy backend)

Cells: mean final error / success rate (<5 cm), averaged over seeds and 100 fixed episodes.

## mass

| value | Nominal-trained (no DR) | Domain-randomized |
|---|---|---|
| 0.5 | 2.1 cm / 99% | 4.9 cm / 57% |
| 0.75 | 1.6 cm / 100% | 4.5 cm / 59% |
| 1.0 | 1.6 cm / 100% | 4.2 cm / 64% |
| 1.25 | 1.6 cm / 100% | 4.0 cm / 68% |
| 1.5 | 3.4 cm / 96% | 3.8 cm / 72% |
| 1.75 | 6.0 cm / 88% | 3.7 cm / 72% |
| 2.0 | 15.7 cm / 69% | 6.1 cm / 67% |

## friction

| value | Nominal-trained (no DR) | Domain-randomized |
|---|---|---|
| 0.25 | 4.5 cm / 95% | 4.2 cm / 64% |
| 0.5 | 2.1 cm / 98% | 4.2 cm / 64% |
| 1.0 | 1.6 cm / 100% | 4.2 cm / 64% |
| 2.0 | 1.6 cm / 100% | 4.2 cm / 64% |
| 3.0 | 1.7 cm / 99% | 4.2 cm / 64% |
| 4.0 | 2.3 cm / 98% | 4.2 cm / 64% |

## delay

| value | Nominal-trained (no DR) | Domain-randomized |
|---|---|---|
| 0 | 1.6 cm / 100% | 4.2 cm / 64% |
| 1 | 7.7 cm / 62% | 3.6 cm / 77% |
| 2 | 29.8 cm / 5% | 4.8 cm / 66% |
| 3 | 58.4 cm / 0% | 11.3 cm / 0% |
| 4 | 77.6 cm / 0% | 25.6 cm / 0% |
| 5 | 81.0 cm / 0% | 51.4 cm / 0% |
| 6 | 90.4 cm / 0% | 71.0 cm / 0% |
| 8 | 90.0 cm / 0% | 86.0 cm / 0% |
