# Architecture

```mermaid
flowchart LR
  A[Raw CSV / KaggleHub] --> B[DataLoader: target + feature selection]
  B --> C[Train/Test Split]
  C --> D[Pipeline: Impute + Encode + Scale]
  D --> E[Model Training + (Optional) CV + Tuning]
  E --> F[Evaluation: ROC/PR, Confusion Matrix, Thresholding]
  E --> G[Scoring: PD Predictions]
  G --> H[Portfolio Segmentation: Risk Bands]
  H --> I[Outputs: CSV + Plots + Reports]
```

