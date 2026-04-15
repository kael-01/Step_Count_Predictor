# Step_Count_Predictor

Predicts next-day step count based on:
- current-day steps
- sleep in minutes
- screen time in minutes

## Data format

The dataset now uses:
- `day` for sequential day index values such as `1, 2, 3, ...`
- `sleep_minutes` instead of `sleep_hours`
- `screen_minutes` in minutes

## Dataset and split

The project now includes **90 raw daily observations**.

Because the target is **next-day** step count, the model uses a one-day shift:
- features from day `t`
- target from day `t+1`

To keep the split chronological and aligned with the EE requirement:
- **days 1-60** are treated as the training period
- **days 61-90** are treated as the testing period

After target shifting, this becomes:
- **59 training rows** predicting days 2-60
- **30 testing rows** predicting days 61-90

This avoids training on targets from the test period.

## Run

From inside `step_prediction`:

```bash
python evaluate.py
```
