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

## Run

From inside `step_prediction`:

```bash
python evaluate.py
```
