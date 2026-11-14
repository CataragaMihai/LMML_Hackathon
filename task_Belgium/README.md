# Belgium · Fry-ndship

**Flag recovered (correction):** `SIGMOID_SUPERVISION`

## Run
```bash
cd task_Belgium && python main.py   # prints R² per model, writes residuals.png
```

## The idea
A high R² doesn't mean the model is right. The residuals still have structure. Fit a linear model for `edge_m` and plot residuals against fitted values. Each time you add the missing term, the residual plot spells a new word at a smaller scale:

| Model (linear, no extra terms) | R² | Residual scale | Word |
|---|---|---|---|
| long side + width (`width = area·10⁴ / long_side`) | 0.9984 | +- 15 m | REGRESSION |
| + `ditch_m` (declared) | 0.99998 | +-1.5 m | **SUPERVISION** |
| + `hedge_m` (Mariette's on-site notes) | 0.99999994 | +-0.1 m | PREDICTION (wrong) |

The co-op has to estimate edges for fields nobody surveyed, so the model may only use what farmers declare: area, long side and ditch. That model is `edge ≈ 0.9·(long + width) + ditch`, and its residuals spell **SUPERVISION**, nor PREDICTION. `hedge_m` is only noted on site by Mariette, so it doesn't exist for unsurveyed fields. Adding it is leakage: R² gets better, but the model can't be used where it's needed, and its residuals spell the "decoy" PREDICTION. REGRESSION is the decoy from an under-fitted model that leaves out the ditch.

![residuals](residuals.png)
