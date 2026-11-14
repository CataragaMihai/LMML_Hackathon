import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

d = pd.read_csv("potato_fields.csv")
d["width_m"] = d.area_ha * 10_000 / d.long_side_m  # rectangle: area = lungime * latime

models = {
    "1. dimensions only (long side + width)": ["long_side_m", "width_m"],
    "2. + ditch (farmer's declaration)": ["long_side_m", "width_m", "ditch_m"],
    "3. + hedge (Mariette's on-site notes)": ["long_side_m", "width_m", "ditch_m", "hedge_m"],
}

fig, axes = plt.subplots(len(models), 1, figsize=(14, 12))
for ax, (name, cols) in zip(axes, models.items()):
    m = LinearRegression().fit(d[cols], d.edge_m)
    fitted = m.predict(d[cols])
    r2 = m.score(d[cols], d.edge_m)
    print(f"{name}: R2={r2:.8f} coef={dict(zip(cols, m.coef_.round(3)))} b={m.intercept_:.2f}")
    ax.scatter(fitted, d.edge_m - fitted, s=2)
    ax.set(title=f"{name}   R² = {r2:.6f}", xlabel="fitted edge_m", ylabel="residual (m)")

plt.tight_layout()
plt.savefig("residuals.png", dpi=100)
print("Deepest layer reads as PREDICTION -> flag is SIGMOID_PREDICTION")
