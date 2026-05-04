import os
import pandas as pd
import matplotlib.pyplot as plt

_REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "reports", "asset_comparison"
)

df = pd.read_csv(os.path.join(_REPORTS_DIR, "summary.csv"))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

ax[0].bar(df["Asset"], df["RMSE"])
ax[0].set_title("RMSE")

ax[1].bar(df["Asset"], df["EnbPI_Coverage"], label="EnbPI")
ax[1].bar(df["Asset"], df["AgACI_Coverage"], alpha=0.5, label="AgACI")
ax[1].axhline(0.9, linestyle="--")
ax[1].set_title("Coverage")

ax[2].bar(df["Asset"], df["AgACI_Width"])
ax[2].set_title("Interval Width")

plt.tight_layout()
plt.savefig(os.path.join(_REPORTS_DIR, "report.png"), dpi=300)
plt.show()