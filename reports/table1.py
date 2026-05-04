from pathlib import Path

from docx import Document

doc = Document()

# Header
headers = [
    "Modeling Paradigm",
    "Common Method",
    "Structural Assumption",
    "Predictive Output",
    "Key UQ Limitation",
    "Vulnerability under Non-Stationarity"
]

# 4 rows (1 header + 3 data) × 6 columns
table = doc.add_table(rows=4, cols=6)
table.style = "Table Grid"

for i, h in enumerate(headers):
    table.cell(0, i).text = h

# Data rows
data = [
    [
        "Classical Econometrics",
        "GARCH, ARCH",
        "Rigid distribution (Gaussian/Student-t)",
        "Volatility Parameter (σ_t)",
        "Reliant on assumed residual distribution; fails under fat tails.",
        "High. Misspecifies variance during abrupt structural breaks."
    ],
    [
        "Standard Deep Learning",
        "LSTM, GRU",
        "Flexible (Non-linear sequence mapping)",
        "Deterministic Point Forecast (ŷ_{t+1})",
        "No inherent risk measurement; tends to be overconfident "
        "(Gal & Ghahramani, 2016).",
        "Severe. Degrades into overconfident guesswork without "
        "reliable risk guardrails."
    ],
    [
        "Trustworthy DL (This Study)",
        "CP-LSTM / CP-GRU",
        "Minimal (Assumes bounded residuals, distribution-free)",
        "Probabilistic Prediction Interval [L_t, U_t]",
        "Balances coverage vs. width; explicit "
        "efficiency/reliability trade-off.",
        "Mitigated. Adaptive CP methods update intervals dynamically."
    ]
]

for r, row in enumerate(data, start=1):
    for c, val in enumerate(row):
        table.cell(r, c).text = val

output_path = Path(__file__).parent / "thesis_table.docx"
doc.save(str(output_path))
print(f"Table saved to: {output_path}")
