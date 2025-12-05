import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#configs data
data = [
    ("cls", 512, "1e-5", 0.2,  9.4,  8.4),
    ("cls", 512, "1e-5", 0.1,  8.2,  7.4),
    ("cls", 512, "2e-5", 0.1,  7.8,  8.6),
    ("cls", 512, "2e-4", 0.1, -3.2, -6.4),
    ("cls", 512, "3e-5", 0.1,  1.8, -1.4),
    ("cls", 512, "2e-4", 0.2,  0.0,  0.0),
    ("cls", 512, "2e-3", 0.2, -10.0, 0.0),
    ("cls", 512, "2e-3", 0.1, -10.0, 0.0),
    ("cls", 512, "2e-5", 0.2,  2.0, -2.2),
    ("cls", 512, "2e-5", 0.2,  2.0, -2.2),
    ("cls", 512, "3e-5", 0.2,  1.8, -2.4),
]

df = pd.DataFrame(data,columns=["pooling", "max_len", "lr", "dropout", "dval", "dtest"])
lr_order = ["1e-5", "2e-5", "3e-5", "2e-4", "2e-3"]
pivot = (df.pivot_table(index="dropout", columns="lr", values="dval", aggfunc="mean").reindex(index=sorted(df["dropout"].unique())).reindex(columns=lr_order))

#adds the asterisks to the values
annot_labels = pivot.applymap(
    lambda v: (
        "" if pd.isna(v)
        else f"{int(v)}*" if v == -10
        else f"{v:.2f}"
    )
)

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(pivot,annot=annot_labels,fmt="", cmap="Reds",linewidths=0.5, cbar_kws={"label": "ΔVal (pp)"})
plt.title("Dropout vs Learning Rate (ΔVal)")
plt.xlabel("Learning Rate")
plt.ylabel("Dropout")
plt.tight_layout()
plt.show()