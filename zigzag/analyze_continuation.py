import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("continuation_chain.csv")

# =========================
# 1. tumor_logit vs step
# =========================
agg = df.groupby("step_idx")["tumor_logit"].agg(["mean", "std"]).reset_index()

plt.figure()
plt.plot(agg["step_idx"], agg["mean"], marker="o")
plt.fill_between(
    agg["step_idx"],
    agg["mean"] - agg["std"],
    agg["mean"] + agg["std"],
    alpha=0.2
)

plt.xlabel("Step")
plt.ylabel("Tumor Logit")
plt.title("Continuation progression")
plt.savefig("plot_tumor_vs_step.png")
plt.close()

# =========================
# 2. Continuation vs Restart
# =========================

decoded = pd.read_csv("decoded_progress_identity.csv")

# Restart: gleiche Anzahl Samples wie steps
restart = (
    decoded.groupby("cell_idx")
    .apply(lambda g: g.sample(n=5, replace=False))
)

restart_score = restart.groupby("cell_idx")["tumor_logit"].max()

# Continuation: letzter Schritt
max_step = df["step_idx"].max()
cont = df[df["step_idx"] == max_step]

cont_score = cont.groupby("cell_idx")["tumor_logit"].max()

print("Restart mean:", np.mean(restart_score))
print("Continuation mean:", np.mean(cont_score))

# Plot
plt.figure()
plt.boxplot([restart_score, cont_score], labels=["Restart", "Continuation"])
plt.ylabel("Tumor Logit")
plt.title("Continuation vs Restart")
plt.savefig("plot_cont_vs_restart.png")
plt.close()