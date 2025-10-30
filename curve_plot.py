import matplotlib.pyplot as plt
import pandas as pd

# 读取两个 CSV
filtered = pd.read_csv("curve_success_filtered.csv")
okg = pd.read_csv("curve_success_okg.csv")

# 绘图
plt.figure(figsize=(6,4))
plt.plot(filtered["budget"], filtered["acc"], label="Filtered (per-question top-B)", marker="o", markersize=3)
plt.plot(okg["budget_t"], okg["acc"], label="OKG (sequential)", marker="s", markersize=3)

# 图形样式
plt.xlabel("Budget")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Budget")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图片
plt.savefig("accuracy_vs_budget_200.png", dpi=200)
plt.show()
