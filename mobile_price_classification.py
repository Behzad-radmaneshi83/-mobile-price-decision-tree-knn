# -*- coding: utf-8 -*-
"""
پروژه درس داده‌کاوی - پیاده‌سازی، تنظیم پارامتر و مقایسه الگوریتم‌های
Decision Tree و KNN
دیتاست: Mobile Price Classification (Kaggle)

نکته: فایل دیتاست را از آدرس زیر دانلود و با نام train.csv در همین پوشه قرار دهید:
https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# ==========================================================
# 1) بارگذاری داده و بررسی اولیه
# ==========================================================
df = pd.read_csv("train.csv")

print("شکل داده:", df.shape)
print(df.head())
print(df.isnull().sum())  # بررسی مقادیر گمشده

# ستون هدف در این دیتاست، price_range است (۴ کلاس: 0,1,2,3)
target_col = "price_range"
X = df.drop(columns=[target_col])
y = df[target_col]

# ==========================================================
# 2) تقسیم داده به آموزش و آزمون (80-20)
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# 3) نرمال‌سازی داده‌ها (به‌خصوص برای KNN ضروری است)
# ==========================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================================
# تابع کمکی برای محاسبه و چاپ معیارهای ارزیابی
# ==========================================================
def evaluate_model(name, y_true, y_pred, plot_cm=True):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    if plot_cm:
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"cm_{name.replace(' ', '_').replace(':', '')}.png")
        plt.close()

    return {"name": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


results = []

# ==========================================================
# 4) مدل Decision Tree - تست پارامترهای مختلف
# ==========================================================
depths = [3, 5, 7, 10, None]
criteria = ["gini", "entropy"]

for crit in criteria:
    for depth in depths:
        model = DecisionTreeClassifier(
            criterion=crit, max_depth=depth, random_state=42
        )
        model.fit(X_train, y_train)  # درخت تصمیم نیازی به نرمال‌سازی ندارد
        y_pred = model.predict(X_test)

        name = f"DecisionTree_{crit}_depth{depth}"
        res = evaluate_model(name, y_test, y_pred, plot_cm=False)
        res["criterion"] = crit
        res["max_depth"] = depth
        results.append(res)

# ==========================================================
# 5) مدل KNN - تست تعداد همسایه‌ها و معیار فاصله
# ==========================================================
k_values = [3, 5, 7, 9, 11, 15]
distance_metrics = ["euclidean", "manhattan"]

for metric in distance_metrics:
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train_scaled, y_train)  # داده نرمال‌شده
        y_pred = model.predict(X_test_scaled)

        name = f"KNN_{metric}_k{k}"
        res = evaluate_model(name, y_test, y_pred, plot_cm=False)
        res["metric"] = metric
        res["k"] = k
        results.append(res)

# ==========================================================
# 6) جدول خلاصه نتایج همه اجراها
# ==========================================================
results_df = pd.DataFrame(results)
results_df.to_csv("all_results_summary.csv", index=False)
print("\n\n===== خلاصه همه نتایج =====")
print(results_df[["name", "accuracy", "precision", "recall", "f1"]])

# ==========================================================
# 7) انتخاب بهترین مدل از هر خانواده و رسم confusion matrix نهایی
# ==========================================================
best_tree_row = results_df[results_df["name"].str.startswith("DecisionTree")].sort_values(
    "f1", ascending=False
).iloc[0]
best_knn_row = results_df[results_df["name"].str.startswith("KNN")].sort_values(
    "f1", ascending=False
).iloc[0]

print("\nبهترین مدل Decision Tree:\n", best_tree_row)
print("\nبهترین مدل KNN:\n", best_knn_row)

# اجرای مجدد بهترین مدل‌ها برای رسم confusion matrix نهایی
best_tree = DecisionTreeClassifier(
    criterion=best_tree_row["criterion"],
    max_depth=best_tree_row["max_depth"],
    random_state=42,
)
best_tree.fit(X_train, y_train)
evaluate_model("Best_DecisionTree", y_test, best_tree.predict(X_test), plot_cm=True)

best_knn = KNeighborsClassifier(
    n_neighbors=int(best_knn_row["k"]), metric=best_knn_row["metric"]
)
best_knn.fit(X_train_scaled, y_train)
evaluate_model("Best_KNN", y_test, best_knn.predict(X_test_scaled), plot_cm=True)

# ==========================================================
# 8) نمودار تاثیر عمق درخت بر دقت (برای تحلیل overfitting)
# ==========================================================
tree_results = results_df[results_df["name"].str.startswith("DecisionTree")]
plt.figure(figsize=(7, 5))
for crit in criteria:
    subset = tree_results[tree_results["criterion"] == crit]
    depths_plot = [str(d) for d in subset["max_depth"]]
    plt.plot(depths_plot, subset["accuracy"], marker="o", label=crit)
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("تاثیر عمق درخت بر دقت مدل")
plt.legend()
plt.tight_layout()
plt.savefig("tree_depth_vs_accuracy.png")
plt.close()

# ==========================================================
# 9) نمودار تاثیر K بر دقت KNN
# ==========================================================
knn_results = results_df[results_df["name"].str.startswith("KNN")]
plt.figure(figsize=(7, 5))
for metric in distance_metrics:
    subset = knn_results[knn_results["metric"] == metric]
    plt.plot(subset["k"], subset["accuracy"], marker="o", label=metric)
plt.xlabel("K (تعداد همسایه‌ها)")
plt.ylabel("Accuracy")
plt.title("تاثیر مقدار K بر دقت مدل KNN")
plt.legend()
plt.tight_layout()
plt.savefig("knn_k_vs_accuracy.png")
plt.close()

print("\nپایان اجرا. فایل‌های خروجی (نمودارها و جدول نتایج) در همین پوشه ذخیره شدند.")
