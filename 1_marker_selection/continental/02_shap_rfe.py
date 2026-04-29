import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import csv
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# ==========================================================
# 0) 读入数据
# ==========================================================

data_path = "../../data/1874_continental.csv"

try:
    df = pd.read_csv(
        data_path, sep=",", engine="python",
        encoding="utf-8-sig", quoting=csv.QUOTE_NONE, on_bad_lines="skip"
    )
except Exception:
    df = pd.read_csv(
        data_path, sep=",", engine="python",
        encoding="gbk", quoting=csv.QUOTE_NONE, on_bad_lines="skip"
    )

# 清理列名
df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()

print("加载后的总列数 =", df.shape[1])
print("前5列：", df.columns[:5].tolist())

target_col = "groups"
if target_col not in df.columns:
    raise ValueError(f"找不到目标列 '{target_col}'，请检查文件表头。")

# ==========================================================
# 1) 划分训练/测试集
# ==========================================================

cols_to_drop = [target_col] + [col for col in df.columns if 'Unnamed' in col]
X = df.drop(columns=cols_to_drop, errors="ignore")
y_raw = df[target_col].astype(str)

print(f"剔除序号和标签后，实际用于训练的 SNP 特征数 = {X.shape[1]}")

le = LabelEncoder()
y_all = le.fit_transform(y_raw)
continents = sorted(y_raw.unique().tolist())

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.3, random_state=42, stratify=y_all
)

print("洲际类别：", continents)

# ==========================================================
# 2) 工具函数
# ==========================================================
def shap_rank_catboost(model, X_ref):
    """
    计算 mean(|SHAP|) 排名，返回 ranked_features(list) 与 importance(series)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_ref)

    # 兼容返回 list 的情况：二分类通常是 [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    imp = np.abs(shap_values).mean(axis=0)
    imp_series = pd.Series(imp, index=X_ref.columns).sort_values(ascending=False)
    return imp_series.index.tolist(), imp_series

def recursive_auc_curve(cat_params, Xtr, ytr, Xte, yte, ranked_features, start=3, step=5, max_k=200):
    """
    按 Top-k 特征（来自SHAP排名）逐步训练，计算AUC曲线，返回 auc_df, best_k, best_auc
    """
    max_k = min(max_k, len(ranked_features))
    records = []
    best_auc = -1.0
    best_k = None

    for k in range(start, max_k + 1, step):
        feats = ranked_features[:k]
        model = CatBoostClassifier(**cat_params)
        model.fit(Xtr[feats], ytr, verbose=0)
        prob = model.predict_proba(Xte[feats])[:, 1]
        score = roc_auc_score(yte, prob)
        records.append({"k": k, "auc": score})

        if score > best_auc:
            best_auc = score
            best_k = k

    auc_df = pd.DataFrame(records)
    return auc_df, best_k, best_auc

# ==========================================================
# 3) 对每个洲际做 One-vs-Rest：SHAP 排名 + 递归筛选
# ==========================================================
cat_params = dict(
    random_state=42, loss_function="Logloss", eval_metric="AUC",
    depth=6, learning_rate=0.1, iterations=500, verbose=0
)

# SHAP 用训练集子样本加速
shap_sample_n = min(500, X_train.shape[0])
X_train_shap = X_train.sample(shap_sample_n, random_state=42)

all_best_features = {}   # 每个洲际最优特征列表
all_auc_curves = []      # 汇总所有洲际的 AUC 曲线

# 递归参数
START_K = 3
STEP_K = 5
MAX_K = 200

for cont in continents:
    print(f"\n===== One-vs-Rest: {cont} =====")
    y_train = (y_train_raw == cont).astype(int)
    y_test  = (y_test_raw == cont).astype(int)

    # 先用全特征训练一个基准模型来做 SHAP 排名
    base_model = CatBoostClassifier(**cat_params)
    base_model.fit(X_train, y_train, verbose=0)

    ranked_feats, shap_imp = shap_rank_catboost(base_model, X_train_shap)
    print("Top10 SHAP 特征：", ranked_feats[:10])

    # 输出该洲际“所有特征”的 SHAP 重要性排序
    shap_imp.name = "mean_abs_shap"
    shap_imp.to_csv(f"{cont}_All_SHAP_importance.csv", encoding="utf-8-sig")
    print(f"已保存：{cont}_All_SHAP_importance.csv (共 {len(shap_imp)} 个特征)")

    # 递归筛选
    auc_df, best_k, best_auc = recursive_auc_curve(
        cat_params=cat_params, Xtr=X_train, ytr=y_train,
        Xte=X_test, yte=y_test, ranked_features=ranked_feats,
        start=START_K, step=STEP_K, max_k=MAX_K
    )

    auc_df["continent"] = cont
    all_auc_curves.append(auc_df)

    best_feats = ranked_feats[:best_k]
    all_best_features[cont] = best_feats

    print(f"最优 k={best_k} | AUC={best_auc:.4f}")

    # 保存该洲际的曲线数据 & 最优特征
    auc_df.to_csv(f"{cont}_AUC_curve.csv", index=False, encoding="utf-8-sig")
    pd.Series(best_feats, name="best_features").to_csv(
        f"{cont}_best_features.csv", index=False, encoding="utf-8-sig"
    )

# 汇总成一个总表
auc_all_df = pd.concat(all_auc_curves, ignore_index=True)
auc_all_df.to_csv("All_Continents_AUC_curves.csv", index=False, encoding="utf-8-sig")

best_summary = pd.DataFrame({
    "continent": list(all_best_features.keys()),
    "best_k": [len(v) for v in all_best_features.values()]
})
best_summary.to_csv("All_Continents_best_k_summary.csv", index=False, encoding="utf-8-sig")

print("\n===== 汇总 best_k =====")
print(best_summary)

# ==========================================================
# 4) 绘制 Macro-average ROC 曲线综合图
# ==========================================================
plt.figure(figsize=(7, 6))

# 设置全局字体参数（使用新罗马字体）
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'pdf.fonttype': 42
})

# 使用 pivot 对齐线条数据
pivot = auc_all_df.pivot_table(index="k", columns="continent", values="auc", aggfunc="mean").sort_index()

for cont in pivot.columns:
    plt.plot(pivot.index, pivot[cont], marker="o", linewidth=1.5, label=cont)

plt.title("Feature Reduction Curves (CatBoost + SHAP, One-vs-Rest)", fontsize=20, fontweight="bold")
plt.xlabel("Number of Features (Top-k by SHAP)", fontsize=18)
plt.ylabel("AUC (One-vs-Rest)", fontsize=18)
plt.grid(alpha=0.3)
plt.legend(title="Continent", fontsize=14)
plt.tight_layout()

plt.savefig("All_Continents_AUC_Combined.pdf", format="pdf", dpi=600)
plt.show()

print("\n分析执行完毕：")
print("1. 综合图已保存：All_Continents_AUC_Combined.pdf")
print("2. 曲线数据已保存：All_Continents_AUC_curves.csv")
print("3. 最优k汇总已保存：All_Continents_best_k_summary.csv")