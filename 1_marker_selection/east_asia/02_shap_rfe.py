import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb

# ==========================================================
# 0) 读入数据
# ==========================================================

data_path = "../../data/1874_east_asian.csv"
target_col = "groups"

def read_csv_robust(path):
    try:
        df_ = pd.read_csv(
            path, sep=",", engine="python",
            encoding="utf-8-sig", quoting=csv.QUOTE_NONE, on_bad_lines="skip"
        )
    except Exception:
        df_ = pd.read_csv(
            path, sep=",", engine="python",
            encoding="gbk", quoting=csv.QUOTE_NONE, on_bad_lines="skip"
        )
    return df_

df = read_csv_robust(data_path)

# 列名清理
df.columns = df.columns.astype(str).str.replace("\ufeff", "", regex=False).str.strip()
df.columns = [re.sub(r"[^\w\s]", "_", c).strip().replace(" ", "_") for c in df.columns]
target_col = re.sub(r"[^\w\s]", "_", target_col).strip().replace(" ", "_")

print("加载后的总列数 =", df.shape[1])
print("前5列：", df.columns[:5].tolist())

if target_col not in df.columns:
    raise ValueError(f"找不到目标列 '{target_col}'，请检查文件表头。")

# ==========================================================
# 1) 数据划分
# ==========================================================
# 【核心修改】：移除目标列，并自动识别和移除多余的序号列（如 'Unnamed: 0'）
cols_to_drop = [target_col] + [col for col in df.columns if 'Unnamed' in col]
X = df.drop(columns=cols_to_drop, errors="ignore")
y_raw = df[target_col].astype(str)

print(f"剔除序号和标签后，实际用于训练的 SNP 特征数 = {X.shape[1]}")

expected_groups = ["Han", "JPT", "SEA"]
present_groups = sorted(y_raw.unique().tolist())
groups = expected_groups if set(expected_groups).issubset(set(present_groups)) else present_groups
print("用于 One-vs-Rest 的群体：", groups)

le = LabelEncoder()
y_all = le.fit_transform(y_raw)

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y_raw, test_size=0.3, random_state=42, stratify=y_all
)

# ==========================================================
# 2) 工具函数：XGBoost 原生 TreeSHAP 排名 + Top-k递归AUC
# ==========================================================
def shap_rank_xgb_native(model: XGBClassifier, X_ref: pd.DataFrame):
    """
    用 XGBoost 原生 TreeSHAP 得到 SHAP ranking（避免 shap 包的兼容/版本问题）
    返回 ranked_features(list) 与 importance(series)
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(X_ref, feature_names=list(X_ref.columns))
    contrib = booster.predict(dmat, pred_contribs=True)
    shap_vals = contrib[:, :-1]  # 去掉 bias 列

    imp = np.abs(shap_vals).mean(axis=0)
    imp_series = pd.Series(imp, index=X_ref.columns).sort_values(ascending=False)
    return imp_series.index.tolist(), imp_series

def recursive_auc_curve_xgb(xgb_params, Xtr, ytr, Xte, yte, ranked_features, start=3, step=5, max_k=200):
    """
    按 Top-k 特征（来自SHAP排名）逐步训练 XGBoost，计算 AUC 曲线
    """
    max_k = min(max_k, len(ranked_features))
    records = []
    best_auc = -1.0
    best_k = None

    for k in range(start, max_k + 1, step):
        feats = ranked_features[:k]
        model = XGBClassifier(**xgb_params)
        model.fit(Xtr[feats], ytr)

        prob = model.predict_proba(Xte[feats])[:, 1]
        score = roc_auc_score(yte, prob)

        records.append({"k": k, "auc": score})
        if score > best_auc:
            best_auc = score
            best_k = k

    return pd.DataFrame(records), best_k, best_auc

# ==========================================================
# 3) One-vs-Rest：XGB → SHAP 排名 → 递归筛选
# ==========================================================
xgb_params = dict(
    n_estimators=800, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=42, eval_metric="logloss", tree_method="hist", n_jobs=-1
)

# SHAP 排名用训练集子样本加速
shap_sample_n = min(300, X_train.shape[0])
X_train_shap = X_train.sample(shap_sample_n, random_state=42)

START_K = 3
STEP_K = 5
MAX_K = 200

all_best_features = {}
all_auc_curves = []

for g in groups:
    print(f"\n===== One-vs-Rest: {g} =====")
    y_train = (y_train_raw == g).astype(int)
    y_test  = (y_test_raw == g).astype(int)

    # 全特征训练 base_model
    base_model = XGBClassifier(**xgb_params)
    base_model.fit(X_train, y_train)

    ranked_feats, shap_imp = shap_rank_xgb_native(base_model, X_train_shap)
    print("Top10 SHAP 特征：", ranked_feats[:10])

    # 输出所有特征的 SHAP 重要性排序
    shap_imp.name = "mean_abs_shap"
    shap_imp.to_csv(f"{g}_All_SHAP_importance.csv", encoding="utf-8-sig")
    print(f"已保存：{g}_All_SHAP_importance.csv (共 {len(shap_imp)} 个特征)")

    # 递归筛选
    auc_df, best_k, best_auc = recursive_auc_curve_xgb(
        xgb_params=xgb_params, Xtr=X_train, ytr=y_train, Xte=X_test, yte=y_test,
        ranked_features=ranked_feats, start=START_K, step=STEP_K, max_k=MAX_K
    )

    auc_df["group"] = g
    all_auc_curves.append(auc_df)
    best_feats = ranked_feats[:best_k]
    all_best_features[g] = best_feats

    print(f"最优 k={best_k} | AUC={best_auc:.4f}")

    # 保存
    auc_df.to_csv(f"{g}_AUC_curve.csv", index=False, encoding="utf-8-sig")
    pd.Series(best_feats, name="best_features").to_csv(
        f"{g}_best_features.csv", index=False, encoding="utf-8-sig"
    )

# 汇总输出
auc_all_df = pd.concat(all_auc_curves, ignore_index=True)
auc_all_df.to_csv("EastAsia_AllGroups_AUC_curves.csv", index=False, encoding="utf-8-sig")

best_summary = pd.DataFrame({
    "group": list(all_best_features.keys()),
    "best_k": [len(v) for v in all_best_features.values()]
})
best_summary.to_csv("EastAsia_AllGroups_best_k_summary.csv", index=False, encoding="utf-8-sig")

print("\n===== 汇总 best_k =====")
print(best_summary)

# ==========================================================
# 4) 三条AUC曲线综合图
# ==========================================================
plt.figure(figsize=(7, 6))
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold',
    'pdf.fonttype': 42
})
pivot = auc_all_df.pivot_table(index="k", columns="group", values="auc", aggfunc="mean").sort_index()

for g in pivot.columns:
    plt.plot(pivot.index, pivot[g], marker="o", linewidth=1.5, label=g)

plt.title("East Asia OVR Feature Reduction (XGBoost + TreeSHAP)", fontsize=20, fontweight="bold")
plt.xlabel("Number of Features (Top-k by SHAP)", fontsize=18)
plt.ylabel("AUC (One-vs-Rest)", fontsize=18)
plt.grid(alpha=0.3)
plt.legend(title="Group", fontsize=14)
plt.tight_layout()
plt.savefig("EastAsia_OVR_AUC_Combined.pdf", format="pdf", dpi=600)
plt.show()

print("\n综合图已保存：EastAsia_OVR_AUC_Combined.pdf")
print("曲线数据已保存：EastAsia_AllGroups_AUC_curves.csv")
print("最优k汇总已保存：EastAsia_AllGroups_best_k_summary.csv")