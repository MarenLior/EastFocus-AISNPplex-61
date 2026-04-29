import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.utils import resample
from scipy import stats

print("正在加载洲际数据并进行预处理...")
data_path = '../../data/1874_continental.csv'
try:
    df = pd.read_csv(data_path, encoding='utf-8-sig')
except:
    df = pd.read_csv(data_path, encoding='gbk')

target_col = 'groups'
df.columns = [re.sub(r'[^\w\s]', '_', col).strip().replace(' ', '_') for col in df.columns]
target_col = re.sub(r'[^\w\s]', '_', target_col).strip().replace(' ', '_')

le = LabelEncoder()
df['y'] = le.fit_transform(df[target_col])
classes = le.classes_
n_classes = len(classes)

X = df.drop([target_col, 'y'], axis=1, errors='ignore')
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

models_dict = {
    "RF": RandomForestClassifier(random_state=42),
    "GBM": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42, algorithm='SAMME')
}

def calculate_auc_ci(y_true, y_prob, n_classes, n_bootstrap=1000):
    bootstrapped_aucs = []
    y_true_arr = np.array(y_true)
    for i in range(n_bootstrap):
        indices = resample(np.arange(len(y_true_arr)), random_state=42 + i)
        if len(np.unique(y_true_arr[indices])) < n_classes: continue
        score = roc_auc_score(y_true_arr[indices], y_prob[indices], multi_class='ovr', average='macro')
        bootstrapped_aucs.append(score)
    sorted_scores = np.sort(bootstrapped_aucs)
    return sorted_scores[int(0.025 * len(sorted_scores))], sorted_scores[int(0.975 * len(sorted_scores))]

def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1); i = j
    T2 = np.empty(N, dtype=float); T2[J] = T + 1
    return T2

def fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count; n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx, ty, tz = np.zeros((k, m)), np.zeros((k, n)), np.zeros((k, m + n))
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / 2 / n
    v01, v10 = (tz[:, :m] - tx) / n, 1 - (tz[:, m:] - ty) / m
    sx, sy = np.cov(v01), np.cov(v10)
    return aucs, sx / m + sy / n

def delong_roc_test(y_true, y_pred1, y_pred2):
    y_true = np.array(y_true); order = np.argsort(-y_true)
    y_true, preds = y_true[order], np.vstack((y_pred1, y_pred2))[:, order]
    label_1_count = int(np.sum(y_true))
    aucs, covariance = fast_delong(preds, label_1_count)
    diff = aucs[0] - aucs[1]
    var = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    if var <= 0: return 1.0
    z = diff / np.sqrt(var)
    return 2 * (1 - stats.norm.cdf(abs(z)))

results, trained_models, probas_all = [], {}, {}
print("正在开始训练模型、计算置信区间及显著性检验...")
for name, model in models_dict.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    y_prob = model.predict_proba(X_test)
    probas_all[name] = y_prob
    acc = accuracy_score(y_test, model.predict(X_test))
    macro_auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')
    ci_low, ci_high = calculate_auc_ci(y_test, y_prob, n_classes)
    results.append({"Model": name, "Accuracy": acc, "Macro-AUC": macro_auc, "AUC_95%_CI": f"({ci_low:.3f}-{ci_high:.3f})"})
    print(f"完成模型: {name} | AUC: {macro_auc:.3f}")

results_df = pd.DataFrame(results)

plt.rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'axes.titleweight': 'bold', 'pdf.fonttype': 42})
plt.figure(figsize=(10, 8))
y_test_bin = label_binarize(y_test, classes=range(n_classes))

for name, model in trained_models.items():
    y_prob = probas_all[name]
    all_fpr = np.unique(np.concatenate([roc_curve(y_test_bin[:, i], y_prob[:, i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    ci_str = results_df.loc[results_df['Model'] == name, 'AUC_95%_CI'].values[0]
    plt.plot(all_fpr, mean_tpr, label=f'{name} AUC={auc(all_fpr, mean_tpr):.3f} 95%CI:{ci_str}', lw=2)

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
plt.title("Macro-average OvR ROC with 95% CI", fontsize=20, fontweight="bold")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=14); plt.grid(alpha=0.3); sns.despine()
plt.tight_layout(); plt.savefig("Macro_ROC_with_CI_Continental.pdf", format='pdf', dpi=600)

model_names = list(trained_models.keys())
delong_df = pd.DataFrame(np.ones((len(model_names), len(model_names))), index=model_names, columns=model_names)
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        p_vals = [delong_roc_test(y_test_bin[:, c], probas_all[model_names[i]][:, c], probas_all[model_names[j]][:, c]) for c in range(n_classes)]
        delong_df.iloc[i, j] = delong_df.iloc[j, i] = np.mean(p_vals)

def pvalue_star(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
plt.figure(figsize=(10, 8))
delong_lower = delong_df.where(np.tril(np.ones(delong_df.shape), k=-1).astype(bool))
ax = sns.heatmap(delong_lower, annot=delong_lower.map(lambda x: f"{x:.3f}\n{pvalue_star(x)}" if not pd.isna(x) else ""),
                 fmt="", cmap="RdYlBu_r", vmin=0, vmax=0.05, square=True, linewidths=1, linecolor="white", cbar_kws={"label": "P value"})
ax.set_title("DeLong Test: Macro-average ROC Comparison", fontsize=20, fontweight="bold", pad=20)
plt.tight_layout(); plt.savefig("delong_comparison_Continental.pdf", format='pdf', dpi=600)