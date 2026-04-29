import os
import re
import warnings
import pickle  #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches

import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from flaml import AutoML

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 全局绘图参数设置
# ==============================================================================
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'axes.linewidth': 1.2,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})


# ==============================================================================
# 2. 数据加载与统一特征工程模块
# ==============================================================================
def load_and_preprocess_data(file_path):
    """读取数据，剔除无用序号，执行标准化与编码，返回处理后的训练/测试集对象"""
    try:
        df = pd.read_csv(file_path, sep='\t')
        if 'Group' not in df.columns:
            df = pd.read_csv(file_path, sep=',')
    except Exception:
        df = pd.read_csv(file_path)

    target_col = 'Group'
    if target_col not in df.columns:
        raise ValueError(f"数据集中未找到标签列 '{target_col}'。请检查文件格式。")

    # 剔除目标列与可能存在的序号列 (Unnamed)
    cols_to_drop = [target_col] + [col for col in df.columns if 'Unnamed' in col]
    X = df.drop(columns=cols_to_drop, errors="ignore")
    y = df[target_col]

    # 基因型数值化
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # 缺失值填补与标准化
    imputer = SimpleImputer(strategy='most_frequent')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)

    # 标签编码
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_

    # 划分数据集 (固化随机种子)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, classes, X.columns


# ==============================================================================
# 3. 高阶可视化绘图模块
# ==============================================================================
def plot_advanced_evaluation(y_test, y_pred, y_pred_proba, classes, model_name, cmap_name="Blues"):
    """绘制带 Total 的混淆矩阵和多分类 ROC 曲线"""
    # ---------------- 1: 改进版混淆矩阵 ----------------
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    cm_with_totals = np.vstack([cm, cm.sum(axis=0)])
    cm_with_totals = np.column_stack([cm_with_totals, cm_with_totals.sum(axis=1)])
    labels_with_totals = list(classes) + ['Total']

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_obj = sns.color_palette(cmap_name, as_cmap=True)
    grey_color = "#b8b8b8"

    ax.set_xlim(0, len(labels_with_totals));
    ax.set_ylim(len(labels_with_totals), 0)

    for i in range(cm_with_totals.shape[0]):
        for j in range(cm_with_totals.shape[1]):
            color = cmap_obj(cm_normalized[i, j]) if (i < cm.shape[0] and j < cm.shape[1]) else grey_color
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor="white", lw=2))

            if i < cm.shape[0] and j < cm.shape[1]:
                val = int(cm[i, j]);
                pct = cm_normalized[i, j] * 100
                fcolor = "white" if i == j else "black"
                ax.text(j + 0.5, i + 0.45, f"{pct:.1f}%", ha="center", va="center", fontsize=15, color=fcolor,
                        fontweight='bold')
                ax.text(j + 0.5, i + 0.65, f"({val})", ha="center", va="center", fontsize=12, color=fcolor)
            elif i == cm.shape[0] or j == cm.shape[1]:
                total_val = int(cm_with_totals[i, j])
                if i == cm.shape[0] and j == cm.shape[1]:
                    ax.text(j + 0.5, i + 0.5, f"{total_val}", ha="center", va="center", fontsize=16, color="black",
                            fontweight='bold')
                else:
                    total_pct = total_val / cm_with_totals[-1, -1] * 100
                    ax.text(j + 0.5, i + 0.45, f"{total_pct:.1f}%", ha="center", va="center", fontsize=14,
                            color="black", fontweight='bold')
                    ax.text(j + 0.5, i + 0.65, f"({total_val})", ha="center", va="center", fontsize=12, color="black")

    ax.set_xticks(np.arange(len(labels_with_totals)) + 0.5)
    ax.set_yticks(np.arange(len(labels_with_totals)) + 0.5)
    ax.set_xticklabels(labels_with_totals, fontsize=12, fontweight='bold')
    ax.set_yticklabels(labels_with_totals, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=14, labelpad=15)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=14, labelpad=15)
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=16, pad=20)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    plt.tight_layout()
    plt.savefig(f'{model_name}_Advanced_CM.pdf', bbox_inches='tight')
    plt.close()

    # ---------------- 2: 多分类 ROC 曲线 ----------------
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = y_test_bin.shape[1]

    plt.figure(figsize=(8, 7))
    colors = sns.color_palette("Set1", n_classes)

    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        plt.plot(fpr, tpr, color=color, lw=2.5, label=f'{classes[i]} (AUC = {auc(fpr, tpr):.3f})')

    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    plt.title(f'ROC Curve - {model_name}', fontweight='bold', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5, color='grey')
    plt.legend(loc="lower right", frameon=True, edgecolor='black', fontsize=11)
    plt.tight_layout()
    plt.savefig(f'{model_name}_Advanced_ROC.pdf', bbox_inches='tight')
    plt.close()


def plot_advanced_shap_no_group(shap_values, X_test, feature_names, classes, model_name, max_display=20):
    """绘制每个分类的 Dot + Bar 重叠 SHAP 解释图"""
    shap_cmap = mcolors.LinearSegmentedColormap.from_list("shap_default", ["#008bfb", "#ff0051"])

    for class_idx, class_name in enumerate(classes):
        if isinstance(shap_values, list):
            shap_vals_class = shap_values[class_idx]
        elif len(np.array(shap_values).shape) == 3:
            shap_vals_class = shap_values[:, :, class_idx]
        else:
            shap_vals_class = shap_values

        fig = plt.figure(figsize=(12, 10), dpi=300);
        ax1 = plt.gca()

        shap.summary_plot(shap_vals_class, X_test, feature_names=feature_names, plot_type="dot",
                          max_display=max_display, show=False, color_bar=False, cmap=shap_cmap)

        feature_names_ordered = [t.get_text() for t in ax1.get_yticklabels()]
        yticks_locs = ax1.get_yticks()

        ax2 = ax1.twiny()
        plt.sca(ax2)
        shap.summary_plot(shap_vals_class, X_test, feature_names=feature_names, plot_type="bar",
                          max_display=max_display, show=False, color_bar=False)

        ax1.set_zorder(10);
        ax1.patch.set_visible(False);
        ax2.set_zorder(1)

        for bar in ax2.patches:
            bar.set_facecolor('#cccccc');
            bar.set_edgecolor('#a6a6a6');
            bar.set_alpha(0.5)

        ylim_range = (-0.5, len(feature_names_ordered) - 0.5)
        ax1.set_ylim(ylim_range);
        ax2.set_ylim(ylim_range)

        ax2.set_xlabel(f'Mean |SHAP value| (Average impact on {class_name})', fontsize=14, weight='bold', labelpad=10)
        ax2.xaxis.set_label_position('top');
        ax2.xaxis.tick_top();
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(True);
        ax2.spines['top'].set_color('gray');
        ax2.spines['top'].set_linewidth(1.5)
        for spine in ['right', 'bottom', 'left']: ax2.spines[spine].set_visible(False)

        ax1.set_xlabel(f'SHAP value (Impact on {class_name})', fontsize=14, weight='bold', labelpad=10)
        ax1.spines['bottom'].set_linewidth(1.5)
        for spine in ['top', 'right']: ax1.spines[spine].set_visible(False)

        ax1.set_yticks(yticks_locs);
        ax1.set_yticklabels(feature_names_ordered, fontsize=14, fontweight='bold')
        ax1.tick_params(axis='y', which='major', left=True, labelleft=True, length=5, width=1.5, pad=5, direction='out',
                        zorder=20)
        ax1.spines['left'].set_visible(True);
        ax1.spines['left'].set_color('black')
        ax1.spines['left'].set_linewidth(1.5);
        ax1.spines['left'].set_zorder(20)

        plt.tight_layout();
        fig.canvas.draw()

        pos = ax1.get_position()
        cax = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.02, pos.height])
        sm = cm.ScalarMappable(cmap=shap_cmap, norm=mcolors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cb = plt.colorbar(sm, cax=cax);
        cb.outline.set_linewidth(1.2)
        cb.set_ticks([0, 1]);
        cb.set_ticklabels(['Low', 'High'])
        cb.ax.tick_params(labelsize=14, length=0)
        cb.set_label('Feature Value', fontsize=14, weight='bold', labelpad=10)

        plt.savefig(f"{model_name}_SHAP_{class_name}.pdf", bbox_inches='tight', dpi=300)
        plt.close()


def plot_global_shap_multiclass_custom(shap_values, X_test, feature_names, classes, model_name, custom_colors,
                                       max_display=10):
    """绘制带定制配色的全局多分类 SHAP 堆叠柱状图"""
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
    elif isinstance(shap_values, list):
        shap_values_list = shap_values
    else:
        shap_values_list = [shap_values]

    fig = plt.figure(figsize=(8, 6), dpi=300);
    ax = plt.gca()

    shap.summary_plot(shap_values_list, X_test, feature_names=feature_names, class_names=classes,
                      max_display=max_display, plot_type="bar", show=False)

    bars = [p for p in ax.patches if isinstance(p, plt.Rectangle)]
    num_features = min(max_display, X_test.shape[1])
    num_classes = len(classes)

    if len(bars) >= num_features * num_classes:
        for i in range(num_classes):
            for j in range(num_features):
                idx = i * num_features + j
                bars[idx].set_facecolor(custom_colors[i % len(custom_colors)])
                bars[idx].set_edgecolor('white')
                bars[idx].set_linewidth(1.5)

    _, labels = ax.get_legend_handles_labels()
    if labels:
        custom_handles = [
            mpatches.Patch(facecolor=custom_colors[i % len(custom_colors)], edgecolor='white', linewidth=1.5) for i in
            range(len(labels))]
        ax.legend(custom_handles, labels, loc='lower right', frameon=True, edgecolor='black', fontsize=12)

    ax.set_xlabel('Mean |SHAP value| (Global average impact)', fontsize=16, weight='bold', labelpad=15)
    ax.tick_params(axis='y', labelsize=14);
    ax.tick_params(axis='x', labelsize=12)
    plt.title(f'Global Feature Importance (Top {max_display} SNPs)', fontsize=18, fontweight='bold', pad=20)

    ax.spines['top'].set_visible(False);
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2.0);
    ax.spines['left'].set_linewidth(2.0)

    plt.tight_layout()
    plt.savefig(f"{model_name}_Global_SHAP_Top{max_display}.pdf", dpi=300)
    plt.close()


# ==============================================================================
# 4. 主干调度函数 (Pipeline)
# ==============================================================================
def run_full_pipeline(file_path, model_name, custom_colors, cmap_name="Blues", max_iter=300, max_display_shap=15):
    """
    统一的训练与评估流水线，杜绝重复加载数据。
    """
    print(f"\n{'=' * 50}\n开始读取数据并构建 {model_name}...\n{'=' * 50}")

    # 1. 仅加载和预处理一次
    X_train, X_test, y_train, y_test, classes, feature_names = load_and_preprocess_data(file_path)

    # 2. 自动化建模与调优
    print("[1/4] 启动 FLAML AutoML 搜索...")
    automl = AutoML()
    automl.fit(X_train=X_train, y_train=y_train, max_iter=max_iter, metric='accuracy', task='classification',
               estimator_list=['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost'], seed=42, n_jobs=1, verbose=0)

    best_model = automl.model.estimator
    y_pred = automl.predict(X_test)
    y_pred_proba = automl.predict_proba(X_test)

    print("\n--- 最佳模型配置 ---")
    print(f"Algorithm: {automl.best_estimator}\nAccuracy: {1 - automl.best_loss:.4f}")
    print("\n--- 分类报告 ---")
    print(classification_report(y_test, y_pred, target_names=classes))

    # 3. 基础与高级评估绘图
    print("[2/4] 生成高级混淆矩阵与 ROC 曲线...")
    plot_advanced_evaluation(y_test, y_pred, y_pred_proba, classes, model_name, cmap_name)

    # 4. 计算并绘制 SHAP 可解释性图
    print("[3/4] 计算并绘制全局 SHAP 解释图...")
    try:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)

        plot_advanced_shap_no_group(shap_values, X_test, feature_names, classes, model_name,
                                    max_display=max_display_shap)
        plot_global_shap_multiclass_custom(shap_values, X_test, feature_names, classes, model_name, custom_colors,
                                           max_display=max_display_shap)
        print(" -> SHAP 图像生成完毕。")
    except Exception as e:
        print(f" -> SHAP 分析失败 (部分集成算法可能不完全兼容SHAP): {e}")

    # 5. [新增] 保存训练好的模型以便后续空间映射脚本使用
    model_save_path = f"{model_name}.pkl"
    with open(model_save_path, 'wb') as f:
        pickle.dump(automl, f)
    print(f"[4/4] 模型已成功序列化并保存至: {model_save_path}")

    return automl


# ==============================================================================
# 5. 主程序入口
# ==============================================================================
if __name__ == "__main__":

    continent_colors = ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"]
    eastasia_colors = ["#8491B4", "#91D1C2", "#DC0000", "#7E6148"]

    # 限制迭代次数控制复现时间
    FLAML_MAX_ITER = 300

    # ----- 1. 执行洲际模型 -----
    continent_file = "../data/continent_3663.csv"
    if os.path.exists(continent_file):
        model_A = run_full_pipeline(
            file_path=continent_file,
            model_name="Continent_Model",
            custom_colors=continent_colors,
            cmap_name="Blues",
            max_iter=FLAML_MAX_ITER,
            max_display_shap=15
        )
    else:
        print(f"未找到文件: {continent_file}")

    # ----- 2. 执行东亚亚群模型 -----
    east_asia_file = "../data/eastasia_957.csv"
    if os.path.exists(east_asia_file):
        model_B = run_full_pipeline(
            file_path=east_asia_file,
            model_name="EastAsia_Model",
            custom_colors=eastasia_colors,
            cmap_name="Purples",
            max_iter=FLAML_MAX_ITER,
            max_display_shap=15
        )
    else:
        print(f"未找到文件: {east_asia_file}")