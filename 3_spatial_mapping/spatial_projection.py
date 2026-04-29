import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import pickle
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

warnings.filterwarnings('ignore')

# ==============================================================================
# 全局绘图参数设置
# ==============================================================================
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.dpi': 300,
    'pdf.fonttype': 42
})


# ==============================================================================
# 模块 1：实际样本预测与概率输出
# ==============================================================================
def predict_and_visualize_real_samples(model, real_data_path, train_data_path, model_name):
    """
    对实际样本进行预测，生成概率分布表。
    """
    print(f"\n{'=' * 50}\n正在使用 {model_name} 对实际样本进行推断预测...\n{'=' * 50}")

    # 1. 极速重构训练集的预处理管线
    try:
        df_train = pd.read_csv(train_data_path, sep='\t')
        if 'Group' not in df_train.columns: df_train = pd.read_csv(train_data_path, sep=',')
    except Exception:
        df_train = pd.read_csv(train_data_path)

    # 自动剔除目标列及可能存在的无用序号列
    cols_to_drop_train = ['Group'] + [col for col in df_train.columns if 'Unnamed' in col]
    X_train = df_train.drop(columns=cols_to_drop_train, errors='ignore')
    y_train = df_train['Group']

    # 训练集: 标签编码
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = LabelEncoder().fit_transform(X_train[col].astype(str))

    # 拟合 Imputer 和 Scaler
    imputer = SimpleImputer(strategy='most_frequent')
    X_train_imputed = imputer.fit_transform(X_train)
    scaler = StandardScaler()
    scaler.fit(X_train_imputed)

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    classes = label_encoder.classes_

    # 2. 读取并处理实际自测样本
    try:
        df_real = pd.read_csv(real_data_path, sep='\t')
        if 'Group' not in df_real.columns: df_real = pd.read_csv(real_data_path, sep=',')
    except Exception:
        df_real = pd.read_csv(real_data_path)

    sample_ids = df_real['Group'].values
    # 自动识别群体前缀 (例如 'SNH1' -> 'SNH')
    populations = [re.sub(r'\d+', '', str(sid)).strip() for sid in sample_ids]

    cols_to_drop_real = ['Group'] + [col for col in df_real.columns if 'Unnamed' in col]
    X_real = df_real.drop(columns=cols_to_drop_real, errors='ignore')

    # 确保特征列顺序和数量与训练集完全一致
    X_real = X_real[X_train.columns]

    # 自测集应用预处理逻辑
    for col in X_real.columns:
        if X_real[col].dtype == 'object':
            X_real[col] = LabelEncoder().fit_transform(X_real[col].astype(str))

    X_real_imputed = imputer.transform(X_real)
    X_real_scaled = scaler.transform(X_real_imputed)

    # 3. 进行模型预测
    y_pred_idx = model.predict(X_real_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred_idx)
    y_pred_proba = model.predict_proba(X_real_scaled)

    # 4. 生成概率分布表并保存
    prob_df = pd.DataFrame(y_pred_proba, columns=classes)
    prob_df.insert(0, 'Sample_ID', sample_ids)
    prob_df.insert(1, 'Population', populations)
    prob_df.insert(2, 'Predicted_Class', y_pred_labels)

    prob_output_file = f"RealSamples_Probabilities_{model_name}.csv"
    prob_df.to_csv(prob_output_file, index=False)
    print(f" -> 概率分布表已成功保存至: {prob_output_file}")


# ==============================================================================
# 模块 2：混合可视化 - 自测样本投影至参考背景
# ==============================================================================
def plot_real_samples_on_reference_background(model, train_data_path, real_data_path, model_name, dr_method='pca'):
    """
    将实际自测样本标记在参考群体背景中，采用降维主成分与预测概率交叉映射。
    """
    print(f"\n[{model_name}] 正在执行 {dr_method.upper()} 背景投影与空间概率可视化...")

    # 1. 数据加载与一致性预处理
    df_train = pd.read_csv(train_data_path, sep=',' if ',' in open(train_data_path).readline() else '\t')
    cols_drop_train = ['Group'] + [col for col in df_train.columns if 'Unnamed' in col]
    X_train = df_train.drop(columns=cols_drop_train, errors='ignore')
    y_train = df_train['Group']

    df_real = pd.read_csv(real_data_path, sep=',' if ',' in open(real_data_path).readline() else '\t')
    real_sample_ids = df_real['Group'].values
    real_populations = np.array([re.sub(r'\d+', '', str(sid)).strip() for sid in real_sample_ids])

    cols_drop_real = ['Group'] + [col for col in df_real.columns if 'Unnamed' in col]
    X_real = df_real.drop(columns=cols_drop_real, errors='ignore')[X_train.columns]

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_real[col] = le.transform(X_real[col].astype(str))

    imputer = SimpleImputer(strategy='most_frequent')
    X_train_imputed = imputer.fit_transform(X_train)
    X_real_imputed = imputer.transform(X_real)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_real_scaled = scaler.transform(X_real_imputed)

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    classes = label_encoder.classes_

    # 2. 获取预测概率
    y_train_proba = model.predict_proba(X_train_scaled)
    y_real_proba = model.predict_proba(X_real_scaled)

    # 3. 降维处理
    if dr_method.lower() == 'pca':
        pca = PCA(n_components=1, random_state=42)
        dr_train = pca.fit_transform(X_train_scaled).flatten()
        dr_real = pca.transform(X_real_scaled).flatten()
        y_label = 'PCA Component 1'
    elif dr_method.lower() == 'tsne':
        combined_X = np.vstack((X_train_scaled, X_real_scaled))
        tsne = TSNE(n_components=1, random_state=42, init='pca', learning_rate='auto')
        combined_dr = tsne.fit_transform(combined_X).flatten()
        dr_train = combined_dr[:len(X_train_scaled)]
        dr_real = combined_dr[len(X_train_scaled):]
        y_label = 't-SNE Component 1'
    else:
        raise ValueError("dr_method 必须为 'pca' 或 'tsne'")

    # 4. 绘图排版
    n_classes = len(classes)
    n_cols = 3 if n_classes >= 3 else n_classes
    n_rows = int(np.ceil(n_classes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), dpi=300)
    axes = np.array(axes).flatten()

    palette_bg = sns.color_palette("husl", n_classes)
    color_bg_dict = dict(zip(classes, palette_bg))

    unique_real_pops = np.unique(real_populations)
    palette_real = ['#FFD700', '#00FFFF', '#FF1493', '#39FF14']  # 扩展高亮色盘

    for i, class_name in enumerate(classes):
        ax = axes[i]

        # 绘制背景参考群
        prob_x_train = y_train_proba[:, i]
        for true_class in classes:
            idx = (y_train == true_class)
            is_main = (true_class == class_name)
            alpha_val = 0.6 if is_main else 0.2
            marker_size = 35 if is_main else 15
            z_order = 5 if is_main else 1

            ax.scatter(prob_x_train[idx], dr_train[idx],
                       c=[color_bg_dict[true_class]],
                       label=f"Ref: {true_class}" if i == 0 else "",
                       marker='o', alpha=alpha_val, s=marker_size, edgecolors='none', zorder=z_order)

        # 绘制重叠的自测样本
        prob_x_real = y_real_proba[:, i]
        for j, pop in enumerate(unique_real_pops):
            idx_real = (real_populations == pop)
            ax.scatter(prob_x_real[idx_real], dr_real[idx_real],
                       c=palette_real[j % len(palette_real)],
                       marker='o', s=60, edgecolors='black', linewidths=1.0,
                       label=f"Test: {pop}" if i == 0 else "", zorder=20)

        # 设置坐标系
        ax.set_title(f'Probability of predicting [{class_name}]', fontsize=14, pad=15)
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, zorder=0)

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

    for j in range(i + 1, len(axes)): fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
               ncol=min(8, len(labels)), frameon=True, edgecolor='black', fontsize=12)

    plt.suptitle(f'{model_name} - Projecting Real Samples into Reference Space\n({dr_method.upper()} vs Probability)',
                 fontsize=18, fontweight='bold', y=1.1)

    plt.tight_layout()
    save_path = f"Projection_{model_name}_{dr_method.upper()}_vs_Probability.pdf"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f" -> 成功！背景投影可视化图表已保存为：{save_path}")


# ==============================================================================
# 直接调用执行 (从 2_ancestry_inference 加载序列化模型)
# ==============================================================================
if __name__ == "__main__":

    # Github 开源数据路径
    real_samples_file = "../data/sample_demo.csv"
    train_continent = "../data/continent_3663.csv"
    train_east_asia = "../data/eastasia_957.csv"

    # 【核心修改区】跨文件夹加载模型文件
    model_a_path = "../2_ancestry_inference/Continent_Model.pkl"
    model_b_path = "../2_ancestry_inference/EastAsia_Model.pkl"

    # 尝试加载 Continent_Model
    if os.path.exists(model_a_path):
        with open(model_a_path, 'rb') as f:
            model_A = pickle.load(f)
        print(f"成功加载预训练模型: {model_a_path}")
    else:
        print(f"警告: 未找到 {model_a_path}，请确保您已先运行 2_ancestry_inference 目录下的建模代码！")

    # 尝试加载 EastAsia_Model
    if os.path.exists(model_b_path):
        with open(model_b_path, 'rb') as f:
            model_B = pickle.load(f)
        print(f"成功加载预训练模型: {model_b_path}")
    else:
        print(f"警告: 未找到 {model_b_path}，请确保您已先运行 2_ancestry_inference 目录下的建模代码！")

    # 1. 洲际模型分析 (PCA + Probability)
    if 'model_A' in locals() or 'model_A' in globals():
        predict_and_visualize_real_samples(model_A, real_samples_file, train_continent, "Continent_Model")
        plot_real_samples_on_reference_background(model_A, train_continent, real_samples_file, "Continent_Model",
                                                  dr_method='pca')

    # 2. 东亚亚群模型分析 (t-SNE + Probability)
    if 'model_B' in locals() or 'model_B' in globals():
        predict_and_visualize_real_samples(model_B, real_samples_file, train_east_asia, "EastAsia_Model")
        plot_real_samples_on_reference_background(model_B, train_east_asia, real_samples_file, "EastAsia_Model",
                                                  dr_method='tsne')