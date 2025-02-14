import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import dump
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt
import seaborn as sns


# 假设 load_data 函数已经定义了不同的特征提取方式
def load_data(filepath, feature_type='TopologicalTorsion'):
    df = pd.read_csv(filepath)
    # 选择指纹类型
    if feature_type == 'Morgan':
        X = np.array([AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), radius=2, nBits=1024) for smi in
                      list(df.iloc[:, 0])])
    elif feature_type == 'MACCS':
        X = np.array([AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(smi)) for smi in list(df.iloc[:, 0])])
    elif feature_type == 'TopologicalTorsion':
        X = np.array([AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(Chem.MolFromSmiles(smi)) for smi in
                      list(df.iloc[:, 0])])
    elif feature_type == 'RDK':
        X = np.array([Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in list(df.iloc[:, 0])])
    else:
        raise ValueError("Unsupported feature type. Choose from 'Morgan', 'MACCS', 'TopologicalTorsion', 'RDK'.")

    y = df['class'].values
    return X, y


# 计算评估指标
def calculate_metrics(tp, tn, fp, fn):
    se = tp / (tp + fn)  # Sensitivity
    sp = tn / (tn + fp)  # Specificity
    q = (tp + tn) / (tp + fn + tn + fp)  # Accuracy (Q)
    mcc = (tp * tn - fn * fp) / math.sqrt(
        (tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))  # Matthews correlation coefficient
    P = tp / (tp + fp)  # Precision
    F1 = (P * se * 2) / (P + se)  # F1 Score
    BA = (se + sp) / 2  # Balanced Accuracy

    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'se': se, 'sp': sp, 'mcc': mcc, 'q': q, 'P': P, 'F1': F1, 'BA': BA}


# 定义保存结果到CSV文件的函数
def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)


# 定义训练模型的函数
def train_model(X_train, y_train, model, params, cv_splitter):
    """
    Train a model using GridSearchCV with a pre-defined cross-validation splitter.
    """
    gc = GridSearchCV(model, param_grid=params, cv=cv_splitter, scoring='roc_auc', return_train_score=True, verbose=2)
    gc.fit(X_train, y_train)
    return gc


# 定义评估模型的函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_roc_score = roc_auc_score(y_test, y_pred)

    # 将预测结果二值化
    y_pred_binary = (y_pred >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()

    # 计算所有评估指标
    metrics = calculate_metrics(tp, tn, fp, fn)
    metrics['auc_roc_score'] = auc_roc_score
    return metrics, y_pred_binary


# 绘制混淆矩阵的函数
def plot_confusion_matrix(cm, class_names):
    # Normalize the confusion matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Use seaborn to create a heatmap.
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', ax=ax)

    # Set the labels for x-axis and y-axis.
    ax.set_xlabel('Predicted label', labelpad=10)
    ax.set_ylabel('True label', labelpad=10)

    # Set the title for the heatmap.
    ax.set_title('XGB Confusion Matrix (Percentage)', pad=20)

    # Position the tick labels at the center of the grid cell.
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Set the alignment of the tick labels.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")


# 主程序部分
if __name__ == "__main__":
    # 创建KFold分割器实例
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 定义不同的特征类型
    feature_types = ['Morgan', 'MACCS', 'TopologicalTorsion', 'RDK']

    # 存储所有的评估结果
    all_results = []

    # 循环遍历每个特征类型
    for feature_type in feature_types:
        print(f"Training with feature type: {feature_type}")

        # 加载数据
        X, y = load_data('cleaned_alldata.csv', feature_type=feature_type)

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 定义XGBoost模型
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, random_state=123)
        xgb_params = {"max_depth": [3, 5, 7, 10, 15]}

        # 训练模型
        xgb_grid_search = train_model(X_train, y_train, xgb_model, xgb_params, kf)
        print(f"Best parameters for {feature_type}: {xgb_grid_search.best_params_}")

        # 评估模型
        xgb_metrics, xgb_pred_binary = evaluate_model(xgb_grid_search.best_estimator_, X_test, y_test)

        # 保存结果
        result = {'feature_type': feature_type, **xgb_metrics}  # 添加特征类型到结果中
        all_results.append(result)



    # 保存所有评估结果到CSV文件
    save_results_to_csv(all_results, 'xgb_results.csv')
    print("Results saved to xgb_results.csv")
