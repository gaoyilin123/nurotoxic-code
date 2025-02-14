import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import dump
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import math
import matplotlib.pyplot as plt
import seaborn as sns


# 加载数据并选择特征
def load_data(filepath, feature_type='TopologicalTorsion'):
    df = pd.read_csv(filepath)
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


# 保存评估结果到CSV文件
def save_results_to_csv(results, filename):
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)


# 训练模型
def train_model(X_train, y_train, model, params, cv_splitter):
    gc = GridSearchCV(model, param_grid=params, cv=cv_splitter, scoring='roc_auc', return_train_score=True, verbose=2)
    gc.fit(X_train, y_train)
    return gc


# 评估模型
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


# 绘制混淆矩阵
def plot_confusion_matrix(cm, class_names):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted label', labelpad=10)
    ax.set_ylabel('True label', labelpad=10)
    ax.set_title('MLP Confusion Matrix (Percentage)', pad=20)
    ax.set_xticks(np.arange(cm.shape[1]) + 0.5)
    ax.set_yticks(np.arange(cm.shape[0]) + 0.5)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")


# 保存模型
def save_model(model, filename):
    dump(model, filename)


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
        X, y = load_data('all_data.csv', feature_type=feature_type)

        # 拆分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 定义MLP模型
        mlp_model = MLPClassifier(random_state=123)
        mlp_params = {
            "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            "activation": ['tanh', 'relu'],
            "solver": ['sgd', 'adam'],
            "learning_rate_init": [0.001, 0.01],
            "max_iter": [10000]
        }

        # 训练模型
        mlp_grid_search = train_model(X_train, y_train, mlp_model, mlp_params, kf)
        print(f"Best parameters for {feature_type}: {mlp_grid_search.best_params_}")

        # 评估模型
        mlp_metrics, mlp_pred_binary = evaluate_model(mlp_grid_search.best_estimator_, X_test, y_test)

        # 将特征类型和评估指标加入结果列表
        result = {'feature_type': feature_type, **mlp_metrics}  # 添加特征类型到结果中
        all_results.append(result)

        # 绘制混淆矩阵
        cm = confusion_matrix(y_test, mlp_pred_binary)
        plot_confusion_matrix(cm, class_names=['Class 0', 'Class 1'])
        plt.show()

    # 保存所有评估结果到CSV文件
    save_results_to_csv(all_results, 'mlp_results.csv')
    print("Results saved to mlp_results.csv")
