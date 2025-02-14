from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import dump
import math
import matplotlib.pyplot as plt
import seaborn as sns


# 定义指纹特征提取方法
def load_data(filepath, feature_type='TopologicalTorsion'):
    df = pd.read_csv(filepath)

    # 根据选择的特征提取方法加载数据
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


# 训练模型并进行概率校准
def train_model(X_train, y_train, base_model, params, cv_splitter):
    """
    Train a model using GridSearchCV. After finding the best parameters,
    fit a CalibratedClassifierCV for probability calibration.
    """
    # 使用 GridSearchCV 查找最佳的 SVC 参数
    gc = GridSearchCV(base_model, param_grid=params, cv=cv_splitter, scoring='roc_auc', return_train_score=True,
                      verbose=2)
    gc.fit(X_train, y_train)

    # 使用找到的最佳模型进行概率校准
    best_model = gc.best_estimator_
    calibrated_model = CalibratedClassifierCV(best_model, cv='prefit')
    calibrated_model.fit(X_train, y_train)
    return calibrated_model, gc.best_params_


# 评估模型性能
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict_proba(X_test)[:, 1]
    auc_roc_score = roc_auc_score(y_test, y_pred)
    y_pred_binary = [round(y, 0) for y in y_pred]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    metrics = calculate_metrics(tp, tn, fp, fn)
    metrics['auc_roc_score'] = auc_roc_score
    return metrics, y_pred_binary


# 计算常见的性能指标
def calculate_metrics(tp, tn, fp, fn):
    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    q = (tp + tn) / (tp + fn + tn + fp)
    mcc = (tp * tn - fn * fp) / math.sqrt((tp + fn) * (tp + fp) * (tn + fn) * (tn + fp))
    P = tp / (tp + fp)
    F1 = (P * se * 2) / (P + se)
    BA = (se + sp) / 2
    return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'se': se, 'sp': sp, 'mcc': mcc, 'q': q, 'P': P, 'F1': F1, 'BA': BA}


# 保存结果到 CSV 文件
def save_results_to_csv(results, filename):
    results_df = pd.DataFrame(results)
    results_df.to_csv(filename, index=False)


# 训练和评估多个模型
def train_and_evaluate_multiple_models(input_filepath, output_filepath):
    # 定义要使用的特征提取方法和模型
    feature_types = ['Morgan', 'MACCS', 'TopologicalTorsion', 'RDK']
    models = {
        'SVM': SVC(probability=True, random_state=42),
    }
    svm_params = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    # 定义交叉验证方法
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 存储所有模型的评估结果
    all_results = []

    for feature_type in feature_types:
        # 加载数据
        X, y = load_data(input_filepath, feature_type)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 遍历模型
        for model_name, model in models.items():
            print(f"Training model: {model_name} with {feature_type} features")

            # 超参数调优
            calibrated_model, best_params = train_model(X_train, y_train, model, svm_params, kf)
            print(f"Best parameters for {model_name} with {feature_type} features: {best_params}")

            # 在独立的测试集上评估模型
            metrics, _ = evaluate_model(calibrated_model, X_test, y_test)

            # 将评估结果添加到字典
            metrics['model'] = model_name
            metrics['feature_type'] = feature_type
            metrics['best_params'] = str(best_params)

            # 保存所有结果
            all_results.append(metrics)

    # 将结果保存到 CSV 文件
    save_results_to_csv(all_results, output_filepath)
    print(f"Results saved to {output_filepath}")


if __name__ == "__main__":
    input_filepath = 'cleaned_alldata.csv'  # 输入数据文件路径
    output_filepath = 'svm_results.csv'  # 输出结果文件路径
    train_and_evaluate_multiple_models(input_filepath, output_filepath)
