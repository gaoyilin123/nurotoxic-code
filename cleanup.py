import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def has_inorganic(mol):
    # 统计分子中碳原子的数量，如果小于等于1，则认为是无机物
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)  # 碳的原子序数为6
    if carbon_count <= 1:
        return True
    else:
        return False


def standardize_smi(smiles,basicClean=True,clearCharge=False, clearFrag=True, canonTautomer=True, isomeric=False):
    try:
        clean_mol = Chem.MolFromSmiles(smiles)
        # 除去氢、金属原子、标准化分子
        if basicClean:
            clean_mol = rdMolStandardize.Cleanup(clean_mol)
        if clearFrag:
            # 仅保留主要片段作为分子
            clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        # 尝试中性化处理分子
        if clearCharge:
            uncharger = rdMolStandardize.Uncharger()
            clean_mol = uncharger.uncharge(clean_mol)
        # 处理互变异构情形，这一步在某些情况下可能不够完美
        if canonTautomer:
            te = rdMolStandardize.TautomerEnumerator() # idem
            clean_mol = te.Canonicalize(clean_mol)
        #set to True 保存立体信息，set to False 移除立体信息，并将分子存为标准化后的SMILES形式
        stan_smiles=Chem.MolToSmiles(clean_mol, isomericSmiles=isomeric)
    except Exception as e:
        print (e, smiles)
        return None
    return stan_smiles

def process_and_save_to_csv(file_path, output_file_path):
    # 读取 CSV 文件
    data = pd.read_csv(file_path)
    # 删除 'SMILES' 列中的空值
    data.dropna(subset=['SMILES'], inplace=True)

    # 对每个 SMILES 进行标准化处理
    standardized_smiles = []
    for smiles in data['SMILES']:
        standardized_smiles.append(standardize_smi(smiles))

    # 创建新的 DataFrame
    df_standardized = pd.DataFrame({'Standardized_SMILES': standardized_smiles})

    # 去除含有无机物的分子
    df_standardized = df_standardized.drop(df_standardized[df_standardized['Standardized_SMILES'].apply(lambda x: has_inorganic(Chem.MolFromSmiles(x)))].index)

    # 将结果保存到 CSV 文件中
    # 将结果保存到 CSV 文件中
    df_standardized.drop_duplicates().to_csv(output_file_path, index=False)


# 示例使用
file_path = 'positive.csv'
output_file_path = 'positive.csv'
process_and_save_to_csv(file_path, output_file_path)
