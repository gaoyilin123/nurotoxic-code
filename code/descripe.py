# 文件名: descriptors.py

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np

def compute_descriptors(smiles_list):
    """计算给定SMILES列表的化学描述符，并返回包含描述符的DataFrame."""
    descriptor_names = [desc[0] for desc in Descriptors._descList]  # 获取所有描述符的名称
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            # 为每个分子计算描述符
            descriptor_values = [desc[1](mol) if mol else np.nan for desc in Descriptors._descList]
            descriptor_dict = dict(zip(descriptor_names, descriptor_values))
            results.append(descriptor_dict)
        else:
            # 如果分子不能从SMILES解析，则返回包含NaN的字典
            results.append({name: np.nan for name in descriptor_names})

    # 创建DataFrame
    df = pd.DataFrame(results)
    return df

def save_descriptors_to_csv(df, filename):
    """将描述符数据保存到CSV文件."""
    df.to_csv(filename, index=False)


filepath = 'cleaned_alldata.csv'
output_csv_path = 'descriptors_output.csv'

# 读取SMILES
df = pd.read_csv(filepath)
smiles_list = df['Standardized_SMILES'].tolist()

# 计算描述符
descriptor_df = compute_descriptors(smiles_list)
descriptor_df['class'] = df['class']  # 添加类别列

# 保存到CSV
descriptor_df.to_csv(output_csv_path, index=False)