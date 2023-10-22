import pandas as pd
import numpy as np
from rdkit.Chem import AllChem as Chem

# =================
# text io functions
# ==================

def smiles_to_mol(smiles):  # 将SMILES字符串转换为化学分子对象
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        pass
    return None

def verify_smiles(smile):   # 验证SMILES字符串是否有效
    return (smile != '') and pd.notnull(smile) and (Chem.MolFromSmiles(smile) is not None)

def pad_smile(string, max_len, padding='right'):  # 对SMILES字符串进行填充，使其达到指定长度
    if len(string) <= max_len:
        if padding == 'right':
            return string + " " * (max_len - len(string))
        elif padding == 'left':
            return " " * (max_len - len(string)) + string
        elif padding == 'none':
            return string

def filter_valid_length(strings, max_len):  # 过滤超出指定长度的SMILES字符串
    return [s for s in strings if len(s) <= max_len]

def smiles_to_hot(smiles, max_len, padding, char_indices, nchars):  # 将SMILES字符串转换为one-hot编码
    smiles = [pad_smile(i, max_len, padding) for i in smiles if pad_smile(i, max_len, padding)]
    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)
    for i, smile in enumerate(smiles):
        for t, char in enumerate(smile):
            X[i, t, char_indices[char]] = 1
    return X

def hot_to_smiles(hot_x, indices_chars):  # 从one-hot编码转换回SMILES字符串
    smiles = []
    for x in hot_x:
        temp_str = ""
        for j in x:
            index = np.argmax(j)
            temp_str += indices_chars[index]
        smiles.append(temp_str)
    return smiles

def load_smiles(smi_file, max_len=None):  # 从文件中加载SMILES字符串并返回它们
    with open(smi_file, 'r') as f:
        smiles = f.readlines()
    smiles = [i.strip() for i in smiles]
    if max_len is not None:
        smiles = filter_valid_length(smiles, max_len)
    return smiles

def smiles2one_hot_chars(smi_list):  # 创建并存储SMILES中的唯一字符集
    char_lists = [list(smi) for smi in smi_list]
    chars = list(set([char for sub_list in char_lists for char in sub_list]))
    chars.append(' ')
    return chars

def canon_smiles(smi):  # 返回给定的mol对象的SMILES表示，可能是规范化的
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi), isomericSmiles=True, canonical=True)

if __name__ == '__main__':
    smiles = load_smiles("zinc/250k_rndm_zinc_drugs_clean_5.csv", 120)
    print(smiles[:5])
