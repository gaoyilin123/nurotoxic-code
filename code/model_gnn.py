import pandas as pd
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import dgl
from dgl import DGLGraph
from dgllife.model.gnn import WeaveGNN
from dgllife.model.readout import WeightedSumAndMax, SumAndMax

from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, GridSearchCV

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from early_stop import EarlyStopping
from utils import *
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from sklearn.metrics import mean_squared_error  # MSE

PWD = os.path.abspath(os.path.dirname(__file__))


###########
## split train test
###########

def split_kn_train_test():
    df = pd.read_csv('cleaned_alldatagnn.csv', header=0)
    df = shuffle(df)
    df_a = df[df['Label'] == 0]
    df_n = df[df['Label'] == 1]
    df = pd.concat([df_a, df_n], axis=0)
    print(df['Label'].value_counts())
    df_train, df_test = train_test_split(df, random_state=42, test_size=0.2, stratify=df['Label'])
    df_train.to_csv('classify/train_an.csv', sep='\t', header=True, index=False)
    df_test.to_csv('classify/test_an.csv', sep='\t', header=True, index=False)


split_kn_train_test()




###########
## feature
###########
# def get_atom_features(atom):
#     atom_features = [atom.GetAtomicNum()]
#     return np.array(atom_features)

# def get_bond_features(bond):
#     return np.array([1.0])

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))



def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'DU']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D
    ])
    return np.array(atom_features)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats)

def Graph_smiles(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    g = dgl.graph(([], []))  # 使用推荐的方法创建图
    g.add_nodes(molecule.GetNumAtoms())
    node_features = []
    edge_features = []
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        atom_i_features = get_atom_features(atom_i)
        node_features.append(atom_i_features)
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                g.add_edges(i, j)
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)
    g.ndata['h'] = torch.from_numpy(np.array(node_features)).float()
    g.edata['e'] = torch.from_numpy(np.array(edge_features)).float()
    return g


###########
## loader
###########
def collate(sample):
    graphs, labels = map(list, zip(*sample))
    batched_graph = dgl.batch(graphs)
    batched_graph.set_n_initializer(dgl.init.zero_initializer)
    batched_graph.set_e_initializer(dgl.init.zero_initializer)
    return batched_graph, torch.tensor(labels)


def load_data_all_notorch(tuple_ls):
    """
    args:
        ls: [(feature,label)]
    """

    random.shuffle(tuple_ls)
    return tuple_ls


def load_data_all_batchsize(tuple_ls, batchsize, graph=False, drop_last=False):
    """
    args:
        ls: [(feature,label)]
        batchsize: int
    """
    if not graph:
        return DataLoader(tuple_ls, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
    else:
        return DataLoader(tuple_ls, batch_size=batchsize, shuffle=True, collate_fn=collate, drop_last=drop_last)


def load_data_kfold_notorch(tuple_ls, Stratify):
    """
    args:
        ls: [(feature,label)]
    """
    random.shuffle(tuple_ls)
    features, labels = list(zip(*tuple_ls))
    if Stratify:
        kf = StratifiedKFold(n_splits=5, shuffle=True)
    else:
        kf = KFold(n_splits=5, shuffle=True)
    kfolds = []
    for train_idxs, val_idxs in kf.split(features, labels):
        trains = [tuple_ls[index] for index in train_idxs]
        vals = [tuple_ls[index] for index in val_idxs]
        kfolds.append((trains, vals))
    return kfolds


def load_data_kfold_batchsize(tuple_ls, batchsize, Stratify=True, graph=False, drop_last=False):
    """
    args:
        ls: [(feature,label)]
        batchsize: int
    """
    features, labels = list(zip(*tuple_ls))
    if Stratify:
        kf = StratifiedKFold(n_splits=5, shuffle=True)
    else:
        kf = KFold(n_splits=5, shuffle=True)
    kfolds = []
    if not graph:
        for train_idxs, val_idxs in kf.split(features, labels):
            trains = [tuple_ls[index] for index in train_idxs]
            trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=None, drop_last=drop_last)
            vals = [tuple_ls[index] for index in val_idxs]
            vals = DataLoader(vals, batch_size=len(vals), shuffle=True, )
            kfolds.append((trains, vals))
    else:
        for train_idxs, val_idxs in kf.split(features, labels):
            trains = [tuple_ls[index] for index in train_idxs]
            trains = DataLoader(trains, batch_size=batchsize, shuffle=True, collate_fn=collate, drop_last=drop_last)
            vals = [tuple_ls[index] for index in val_idxs]
            vals = DataLoader(vals, batch_size=len(vals), shuffle=True, collate_fn=collate, drop_last=drop_last)
            kfolds.append((trains, vals))
    return kfolds


def load_data(tuple_ls, featurizer=None, if_all=False, Stratify=False, if_torch=False, batchsize=32, graph=True,
              drop_last=False):
    mols, labels = map(list, zip(*tuple_ls))
    features = np.array([featurizer(mol) for mol in mols])
    tuple_ls = list(zip(features, labels))
    if if_all:
        if if_torch:
            return load_data_all_batchsize(tuple_ls, batchsize, graph, drop_last)
        else:
            return load_data_all_notorch(tuple_ls)
    else:
        if if_torch:
            return load_data_kfold_batchsize(tuple_ls, batchsize, graph, drop_last)
        else:
            return load_data_kfold_notorch(tuple_ls)


###########
## utils
###########
def regress_metrics(labels, preds):
    return np.array(
        [
            r2_score(labels, preds),
            mean_absolute_error(labels, preds),
            mean_squared_error(labels, preds),
            sqrt(mean_squared_error(labels, preds)),
        ]
    )


def plt_multi_cor(y_train, y_test, y_pred_train, y_pred_test, name):
    total = list(y_test) + list(y_train)
    mt = regress_metrics(y_test, y_pred_test)
    df_train = pd.DataFrame(
        {'Experiment pT': y_train, 'Predicted pT': y_pred_train, 'Group': ['Train Set' for i in range(len(y_train))]})
    df_test = pd.DataFrame(
        {'Experiment pT': y_test, 'Predicted pT': y_pred_test, 'Group': ['Test Set' for i in range(len(y_test))]})
    df_ds = pd.DataFrame(
        {'Experiment pT': total, 'Predicted pT': total, 'Group': ['Experiment Set' for i in range(len(total))]})
    print(r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test), r2_score(total, total))
    df = pd.concat([df_train, df_test, df_ds], axis=0)
    font = {'family': 'Arial', 'size': 15, 'weight': 'medium'}
    plt.rc('font', **font)
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    plt.rc('xtick.major', width=1.5)
    plt.rc('ytick.major', width=1.5)
    g = sns.lmplot(x='Experiment pT', y='Predicted pT', data=df, hue='Group', legend=False)
    ax = g.axes[0, 0]
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    plt.text(
        x=6, y=2,
        s='R^2={:.2f}\nMAE={:.2f}\nMSE={:.2f}\nRMSE={:.2f}'.format(mt[0], mt[1], mt[2], mt[3]), size=15
    )
    plt.subplots_adjust(top=0.95, right=0.95)
    plt.legend(loc="upper left", frameon=False, prop={'size': 14})
    plt.tight_layout()
    plt.savefig('classify/regress_{}.png'.format(name))
    plt.show()


def plot_gnn(rst, savename):
    # rst = np.loadtxt(rst)
    rst = np.array(rst).transpose()
    fig, ax = plt.subplots()
    ax.plot(rst[0], label='loss of training')
    ax.plot(rst[1], label='ACC of training')
    ax.plot(rst[2], label='AUC of training')
    plt.title('GNN predictor training')
    plt.legend()
    plt.tight_layout()
    plt.savefig('classify/' + savename)


def plot_gnn_reg(rst, savename):
    # rst = np.loadtxt(rst)
    rst = np.array(rst).transpose()
    fig, ax = plt.subplots()
    ax.plot(rst[0], label='loss of training')
    ax.plot(rst[1], label='MSE of training')
    plt.title('GNN predictor training')
    plt.legend()
    plt.tight_layout()
    plt.savefig('classify/' + savename)


###########
## model
###########
class WeavePredictor(nn.Module):
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 num_gnn_layers=2,
                 gnn_hidden_feats=128,
                 gnn_activation=F.relu,
                 n_tasks=1):
        super(WeavePredictor, self).__init__()
        self.gnn = WeaveGNN(node_in_feats=node_in_feats,
                            edge_in_feats=edge_in_feats,
                            num_layers=num_gnn_layers,
                            hidden_feats=gnn_hidden_feats,
                            activation=gnn_activation)
        self.readout = WeightedSumAndMax(in_feats=gnn_hidden_feats)
        self.predict = nn.Sequential(
            nn.Linear(2 * gnn_hidden_feats, 64),
            nn.Linear(64, n_tasks),
        )

    def forward(self, g, node_feats, edge_feats):
        node_feats, _ = self.gnn(g, node_feats, edge_feats, node_only=False)
        g_feats = self.readout(g, node_feats)
        return self.predict(g_feats)


def test_model():
    model = WeavePredictor(node_in_feats=26, edge_in_feats=6, n_tasks=1)
    graphs = Graph_smiles('CCO')
    print(graphs)
    ndata = graphs.ndata.pop('h')
    edata = graphs.edata.pop('e')
    model(graphs, ndata, edata)


###########
## train
###########
class train_gnn(object):
    @classmethod
    def train_classify_kfolds(cls, n_atom_feat=1, n_bond_feat=1, n_classes=1, kfolds=None, classify_metrics=None, max_epochs=50, patience=10, save_folder='models/', save_name='gnn.pth', device='cpu'):
        train_losses = []
        train_accs = []
        train_aucs = []
        val_metrics = []
        for train_loader, val_loader in kfolds:
            model = WeavePredictor(node_in_feats=n_atom_feat, edge_in_feats=n_bond_feat, n_tasks=n_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            early_stopping = EarlyStopping(save_folder, save_name, patience=patience)
            for epoch in range(1, max_epochs + 1):
                model.train()
                loss_train = 0.
                auc_train = 0.
                acc_train = 0.
                for batch_idx, (train_graphs, train_labels) in enumerate(train_loader):
                    if batch_idx == 32:  # 如果批次索引为31，则停止当前epoch的循环
                        break
                    graphs, labels = train_graphs.to(device), train_labels.to(device)
                    preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                    loss = CrossEntropyLoss()(preds, labels)
                    loss_train += loss.detach().item()
                    acc, auc = acc_auc(labels.cpu().numpy(), preds.detach().cpu().numpy())
                    acc_train += acc
                    auc_train += auc
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_train /= (batch_idx + 1)
                auc_train /= (batch_idx + 1)
                acc_train /= (batch_idx + 1)
                if epoch % 1 == 0:
                    print('loss:', loss_train, 'ACC:{:.2f}'.format(acc_train), 'AUC:{:.2f}'.format(auc_train))
                    train_losses.append(loss_train)
                    train_accs.append(acc_train)
                    train_aucs.append(auc_train)
                model.eval()
                with torch.no_grad():
                    for val_graphs, val_labels in val_loader:
                        graphs, labels = val_graphs.to(device), val_labels.to(device)
                        preds = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                        loss = CrossEntropyLoss()(preds, labels)
                        loss_val = loss.detach().item()
                        metrics_val = classify_metrics(labels.cpu().numpy(), preds.detach().cpu().numpy())
                early_stopping(loss_val, model)
                if early_stopping.early_stop:
                    val_metrics.append(metrics_val)
                    print("~~~~~Early stopping~~~~")
                    break
                else:
                    if epoch == max_epochs:
                        val_metrics.append(metrics_val)
        print('len(val_metrics):')
        print(len(val_metrics))
        np.savetxt(save_folder + 'val.txt', np.array(val_metrics).mean(0), fmt='%.02f')
        return np.array(val_metrics).mean(0)

    @classmethod
    def train_classify_all(cls, n_atom_feat=1, n_bond_feat=1, n_classes=1, all=None, max_epochs=58, save_folder='models/', save_name='gnn.pth', patience=10, device='cpu'):
        model = WeavePredictor(node_in_feats=n_atom_feat, edge_in_feats=n_bond_feat, n_tasks=n_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        early_stopping = EarlyStopping(save_folder, save_name=save_name, patience=patience)
        rst = []
        for epoch in range(1, max_epochs + 1):
            loss_train = 0.
            acc_train = 0.
            auc_train = 0.
            model.train()
            for batch_idx, (train_graphs, train_labels) in enumerate(all):
                graphs, labels = train_graphs.to(device), train_labels.to(device)
                logits = model(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
                loss = CrossEntropyLoss()(logits, labels)
                loss_train += loss.detach().item()
                acc, auc = acc_auc(labels.cpu().numpy(), logits.detach().cpu().numpy())
                acc_train += acc
                auc_train += auc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_train /= (batch_idx + 1)
            auc_train /= (batch_idx + 1)
            acc_train /= (batch_idx + 1)
            if epoch % 1 == 0:
                print('loss:', loss_train, 'ACC:', acc_train, 'AUC:', auc_train)
                rst.append(np.array([loss_train, acc_train, auc_train]))
            early_stopping(loss_train, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        plot_gnn(rst, 'gnn_an.png')

    @classmethod
    def test_classify(cls, n_atom_feat=1, n_bond_feat=1, n_classes=1, test=None, classify_metrics=None,
                      save_path='models/an_gnn.pth', device='cpu'):
        model_gnn = WeavePredictor(node_in_feats=n_atom_feat, edge_in_feats=n_bond_feat, n_tasks=n_classes).to(device)
        state_dict = torch.load(save_path, map_location=device)
        model_gnn.load_state_dict(state_dict)
        model_gnn.eval()

        all_preds = []
        all_labels = []
        for val_graphs, val_labels in test:
            graphs, labels = val_graphs.to(device), val_labels.to(device)
            preds = model_gnn(graphs, graphs.ndata.pop('h'), graphs.edata.pop('e'))
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 转换为 numpy 数组
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算准确性
        accuracy = accuracy_score(all_labels, all_preds.argmax(axis=1))
        print(f"Test Accuracy: {accuracy:.4f}")

        # 使用 classify_metrics 打印其他评估指标
        classify_metrics(all_labels, all_preds, plot_cm=True, savename='GNN_test',
                         classnames=['Class0', 'Class1'])


###########
## train
###########
def my_gnn_train_bi():
    # 读取训练数据并打印标签的统计信息
    df = pd.read_csv('classify/train_an.csv', sep='\t', header=0)
    print(df.columns)  # 打印数据框的列名

    print(df['Label'].value_counts())  # 打印训练数据标签的统计信息

    # 将 SMILES 和标签转换为元组列表
    tuple_ls = list(zip(list(df['Smiles']), list(df['Label'])))

    # 加载数据并进行 K 折交叉验证训练
    kfolds = load_data(tuple_ls, featurizer=Graph_smiles, if_all=False, Stratify=True, if_torch=True, batchsize=48, graph=True, drop_last=True)
    rst = train_gnn.train_classify_kfolds(n_atom_feat=26, n_bond_feat=6, n_classes=2, kfolds=kfolds, classify_metrics=bi_classify_metrics, max_epochs=58, patience=10, device='cpu')
    print("K-fold cross-validation results:")
    print(rst)  # 打印 K 折交叉验证训练的结果

    # 加载所有数据并进行训练
    all = load_data(tuple_ls, featurizer=Graph_smiles, if_all=True, Stratify=True, if_torch=True, batchsize=48, graph=True, drop_last=True)
    train_gnn.train_classify_all(n_atom_feat=26, n_bond_feat=6, n_classes=2, all=all, max_epochs=58, patience=10,
                                 device='cpu')


    # 读取测试数据并打印标签的统计信息
    df_test = pd.read_csv('classify/test_an.csv', sep='\t', header=0)
    print(df_test['Label'].value_counts())  # 打印测试数据标签的统计信息

    # 将测试数据的 SMILES 和标签转换为元组列表
    tuple_test_ls = list(zip(list(df_test['Smiles']), list(df_test['Label'])))

    # 加载测试数据并进行测试，打印测试集的准确性
    test = load_data(tuple_test_ls, featurizer=Graph_smiles, if_all=True, Stratify=True, if_torch=True, batchsize=1000, graph=True, drop_last=False)
    a = train_gnn.test_classify(n_atom_feat=26, n_bond_feat=6, n_classes=2, test=test, classify_metrics=bi_classify_metrics, save_path='models/gnn.pth', device='cpu')






if __name__ == "__main__":
    my_gnn_train_bi()

    test_model()
    pass
