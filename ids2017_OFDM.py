import higher
import random
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.decomposition import PCA
from math import acos, degrees
from sklearn.metrics import accuracy_score,f1_score
from kl_divergence import *
from ft_transformer import FTTransformer
from onlineDNN import *
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model0 = FTTransformer(
        categories = [],      # tuple containing the number of unique values within each category
        num_continuous = 76,                # number of continuous values #77,task_all_features
        dim = 128,                           # dimension, paper set at 32
        dim_out = 15,                        # binary prediction, but could be anything
        depth = 4,                          # depth, paper recommended 6
        heads = 11,                          # heads, paper recommends 8
        attn_dropout = 0.034,                 # post-attention dropout
        ff_dropout = 0.104,                   # feed forward dropout
    ).to(device)
model0.load_state_dict(torch.load("../../data_paper1/CSE-CIC-IDS2018/model/pre/ftt_pre.pt"))
model1 = OnlineDNN(76,128,15).to(device) #77
model1.load_state_dict(torch.load("../../data_paper1/CSE-CIC-IDS2018/model/pre/online_dnn_model.pt"))

# 定义基本分类器
# class BaseClassifier(nn.Module):
#     def __init__(self, models):
#         super(BaseClassifier, self).__init__()
#         self.models = nn.ModuleList(models)
#
#     def forward(self, x):
#         preds = []
#         for model_all_features in self.models:
#             pred = model_all_features.predict(x)
#             preds.append(pred)
#         return preds

# 定义数据集
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 添加一个函数来处理分类任务
# def classify_with_remaining_models(models, data_loader, device, excluded_index):
#     combined_true_labels = []
#     combined_pred_labels = []
#
#     # 对未参与更新的模型进行分类
#     for idx, model_all_features in enumerate(models):
#         if idx == excluded_index:
#             continue
#         true_labels = []
#         pred_labels = []
#         with torch.no_grad():
#             for inputs, labels in data_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model_all_features(inputs)
#                 _, predicted = torch.max(outputs.data_paper1, 1)
#                 true_labels.extend(labels.cpu().numpy())
#                 pred_labels.extend(predicted.cpu().numpy())
#         # 计算单个模型的准确率
#         accuracy = accuracy_score(true_labels, pred_labels)
#         print(f"Model {idx} Classification Accuracy: {accuracy:.4f}")
#         combined_true_labels.extend(true_labels)
#         combined_pred_labels.extend(pred_labels)
#
#     # 可以计算组合模型的准确率
#     combined_accuracy = accuracy_score(combined_true_labels, combined_pred_labels)
#     print(f"Combined Classification Accuracy: {combined_accuracy:.4f}")
#     return combined_accuracy
def detect_data_drift_kl(data_old, data_new, kl_threshold=0.3):
    kl_drift_detected = calculate_kl_divergence(data_old, data_new) > kl_threshold
    return kl_drift_detected

# PCA based drift detection
def detect_data_drift_pca(data_old, data_new):
    scaler = StandardScaler()
    data_old_standardized = scaler.fit_transform(data_old)
    data_new_standardized = scaler.transform(data_new)

    pca_old = PCA().fit(data_old_standardized)
    pca_new = PCA().fit(data_new_standardized)

    angles = []
    for i in range(len(pca_old.components_)):
        dot_product = np.dot(pca_old.components_[i], pca_new.components_[i].T)
        dot_product = min(1, max(-1, dot_product))
        angle = acos(dot_product)
        angles.append(degrees(angle))

    drift_detected = any(angle >= 60 for angle in angles)
    return drift_detected

# 加载测试数据
def load_test_data(test_data_dir):
    test_files = [file for file in os.listdir(test_data_dir) if file.endswith(".csv")]
    test_data = pd.concat([pd.read_csv(os.path.join(test_data_dir, f)) for f in test_files])
    features = torch.tensor(test_data.drop(columns=['Label']).values, dtype=torch.float32)
    labels = torch.tensor(test_data['Label'].values, dtype=torch.long)
    dataset = CustomDataset(features, labels)
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
    return data_loader

# 定义一个函数用于计算准确率和F1指标
def calculate_accuracy_and_f1(model, data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    model.train()
    return accuracy, f1

def ft_calculate_accuracy_and_f1(model, data_loader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(torch.empty(inputs.size(0), 0).to(device), inputs)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    model.train()
    return accuracy, f1

def sample_tasks(old_data_loader, new_data_loader, num_samples_per_task=32):
    """
    从旧数据和新数据加载器中随机采样任务数据集。

    参数:
    - old_data_loader: 代表"旧数据"的DataLoader，通常用作支持集。
    - new_data_loader: 代表"新数据"的DataLoader，通常用作查询集。
    - num_samples_per_task: 每个任务（支持集和查询集）中样本的数量。

    返回:
    - 两个元组，分别代表一个模拟任务的“支持集”和“查询集”的数据加载器。
    """
    # 确保加载器有足够数据
    assert len(old_data_loader.dataset) >= num_samples_per_task, "Not enough samples in the old data_paper1 loader"
    assert len(new_data_loader.dataset) >= num_samples_per_task, "Not enough samples in the new data_paper1 loader"

    # 随机采样索引
    old_indices = random.sample(range(len(old_data_loader.dataset)), num_samples_per_task)
    new_indices = random.sample(range(len(new_data_loader.dataset)), num_samples_per_task)

    # 创建子采样器
    old_sampler = SubsetRandomSampler(old_indices)
    new_sampler = SubsetRandomSampler(new_indices)

    # 使用子采样器创建新的数据加载器
    sampled_old_data_loader = torch.utils.data.DataLoader(old_data_loader.dataset,
                                                          batch_size=old_data_loader.batch_size,
                                                          sampler=old_sampler)
    sampled_new_data_loader = torch.utils.data.DataLoader(new_data_loader.dataset,
                                                          batch_size=new_data_loader.batch_size,
                                                          sampler=new_sampler)

    return sampled_old_data_loader, sampled_new_data_loader

class DriftDetection:
    def __init__(self):
        self.warning_state = False
        self.warning_counter = 0

    def detect_drift(self, data_old, data_new):
        drift_pca = detect_data_drift_pca(data_old, data_new)
        drift_kl = detect_data_drift_kl(data_old, data_new)

        if drift_pca and drift_kl:
            self.reset_warning_state()
            return True
        elif drift_pca or drift_kl:
            if self.warning_state:
                self.warning_counter += 1
                if self.warning_counter >= 3:
                    self.reset_warning_state()
                    return True
            else:
                self.warning_state = True
                self.warning_counter = 1
        else:
            self.reset_warning_state()

        return False

    def reset_warning_state(self):
        self.warning_state = False
        self.warning_counter = 0

detector = DriftDetection()

# 加载测试数据
test_data_loader = load_test_data("task_test/test")

# 初始化基础分类模型
models = [model0, model1]
#classifier = BaseClassifier(models)

# 加载所有CSV文件
data_files = sorted([file for file in os.listdir("task_test/train") if file.endswith(".csv")])
previous_data = None
normal_training_counter = 0
normal_training_threshold = 5
# 保存结果的列表
results = []

# 训练循环
for file_index, file_name in enumerate(data_files):
    # 加载新数据
    data = pd.read_csv(os.path.join("task_test/train", file_name))
    new_features = torch.tensor(data.drop(columns=['Label']).values, dtype=torch.float32)
    new_labels = torch.tensor(data['Label'].values, dtype=torch.long)
    new_dataset = CustomDataset(new_features, new_labels)
    new_data_loader = DataLoader(new_dataset, batch_size=128, shuffle=True)

    if previous_data is not None:
        # 加载旧数据
        old_features = torch.tensor(previous_data.drop(columns=['Label']).values, dtype=torch.float32)
        old_labels = torch.tensor(previous_data['Label'].values, dtype=torch.long)
        old_dataset = CustomDataset(old_features, old_labels)
        old_data_loader = DataLoader(old_dataset, batch_size=128, shuffle=True)

        # 检测数据漂移
        drift_detected = detector.detect_drift(previous_data, data)
        print("检测到数据漂移" if drift_detected else "未检测到数据漂移")

        if drift_detected:
            # 进行在线学习和元学习
            for idx,model in enumerate(models):
                if isinstance(model, OnlineDNN):
                    if isinstance(model, OnlineDNN):
                        optimizer = optim.Adam(model.parameters(), lr=0.0001)
                        scheduler = CosineAnnealingLR(optimizer, T_max=20)
                        for epoch in range(20):
                            for batch_index, (inputs, targets) in enumerate(new_data_loader):
                                optimizer.zero_grad()
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = model(inputs)
                                loss = nn.CrossEntropyLoss()(outputs, targets.long().squeeze(-1))
                                loss.backward()
                                optimizer.step()
                                scheduler.step()
                                model.update_alpha(loss.item())
                                del inputs, targets, outputs, loss  # 手动释放显存
                                torch.cuda.empty_cache()  # 清理缓存

                        torch.save(model.state_dict(), f'model/online/online_dnn_{file_index}.pt')
                else:
                    # 其他模型用于分类
                    acc, f1 = ft_calculate_accuracy_and_f1(model, test_data_loader, device)
                    print(f"Model {idx}在新到达数据{file_index}更新时的测试准确率: {acc:.4f},, F1 Score: {f1:.4f}")
                    results.append(f"Model{idx}在数据流{file_index}上的测试准确率: {acc:.4f}, F1得分: {f1:.4f}")
        else:
            normal_training_counter += 1
            if normal_training_counter >= normal_training_threshold:
                # 进行元学习训练
                for idx, model in enumerate(models):
                    if not isinstance(model, OnlineDNN):
                        print("超过更新阈值，进入元学习训练")
                        meta_lr = 0.0001
                        inner_lr = 0.001
                        num_tasks_per_epoch = 30  # 假设每个epoch处理两个任务
                        num_inner_steps = 5
                        num_epochs = 50

                        meta_optimizer = torch.optim.Adamax(model.parameters(), lr=inner_lr)
                        inner_optimizer = optim.SGD(model.parameters(), lr=inner_lr)
                        old_features = torch.tensor(previous_data.drop(columns=['Label']).values, dtype=torch.float32)
                        old_labels = torch.tensor(previous_data['Label'].values, dtype=torch.long)
                        old_dataset = CustomDataset(old_features, old_labels)
                        old_data_loader = DataLoader(old_dataset, batch_size=128, shuffle=True)

                        for epoch in range(num_epochs):
                            for _ in range(num_tasks_per_epoch):  # 对于每个任务
                                # 模拟从任务分布中采样支持集和查询集（这里简化处理，实际应用中应更复杂）
                                sampled_old_data, sampled_new_data = sample_tasks(old_data_loader, new_data_loader)  # 这里需要你实现sample_tasks函数

                                # 外循环（元更新）
                                with higher.innerloop_ctx(model, inner_optimizer, copy_initial_weights=False) as (fmodel, diffopt):
                                    # 内循环（任务适应）
                                    for _ in range(num_inner_steps):
                                        # 假设 sampled_new_data 是包含特征和标签的数据加载器
                                        new_numer_inputs, new_labels = next(iter(sampled_new_data))  # 这次我们保留了标签
                                        new_numer_inputs, new_labels = new_numer_inputs.to(device), new_labels.to(device)
                                        # 现在，new_labels 就包含了当前batch的类标签，您可以使用它来计算损失、评估模型性能等
                                        new_outputs = fmodel(torch.empty(new_numer_inputs.size(0), 0).to(device), new_numer_inputs)
                                        inner_loss = nn.CrossEntropyLoss()(new_outputs, new_labels)
                                        diffopt.step(inner_loss)  # 使用 diffopt.step 而不是 inner_optimizer.step

                                    # 适应新任务（模拟查询集）
                                    query_inputs, query_targets = next(iter(sampled_new_data))  # 从查询集中获取数据
                                    query_inputs, query_targets = query_inputs.to(device), query_targets.to(device)
                                    query_outputs = fmodel(torch.empty(query_inputs.size(0), 0).to(device), query_inputs)
                                    outer_loss = nn.CrossEntropyLoss()(query_outputs, query_targets)

                                meta_optimizer.zero_grad()
                                outer_loss.backward()  # 反向传播外循环的损失
                                meta_optimizer.step()

                            # 计算并打印准确率
                            accuracy,f1 = ft_calculate_accuracy_and_f1(model, test_data_loader, device)
                            print(f"MamlEpoch {epoch + 1}: Average Loss: {outer_loss.item():.4f}, "
                                  f"Test Accuracy: {accuracy:.4f}, , F1 Score: {f1:.4f}")

                        torch.save(model.state_dict(), f'model/offline/meta_ft_model_{idx}.pt')
                    else:
                        # 在线模型分类
                        acc, f1 = calculate_accuracy_and_f1(model, test_data_loader, device)
                        print(f"Model {idx}在新到达数据流{file_index}更新时的测试准确率: {acc:.4f}, F1 Score: {f1:.4f}")
                        results.append(f"Model {idx}在数据流{file_name}上的检测准确率: {acc:.4f}, F1得分: {f1:.4f}")
                normal_training_counter = 0
            else:
                # 进行正常反向传播
                for idx, model in enumerate(models):
                    if isinstance(model, FTTransformer):
                        print("更新Transformer，onlineDNN负责入侵检测")
                        optimizer = optim.Adam(model.parameters(), lr=0.0001)
                        scheduler = CosineAnnealingLR(optimizer, T_max=20)
                        for epoch in range(20):
                            for batch_index, (inputs, targets) in enumerate(new_data_loader):
                                optimizer.zero_grad()
                                inputs, targets = inputs.to(device), targets.to(device)
                                outputs = model(torch.empty(inputs.size(0), 0).to(device), inputs)
                                loss = nn.CrossEntropyLoss()(outputs, targets)
                                loss.backward()
                                optimizer.step()
                                # 更新学习率
                                scheduler.step()
                                accuracy,f1 = ft_calculate_accuracy_and_f1(model, test_data_loader, device)
                                print(f"Model ({type(model).__name__}): NormalEpoch {epoch + 1}, Batch {batch_index + 1}: Train Loss {loss.item():.4f}, Test Accuracy {accuracy:.4f}, F1 Score: {f1:.4f}")
                        torch.save(model.state_dict(), f'model/offline/fttransformer_model_{file_index}.pt')
                    else:
                        # 在线模型分类
                        acc,f1 = calculate_accuracy_and_f1(model, test_data_loader, device)
                        print(f"Model {idx}在新到达数据流{file_index}更新时的测试准确率: {acc:.4f}, F1 Score: {f1:.4f}")
                        results.append(f"Model {idx}在数据流{file_name}上的检测准确率: {acc:.4f}, F1得分: {f1:.4f}")
    else:
        print("第一次加载数据，跳过漂移检测")

    previous_data = data

# 将结果写入文件
with open("results.txt", "w") as f:
    for result in results:
        f.write(result + "\n")
#数据筛选，模型设计
#没必要使用三个模型，两个即可
#两种漂移检测相结合，当两种方式都检测到漂移时，进行一次在线更新，当一种方法检测到时，进入预警状态，再次有一种出现时，进行在线更新，如何结合元学习
#0621,主要完成论文撰写，开始对比实验并绘制曲线图，柱状图，混淆矩阵等实验结果
#调高18和19的警告阈值