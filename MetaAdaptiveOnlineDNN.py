import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据读取函数
def read_csv_files(folder_path, num_files, file_prefix):
    data_list = []
    labels_list = []
    for i in range(num_files):
        file_path = os.path.join(folder_path, f"{file_prefix}_task{i}.csv")
        df = pd.read_csv(file_path)
        data = df.iloc[:, :-1].values
        labels = df.iloc[:, -1].values
        data_list.append(data)
        labels_list.append(labels)
    data = np.vstack(data_list)
    labels = np.hstack(labels_list)
    return data, labels

# 模型定义
class MetaAdaptiveOnlineDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=12, n_classifiers=3, eta=0.1):
        super(MetaAdaptiveOnlineDNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - n_classifiers):
            self.hidden_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.classifiers = nn.ModuleList()
        for _ in range(n_classifiers):
            self.classifiers.append(nn.Linear(hidden_dim, output_dim))
        self.softmax = nn.Softmax(dim=1)
        self.alpha = torch.ones(n_classifiers, requires_grad=False).to(device) / n_classifiers  # 初始化alpha，均分权重
        self.eta = eta
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        outputs = []
        for hidden in self.hidden_layers:
            x = F.relu(hidden(x))
            x = self.dropout(x)
        for classifier in self.classifiers:
            output = classifier(x)
            outputs.append(output)
        # 使用alpha进行加权求和
        final_output = sum(a * o for a, o in zip(self.alpha, outputs))
        return final_output, outputs  # 返回最终输出和每层的输出

    # def update_alpha(self, losses):
    #     # 使用每层的损失值更新alpha，损失越大，权重越小
    #     losses = torch.tensor(losses).to(device)
    #     # 反转损失以便损失大的层权重小
    #     losses = losses - losses.min() + 1e-8  # 确保没有负数
    #     inverted_losses = 1.0 / (losses + 1e-8)  # 损失越大，权重越小
    #     self.alpha = inverted_losses / inverted_losses.sum()  # 归一化，保证权重和为1
    def update_alpha(self, losses):
        """
        根据每一层的损失值更新权重 alpha，损失大的权重小
        """
        # 如果 losses 已经是 float 类型的列表，直接转换为 tensor
        if isinstance(losses[0], float):
            losses = torch.tensor(losses, dtype=torch.float32).to(device)
        else:
            losses = torch.tensor([loss.item() for loss in losses], dtype=torch.float32).to(device)

        # 计算损失倒数以避免除零错误
        inv_losses = 1 / (losses + 1e-8)
        total_inv_loss = inv_losses.sum()
        self.alpha = inv_losses / total_inv_loss  # 归一化权重，确保权重和为1


# 训练函数
def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0  # 记录每一轮的总损失
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            final_output, layer_outputs = model(inputs)  # 获取最终输出和每层的输出

            # 计算每层分类器的损失
            losses = []
            for output in layer_outputs:
                loss = criterion(output, targets)
                losses.append(loss.item())  # 每层的损失值

            model.update_alpha(losses)  # 根据每层损失更新alpha权重

            # 计算最终加权后的损失
            final_loss = criterion(final_output, targets)
            final_loss.backward()  # 反向传播
            optimizer.step()

            # 累计损失
            running_loss += final_loss.item()

        # 调度学习率
        scheduler.step()

        # 在验证集上评估
        model.eval()
        val_predictions = []
        val_labels = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs, _ = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(targets.cpu().numpy())

        # 计算验证集准确率
        accuracy = accuracy_score(val_labels, val_predictions)

        # 打印每一轮的损失和验证集准确率
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Val Accuracy: {accuracy:.4f}")

# 测试函数
def test(model, test_data, test_labels, device):
    model.to(device)
    model.eval()
    test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_labels, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs, _ = model(test_data)
        _, predicted = torch.max(outputs, 1)

    y_true = test_labels.cpu().numpy()
    y_pred = predicted.cpu().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return accuracy, precision, recall, f1

# 主函数
def main():
    # 文件路径和参数设置
    train_folder_path = "task_new/train/"
    test_folder_path = "task_new/test/"
    num_files = 10
    batch_size = 128
    epochs = 50
    learning_rate = 0.001
    hidden_dim = 128
    output_dim = 15  # 根据实际分类数设置
    n_layers = 12

    # 读取数据
    train_data, train_labels = read_csv_files(train_folder_path, num_files, 'train')
    test_data, test_labels = read_csv_files(test_folder_path, num_files, 'test')

    # 数据预处理
    # scaler = StandardScaler()
    # train_data = scaler.fit_transform(train_data)
    # test_data = scaler.transform(test_data)

    x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    # 转换为 Tensor
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    x_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、优化器和学习率调度器
    model = MetaAdaptiveOnlineDNN(input_dim=x_train.shape[1], hidden_dim=hidden_dim, output_dim=output_dim, n_layers=n_layers)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # 训练和测试
    train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=epochs)
    accuracy, precision, recall, f1 = test(model, test_data, test_labels, device)

    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    torch.save(model.state_dict(), 'model/pre/metaadaptive_online_dnn_model_new.pt')

if __name__ == "__main__":
    main()
