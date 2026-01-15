import numpy as np
import pandas as pd
import torch
import warnings
from sklearn.preprocessing import StandardScaler
from botorch.models import KroneckerMultiTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.models.transforms import Normalize, Standardize
from botorch.models.transforms.input import InputStandardize

# 忽略所有警告
warnings.filterwarnings("ignore")

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 读取初始数据
experiment_data = pd.read_excel(r'D:\\7华工\第一个工作\LHS+器件性能.xlsx')
X = experiment_data.iloc[:, 0:8].values
Y = experiment_data.iloc[:, 8:11].values

# 2. 定义标准化器（对X和Y都做标准化），只在初始数据fit
x_scaler = StandardScaler()
y_scaler = StandardScaler()
x_scaler.fit(X)
y_scaler.fit(Y)

X_scaled = x_scaler.transform(X)
Y_scaled = y_scaler.transform(Y)

# 转换为PyTorch张量
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

# 3. 定义搜索空间(实验参数范围，标准化后为0-1附近)
param_min = np.min(X, axis=0)
param_max = np.max(X, axis=0)
bounds_np = np.vstack([param_min, param_max])
bounds_scaled = np.vstack([
    x_scaler.transform(bounds_np[0:1])[0],  # min
    x_scaler.transform(bounds_np[1:2])[0]   # max
])
bounds = torch.tensor(bounds_scaled, dtype=torch.float32)

# ========== 采样步长 ==========
sample_steps = np.array([0.1, 0.1, 0.1, 0.1, 0.005, 5, 100, 1])  # 按需修改

def align_to_steps(x, param_min, param_max, steps):
    x_aligned = np.copy(x)
    for i in range(len(steps)):
        grid = np.arange(param_min[i], param_max[i]+steps[i]/2, steps[i])
        idx = np.abs(grid - x[i]).argmin()
        x_aligned[i] = grid[idx]
    return x_aligned

def align_points_to_steps(X_original, param_min, param_max, steps):
    X_aligned = np.zeros_like(X_original)
    for i, x in enumerate(X_original):
        X_aligned[i] = align_to_steps(x, param_min, param_max, steps)
    return X_aligned

# 4. 构建多目标高斯过程模型
def initialize_model(X, Y):
    num_tasks = Y.shape[1]
    # Add input and output transforms
    input_transform = InputStandardize(d=X.shape[1])
    outcome_transform = Standardize(m=num_tasks)
    model = KroneckerMultiTaskGP(
        X,
        Y,
        num_tasks=num_tasks,
        input_transform=input_transform,
        outcome_transform=outcome_transform
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

mll, model = initialize_model(X_tensor, Y_tensor)
fit_gpytorch_model(mll)

# 5. 计算参考点(5%分位数-0.1，标准化空间)
def get_ref_point(Y):
    return torch.quantile(Y, 0.05, dim=0) - 0.1

ref_point = get_ref_point(Y_tensor)
print(f"初始参考点 (标准化空间): {ref_point}")

# 6. 获取EHVI采集函数
def get_ehvi(model, Y_tensor, ref_point):
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=Y_tensor)
    ehvi = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point.tolist(),
        partitioning=partitioning
    )
    return ehvi

ehvi = get_ehvi(model, Y_tensor, ref_point)

# 7. 优化采集函数获取下一个实验点
def optimize_ehvi_and_get_next_point(ehvi, bounds, batch_size=1):
    candidates, _ = optimize_acqf(
        acq_function=ehvi,
        bounds=bounds,
        q=batch_size,  # 推荐点数量
        num_restarts=10,
        raw_samples=512,
    )
    return candidates

# ================== 记录所有建议的实验条件 ==================
suggested_points_all = []

# 首次建议点
next_X_scaled = optimize_ehvi_and_get_next_point(ehvi, bounds)
next_X_original = x_scaler.inverse_transform(next_X_scaled.detach().cpu().numpy())
next_X_original_aligned = align_points_to_steps(next_X_original, param_min, param_max, sample_steps)
print(f"建议的下一个实验条件（对齐步长后，原始量纲）: {next_X_original_aligned}")

suggested_points_all.append(next_X_original_aligned[0])  # 若每次只采样1点

# 8. 多次迭代优化
n_iterations = 5  # 可根据需求调整

for i in range(n_iterations):
    print(f"\n=== 迭代 {i + 1} ===")
    # 加载新实验数据
    new_data = pd.read_excel(r'D:\\7华工\\第一个工作\\all_suggested_points-20250613-第三轮条件+器件性能.xlsx')
    new_X = new_data.iloc[:, :8].values
    new_Y = new_data.iloc[:, 8:11].values

    X = np.vstack([X, new_X])
    Y = np.vstack([Y, new_Y])

    # 只用transform，不再fit
    X_scaled = x_scaler.transform(X)
    Y_scaled = y_scaler.transform(Y)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32)

    mll, model = initialize_model(X_tensor, Y_tensor)
    fit_gpytorch_model(mll)

    # 动态更新参考点
    ref_point = get_ref_point(Y_tensor)
    ehvi = get_ehvi(model, Y_tensor, ref_point)

    # 优化采集函数
    next_X_scaled = optimize_ehvi_and_get_next_point(ehvi, bounds)
    next_X_original = x_scaler.inverse_transform(next_X_scaled.detach().cpu().numpy())
    next_X_original_aligned = align_points_to_steps(next_X_original, param_min, param_max, sample_steps)
    print(f"建议的下一个实验条件（对齐步长后，原始量纲）: {next_X_original_aligned}")

    suggested_points_all.append(next_X_original_aligned[0])  # 若每次只采样1点

# ============== 最后一次性保存所有建议点 ==============
suggested_points_all = np.array(suggested_points_all)
columns = [f'param_{i+1}' for i in range(X.shape[1])]
df_suggest = pd.DataFrame(suggested_points_all, columns=columns)
df_suggest.to_excel('第四轮_all_suggested_points.xlsx', index=False)
print("所有建议点已保存到 第四轮_all_suggested_points.xlsx")