import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1. 算法核心函数定义
# ==========================================
def compute_mse(X, y, theta):
    """计算均方误差 (MSE)"""
    y_pred = X.dot(theta)
    return np.mean((y - y_pred) ** 2)


def least_squares(X, y):
    """最小二乘法求解析解"""
    # theta = (X^T * X)^-1 * X^T * y
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta


def gradient_descent(X, y, lr=0.01, epochs=1000):
    """梯度下降法"""
    N, d = X.shape
    theta = np.zeros((d, 1))  # 初始化参数
    loss_history = []

    for i in range(epochs):
        y_pred = X.dot(theta)
        gradient = (1 / N) * X.T.dot(y_pred - y)
        theta = theta - lr * gradient
        loss_history.append(compute_mse(X, y, theta))

    return theta, loss_history


def newtons_method(X, y, epochs=5):
    """牛顿法"""
    N, d = X.shape
    theta = np.zeros((d, 1))  # 初始化
    loss_history = []

    for i in range(epochs):
        y_pred = X.dot(theta)
        gradient = (1 / N) * X.T.dot(y_pred - y)
        hessian = (1 / N) * X.T.dot(X)

        # theta = theta - H^-1 * gradient
        theta = theta - np.linalg.inv(hessian).dot(gradient)
        loss_history.append(compute_mse(X, y, theta))

    return theta, loss_history


def create_polynomial_features(X, degree):
    """生成多项式特征矩阵 [1, x, x^2, ..., x^degree]"""
    X_poly = np.ones((X.shape[0], 1))
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, X ** d))
    return X_poly


# ==========================================
# 2. 真实数据加载与预处理
# ==========================================
file_name = 'Data4Regression.xlsx'
print(f"正在读取当前目录下的文件: {file_name} ...")

# 修复：设置 header=0 跳过表头，并强制转换为 float 浮点数，防止字符串引起乘法报错
train_data = pd.read_excel(file_name, sheet_name=0, header=0).values.astype(float)
test_data = pd.read_excel(file_name, sheet_name=1, header=0).values.astype(float)

# 提取 X 和 Y 并重塑为 (N, 1) 的列向量
x_train, y_train = train_data[:, 0].reshape(-1, 1), train_data[:, 1].reshape(-1, 1)
x_test, y_test = test_data[:, 0].reshape(-1, 1), test_data[:, 1].reshape(-1, 1)

print(f"数据加载成功！训练集大小: {x_train.shape[0]}, 测试集大小: {x_test.shape[0]}")

# ==========================================
# 3. 实验一：线性拟合对比
# ==========================================
print("\n--- 实验一：线性拟合 ---")
# 构建增广矩阵 (加上一列1作为偏置项 x0)
X_train_lin = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
X_test_lin = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

# --- 最小二乘法 ---
theta_ls = least_squares(X_train_lin, y_train)
mse_train_ls = compute_mse(X_train_lin, y_train, theta_ls)
mse_test_ls = compute_mse(X_test_lin, y_test, theta_ls)
print(f"1. 最小二乘法 - 训练误差: {mse_train_ls:.4f}, 测试误差: {mse_test_ls:.4f}")

# --- 梯度下降法 ---
# 提示：如果输出是 NaN（梯度爆炸），请把这里的 lr 改小，比如 lr=0.001
theta_gd, loss_gd = gradient_descent(X_train_lin, y_train, lr=0.01, epochs=1000)
mse_train_gd = compute_mse(X_train_lin, y_train, theta_gd)
mse_test_gd = compute_mse(X_test_lin, y_test, theta_gd)
print(f"2. 梯度下降法 - 训练误差: {mse_train_gd:.4f}, 测试误差: {mse_test_gd:.4f}")

# --- 牛顿法 ---
theta_nt, loss_nt = newtons_method(X_train_lin, y_train, epochs=2)
mse_train_nt = compute_mse(X_train_lin, y_train, theta_nt)
mse_test_nt = compute_mse(X_test_lin, y_test, theta_nt)
print(f"3. 牛顿法     - 训练误差: {mse_train_nt:.4f}, 测试误差: {mse_test_nt:.4f}")

# ==========================================
# 4. 实验二：非线性拟合 (多项式回归)
# ==========================================
print("\n--- 实验二：非线性拟合 (多项式回归) ---")
degree = 3  # 你可以在报告里修改这个阶数(比如尝试 2, 4, 9)，观察不同阶数下的效果
X_train_poly = create_polynomial_features(x_train, degree)
X_test_poly = create_polynomial_features(x_test, degree)

theta_poly = least_squares(X_train_poly, y_train)
mse_train_poly = compute_mse(X_train_poly, y_train, theta_poly)
mse_test_poly = compute_mse(X_test_poly, y_test, theta_poly)
print(f"多项式 (M={degree})  - 训练误差: {mse_train_poly:.4f}, 测试误差: {mse_test_poly:.4f}")

# ==========================================
# 5. 保存结果到 Excel (写报告用)
# ==========================================
results_df = pd.DataFrame({
    '算法模型': ['最小二乘法', '梯度下降法', '牛顿法', f'多项式回归(M={degree})'],
    '训练误差(MSE)': [mse_train_ls, mse_train_gd, mse_train_nt, mse_train_poly],
    '测试误差(MSE)': [mse_test_ls, mse_test_gd, mse_test_nt, mse_test_poly]
})
results_df.to_excel('regression_results.xlsx', index=False)
print("\n✅ 误差数据已成功保存为 regression_results.xlsx！")

# ==========================================
# 6. 可视化绘图与保存
# ==========================================
plt.figure(figsize=(12, 5))

# 图 1：线性拟合
plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color='gray', label='Train Data', alpha=0.6)
plt.scatter(x_test, y_test, color='red', label='Test Data', marker='x')

x_plot = np.linspace(np.min(x_train) - 1, np.max(x_train) + 1, 100).reshape(-1, 1)
X_plot_lin = np.hstack((np.ones((x_plot.shape[0], 1)), x_plot))

plt.plot(x_plot, X_plot_lin.dot(theta_ls), 'b-', label='Least Squares')
plt.plot(x_plot, X_plot_lin.dot(theta_gd), 'g--', label='Gradient Descent')
plt.title('Part 1: Linear Fitting')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

# 图 2：多项式拟合
plt.subplot(1, 2, 2)
plt.scatter(x_train, y_train, color='gray', label='Train Data', alpha=0.6)
plt.scatter(x_test, y_test, color='red', label='Test Data', marker='x')

X_plot_poly = create_polynomial_features(x_plot, degree)
plt.plot(x_plot, X_plot_poly.dot(theta_poly), 'm-', label=f'Polynomial (M={degree})')
plt.title(f'Part 2: Non-Linear Fitting (Polynomial M={degree})')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()

# 保存高清图片到当前目录 (dpi=300)
plt.savefig('fitting_results.png', dpi=300, bbox_inches='tight')
print("✅ 拟合图片已成功保存为 fitting_results.png！")

# 弹出窗口展示
plt.show()