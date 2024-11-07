import numpy as np

# 参数定义
N = 300  # 时间步长（假设WLTC工况在300个时间步完成）
SOC_min, SOC_max = 0.2, 0.8  # 电池SOC的最小和最大值
P_engine_max, P_engine_min = 70, 0  # 发动机功率范围（kW）
P_battery_max, P_battery_min = 50, -50  # 电池功率范围（kW）
efficiency_engine = 0.35  # 发动机效率
efficiency_battery_charge = 0.85  # 电池充电效率
efficiency_battery_discharge = 0.9  # 电池放电效率

# 模拟WLTC工况的功率需求序列（假设这里有真实的WLTC数据）
P_demand = np.random.uniform(20, 60, N)  # 随机生成功率需求，实际应使用WLTC数据

# 电池SOC更新函数
def update_SOC(SOC, P_battery):
    if P_battery > 0:  # 放电
        SOC -= P_battery / 1000 / efficiency_battery_discharge # 变量 = 变量 - 等式右边值
    else:  # 充电
        SOC -= P_battery * efficiency_battery_charge / 1000
    return SOC

# 燃油消耗模型
def fuel_consumption(P_engine):
    return P_engine / efficiency_engine

# 成本函数，计算燃油消耗成本和电池功率使用成本
def cost_function(P_engine, P_battery):
    return fuel_consumption(P_engine) + abs(P_battery) * 0.01  # 0.01是电池使用权重系数

# 初始化DP表
SOC_grid = np.linspace(SOC_min, SOC_max, 101)  # SOC离散化
cost = np.full((N, len(SOC_grid)), np.inf)  # 成本表
cost[-1, :] = 0  # 最后一个时间步的成本为0
policy = np.zeros((N - 1, len(SOC_grid)))  # 存储最优策略

# 反向动态规划
for t in range(N - 2, -1, -1):
    for i, SOC in enumerate(SOC_grid):
        for P_engine in np.linspace(P_engine_min, P_engine_max, 11):
            P_battery = P_demand[t] - P_engine
            if P_battery < P_battery_min or P_battery > P_battery_max:
                continue
            new_SOC = update_SOC(SOC, P_battery)
            if new_SOC < SOC_min or new_SOC > SOC_max:
                continue
            j = np.argmin(abs(SOC_grid - new_SOC))
            temp_cost = cost_function(P_engine, P_battery) + cost[t + 1, j]
            if temp_cost < cost[t, i]:
                cost[t, i] = temp_cost
                policy[t, i] = P_engine  # 记录当前时间步的最优发动机功率

# 输出最优策略
optimal_policy = []
SOC = (SOC_max + SOC_min) / 2  # 初始SOC
for t in range(N - 1):
    i = np.argmin(abs(SOC_grid - SOC))
    P_engine = policy[t, i]
    P_battery = P_demand[t] - P_engine
    optimal_policy.append((P_engine, P_battery))
    SOC = update_SOC(SOC, P_battery)

print("最优策略（部分输出）：")
for t, (P_engine, P_battery) in enumerate(optimal_policy[:]):
    print(f"时间步 {t + 1}: 发动机功率 = {P_engine:.2f} kW, 电池功率 = {P_battery:.2f} kW")
