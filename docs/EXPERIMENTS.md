# 实验详解

本文档说明各实验的设置、对应代码、结果数据与插图，便于复现与理解。

## 1. 仿真参数

| 参数 | 数值 | 参数 | 数值 |
|------|------|------|------|
| 轨道面数 P | 6 | 学习率 | 5e-4 |
| 每面卫星数 S | 11 | 折扣因子 γ | 0.95 |
| 卫星总数 N | 66 | 探索率 ε | 1.0 → 0.05 |
| 轨道高度 h | 550 km | 经验池容量 | 5e4 |
| 轨道倾角 | 53° | 批大小 | 128 |
| 链路容量 | 50 Mbps | 最大跳数 | 25 |
| 队列容量 K | 50 | 训练回合数 | 5000 |

参数定义位于 `src/network.py`（业务与链路）与 `src/train.py`（训练超参数）。

## 2. 拥塞与链路模型（`network.py`）

- 链路利用率 `u = ρ / C`。
- 链路时延 = 传播时延 + M/M/1 排队时延，利用率越高时延越大。
- 链路丢包采用 M/M/1/K 队列溢出模型。
- 多条流依次注入、累加链路负载，实现拥塞传导。

## 3. POMDP 与 CA-MQR（`network.py` 中 `RoutingMDP`）

- **状态**：到目的的网格方向与跳距 + 业务类型独热 + 4 个邻居链路的（利用率、时延、丢包率、可用标志、下游拥塞势）+ 已走跳数。
- **动作**：从 4 个 +Grid 邻居中选下一跳（含动作掩码，屏蔽失效链路与已访问节点）。
- **奖励**：方向引导 + 每跳惩罚 + 业务差异化 QoS 惩罚 + 到达奖励；A 类重罚时延、B 类重罚利用率、C 类重罚丢包。
- **下游拥塞势场**：`SatNetwork.congestion_potential(dst)`，通过邻居间迭代松弛计算，作为状态特征提供拥塞纵深感知。

## 4. 求解算法（`train.py`）

- 网络：`QNet`（支持 Dueling 结构，将 Q 分解为状态价值 V 与动作优势 A）。
- 训练：`DQN` 类，采用 Double DQN 目标 + Huber 损失 + 经验回放 + 目标网络。
- 主函数 `train_dqn(use_congestion, use_potential, dueling, bg_hotspot, ...)` 通过开关控制不同变体，用于主实验与消融。

## 5. 各实验与对应产物

| 实验 | 代码位置 | 结果字段（results.json） | 插图 |
|------|---------|------------------------|------|
| 训练收敛 | `train_dqn` | `curve_full` / `curve_van` | fig_convergence.png |
| 负载扫描（时延/利用率/QoS） | `evaluate_scan` | `scan` | fig_delay / fig_util / fig_satrate |
| 分业务 QoS | `evaluate_perclass` | `perclass` | fig_perclass.png |
| Dueling 消融（多种子） | `conv_stats` | `abl_dueling` / `abl_plain` | fig_dueling.png |
| 下游势场增强（热点场景） | `eval_hotspot` | `potential` | fig_potential.png |
| 星座与场景示意 | `plot.py` | — | fig_constellation / fig_scenario |

## 6. 对比方法实现（`train.py`）

- `route_random`：随机可行下一跳。
- `route_dijkstra`：基于静态传播时延的最短路。
- `route_ecmp`：对最短路权重加随机扰动以分散负载。
- `route_dqn`：调用训练好的策略网络逐跳决策（CA-MQR 及其消融版本均通过此函数，由 `use_congestion` / `use_potential` 开关区分）。

## 7. 复现说明

- 结果具有一定随机性（训练随机种子、流量采样），复现的具体数值可能与论文有小幅波动，但整体规律（`Random < Dijkstra < ECMP < Vanilla-DQN < CA-MQR` 的性能梯度、重载下 CA-MQR 显著领先）稳定成立。
- 若只需查看结果，直接运行 `python plot.py` 即可（使用仓库自带的 `results/results.json`）。
