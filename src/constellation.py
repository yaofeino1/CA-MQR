"""
Walker Delta 星座建模 + 四维链路权重(含链路剩余寿命 tau_life)
纯 numpy + networkx 实现, 无需 STK / NS-3 / Hypatia。
"""
import numpy as np
import networkx as nx

# ---------------- 物理常数 ----------------
C = 3.0e8              # 光速 m/s
R_E = 6371e3           # 地球半径 m
MU = 3.986e14         # 地球引力常数 m^3/s^2


class Constellation:
    def __init__(self, P=6, S=11, h=550e3, inc_deg=53.0, F=1, n_gw=5, seed=42):
        self.P = P                      # 轨道面数
        self.S = S                      # 每面卫星数
        self.N = P * S                  # 卫星总数
        self.F = F                      # Walker Delta 相位因子(0<=F<=P-1)
        self.h = h                      # 轨道高度
        self.r = R_E + h                # 轨道半径
        self.inc = np.deg2rad(inc_deg)  # 倾角
        self.n_gw = n_gw
        self.T_orbit = 2 * np.pi * np.sqrt(self.r ** 3 / MU)  # 轨道周期 s
        self.omega = 2 * np.pi / self.T_orbit                  # 角速度
        self.rng = np.random.default_rng(seed)
        # 信关站固定经纬度(均匀分布)
        self.gw_lat = np.deg2rad(self.rng.uniform(-50, 50, n_gw))
        self.gw_lon = np.deg2rad(self.rng.uniform(-180, 180, n_gw))

    # 卫星在时刻 t 的 ECI 三维坐标
    def sat_positions(self, t):
        pos = np.zeros((self.N, 3))
        for p in range(self.P):
            Omega = 2 * np.pi * p / self.P          # 升交点赤经(均匀分布于360度)
            for s in range(self.S):
                idx = p * self.S + s
                # 标准 Walker Delta 相位: 面内均匀 + 相邻面相位偏移 F*2pi/N
                phase = (self.omega * t + 2 * np.pi * s / self.S
                         + 2 * np.pi * self.F * p / self.N)
                # 轨道面内坐标
                x0 = self.r * np.cos(phase)
                y0 = self.r * np.sin(phase)
                z0 = 0.0
                # 绕 x 轴转倾角
                y1 = y0 * np.cos(self.inc)
                z1 = y0 * np.sin(self.inc)
                # 绕 z 轴转升交点赤经
                x = x0 * np.cos(Omega) - y1 * np.sin(Omega)
                y = x0 * np.sin(Omega) + y1 * np.cos(Omega)
                z = z1
                pos[idx] = [x, y, z]
        return pos

    # +Grid 拓扑: 每颗星连 前/后(同面) 左/右(邻面)
    def grid_neighbors(self, idx):
        p, s = idx // self.S, idx % self.S
        front = p * self.S + (s + 1) % self.S
        back = p * self.S + (s - 1) % self.S
        left = ((p - 1) % self.P) * self.S + s
        right = ((p + 1) % self.P) * self.S + s
        return {'front': front, 'back': back, 'left': left, 'right': right}

    # 计算链路寿命: 用解析几何估计跨面链路距离达到阈值的时刻(快速近似)
    def link_lifetime(self, i, j, t, pos=None):
        p_i, p_j = i // self.S, j // self.S
        if p_i == p_j:
            return 1e4  # 同轨道面链路近似恒定, 设大常数
        # 跨面链路: 由当前相位到"反向缝/极区"的相位差推算剩余可用时间
        s_i = i % self.S
        phase_i = (self.omega * t + 2 * np.pi * s_i / self.S
                   + 2 * np.pi * self.F * p_i / self.N) % (2 * np.pi)
        # 卫星运行到纬度幅角接近 +-90 deg(极区)时跨面链路拉长/断开
        # 纬度幅角 = phase, 距最近极点(pi/2 或 3pi/2)的相位差
        to_pole = min(abs(phase_i - np.pi / 2), abs(phase_i - 3 * np.pi / 2))
        to_pole = min(to_pole, abs(phase_i + np.pi / 2),
                      abs(phase_i - 5 * np.pi / 2))
        tau = to_pole / self.omega
        return float(np.clip(tau, 5.0, 600.0))

    # 构建时刻 t 的加权图快照
    def build_snapshot(self, t, load_factor=0.4, fault_ratio=0.0):
        pos = self.sat_positions(t)
        G = nx.DiGraph()
        for i in range(self.N):
            G.add_node(i)
        B_total = 1e9        # ISL 带宽 1 Gbps
        for i in range(self.N):
            nb = self.grid_neighbors(i)
            for direction, j in nb.items():
                d_geo = np.linalg.norm(pos[i] - pos[j])
                delay = d_geo / C + 1e-3          # 传播 + 处理时延
                # 链路利用率(随机负载, 均值=load_factor)
                rho = np.clip(self.rng.normal(load_factor, 0.15), 0.01, 0.99)
                b_avail = B_total * (1 - rho)
                # M/M/1/K 丢包
                K = 100
                if abs(rho - 1) < 1e-6:
                    plr = 1.0 / (K + 1)
                else:
                    plr = (1 - rho) * rho ** K / (1 - rho ** (K + 1))
                plr = max(plr, 1e-7)
                tau = self.link_lifetime(i, j, t)
                # 随机故障: 提前使部分链路失效
                alive = True
                if fault_ratio > 0 and self.rng.random() < fault_ratio:
                    alive = False
                G.add_edge(i, j, delay=delay, band=b_avail, plr=plr,
                           tau=tau, rho=rho, direction=direction, alive=alive)
        return G, pos


if __name__ == '__main__':
    c = Constellation()
    print(f"卫星数={c.N}, 轨道周期={c.T_orbit:.1f}s")
    G, pos = c.build_snapshot(0.0)
    print(f"边数={G.number_of_edges()}")
    taus = [d['tau'] for _, _, d in G.edges(data=True)]
    print(f"tau_life: min={min(taus):.1f}, max={max(taus):.1f}")
