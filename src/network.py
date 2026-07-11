"""
带真实拥塞传导的 LEO 卫星网络流级路由仿真环境。
核心机制: 多条业务流依次注入, 每条流占用其路径上各链路的带宽/队列,
后续流看到前面流造成的拥塞 -> 最短路会把流量堆到热点造成排队爆炸,
拥塞感知的 RL 通过绕开热点取胜。

三类业务:
  A 时延敏感 (小包高频, 重罚时延)
  B 带宽密集 (大流, 需要高带宽)
  C 可靠性优先 (重罚丢包)
"""
import numpy as np
import networkx as nx
from constellation import Constellation

CLASSES = ['A', 'B', 'C']
DIRS = ['front', 'back', 'left', 'right']

# 业务 QoS 需求与流量特征 (速率相对链路容量 50Mbps 标定, 使高负载真正拥塞)
BIZ = {
    'A': {'rate': 2e6,  'D_max': 0.12,  'B_min': 1e6,   'P_max': 1e-2, 'prob': 0.3},
    'B': {'rate': 8e6,  'D_max': 0.30,  'B_min': 4e6,   'P_max': 5e-2, 'prob': 0.5},
    'C': {'rate': 1.5e6,'D_max': 0.30,  'B_min': 0.5e6, 'P_max': 1e-4, 'prob': 0.2},
}
LINK_CAP = 50e6   # ISL 容量 50 Mbps (标定使多流竞争产生拥塞)
H_MAX = 25


class SatNetwork:
    """管理一个时刻快照下的网络状态与拥塞。"""

    def __init__(self, constellation, load_factor=1.0, fault_ratio=0.0, seed=0):
        self.c = constellation
        self.N = constellation.N
        self.rng = np.random.default_rng(seed)
        self.load_factor = load_factor
        self.fault_ratio = fault_ratio
        self.reset_snapshot()

    def reset_snapshot(self):
        t = self.rng.uniform(0, self.c.T_orbit)
        self.pos = self.c.sat_positions(t)
        self.t = t
        # 邻接与链路静态属性
        self.cap = {}       # 链路容量 bps
        self.delay0 = {}    # 传播时延 s
        self.tau = {}       # 链路寿命 s
        self.alive = {}     # 是否可用
        self.load = {}      # 当前已占用带宽 bps (拥塞状态)
        for i in range(self.N):
            nb = self.c.grid_neighbors(i)
            for d, j in nb.items():
                dist = np.linalg.norm(self.pos[i] - self.pos[j])
                self.cap[(i, j)] = LINK_CAP     # ISL 容量
                self.delay0[(i, j)] = dist / 3e8 + 1e-3
                self.tau[(i, j)] = self.c.link_lifetime(i, j, t)
                alive = not (self.fault_ratio > 0 and self.rng.random() < self.fault_ratio)
                self.alive[(i, j)] = alive
                self.load[(i, j)] = 0.0

    def clear_load(self):
        for k in self.load:
            self.load[k] = 0.0

    def neighbors(self, i):
        return [self.c.grid_neighbors(i)[d] for d in DIRS]

    def link_util(self, i, j):
        return self.load.get((i, j), 0.0) / self.cap.get((i, j), 1e9)

    def link_delay(self, i, j):
        """含排队的链路时延: 传播 + M/M/1 排队(利用率越高时延越大)。"""
        u = min(self.link_util(i, j), 0.99)
        # M/M/1 平均时延放大因子 1/(1-u)
        queue = self.delay0[(i, j)] * (u / (1 - u)) if u < 0.99 else self.delay0[(i, j)] * 99
        return self.delay0[(i, j)] + queue

    def link_plr(self, i, j):
        """利用率越高, 队列溢出丢包越大。"""
        u = min(self.link_util(i, j), 0.999)
        K = 50
        rho = u
        if rho < 1e-6:
            return 1e-6
        if abs(rho - 1) < 1e-6:
            return 1.0 / (K + 1)
        return max((1 - rho) * rho ** K / (1 - rho ** (K + 1)), 1e-6)

    def add_flow_load(self, path, rate):
        for u, v in zip(path[:-1], path[1:]):
            if (u, v) in self.load:
                self.load[(u, v)] += rate

    def congestion_potential(self, dst, gamma=0.6, iters=8):
        """计算所有节点朝目的 dst 的下游拥塞势场 Psi。
        Psi[i] 编码从 i 沿"靠近目的且低拥塞"方向前行时, 下游 K 跳内累积的
        拥塞水平(经折扣)。通过邻居间迭代松弛得到, 仅依赖局部信息交换:
            Psi[i] = min_{j: grid_dist(j,dst)<grid_dist(i,dst)}
                        ( u_ij + gamma * Psi[j] )
        物理含义: 势越高, 说明沿该节点通往目的的道路越拥塞。
        该值作为状态特征, 使智能体获得超越 1 跳的拥塞纵深感知。"""
        S = self.c.S
        gd = {i: self._grid_dist_ij(i, dst) for i in range(self.N)}
        Psi = {i: 0.0 for i in range(self.N)}
        for _ in range(iters):
            newPsi = dict(Psi)
            for i in range(self.N):
                if i == dst:
                    newPsi[i] = 0.0
                    continue
                best = None
                for j in self.neighbors(i):
                    key = (i, j)
                    if key not in self.cap or not self.alive[key]:
                        continue
                    if gd[j] >= gd[i]:      # 仅沿靠近目的的方向传播
                        continue
                    val = self.link_util(*key) + gamma * Psi[j]
                    if best is None or val < best:
                        best = val
                if best is not None:
                    newPsi[i] = best
            Psi = newPsi
        return Psi

    def _grid_dist_ij(self, a, b):
        S = self.c.S
        pa, sa = a // S, a % S
        pb, sb = b // S, b % S
        P = self.c.P
        dp = min(abs(pa - pb), P - abs(pa - pb))
        ds = min(abs(sa - sb), S - abs(sa - sb))
        return dp + ds

    def path_metrics(self, path, rate):
        """给定路径与流速率, 返回端到端 QoS(基于当前拥塞状态)。"""
        if path is None or len(path) < 2:
            return None
        d, b, surv = 0.0, 1e12, 1.0
        for u, v in zip(path[:-1], path[1:]):
            if (u, v) not in self.cap or not self.alive[(u, v)]:
                return None
            d += self.link_delay(u, v)
            b = min(b, self.cap[(u, v)] - self.load[(u, v)])  # 剩余带宽
            surv *= (1 - self.link_plr(u, v))
        return {'delay': d, 'band': max(b, 0), 'plr': 1 - surv, 'hop': len(path) - 1}


class RoutingMDP:
    """逐跳路由 MDP 封装(供 DQN 交互)。一次 episode = 路由一条流。"""

    def __init__(self, net, use_congestion=True, use_potential=False):
        self.net = net
        self.use_congestion = use_congestion
        self.use_potential = use_potential
        # 状态: 方向3 + 业务3 + 4邻居*(util,delay,plr,tau,valid,psi)=24 + 跳数1 = 31
        nf = 6 if use_potential else 5
        self.state_dim = 3 + 3 + 4 * nf + 1
        self.action_dim = 4

    def start(self, src, dst, cls):
        self.src, self.dst, self.cls = src, dst, cls
        self.cur = src
        self.hop = 0
        self.visited = {src}
        self.path = [src]
        # 计算朝目的的下游拥塞势场(创新点)
        if self.use_potential:
            self.psi = self.net.congestion_potential(dst)
            self._psi_norm = max(max(self.psi.values()), 1e-6)
        return self._obs()

    def _obs(self):
        # 网格方向: 到目的的 (轨道面差, 面内相位差, 总跳距) 归一化
        P, S = self.net.c.P, self.net.c.S
        pc, sc = self.cur // S, self.cur % S
        pd_, sd_ = self.dst // S, self.dst % S
        dp = ((pd_ - pc + P // 2) % P - P // 2) / (P / 2)
        ds = ((sd_ - sc + S // 2) % S - S // 2) / (S / 2)
        gd = self._grid_dist(self.cur, self.dst) / (P + S)
        dpos = [dp, ds, gd]
        cls_oh = [1.0 if self.cls == k else 0.0 for k in CLASSES]
        feats = []
        for j in self.net.neighbors(self.cur):
            key = (self.cur, j)
            if key in self.net.cap and self.net.alive[key]:
                util = self.net.link_util(*key) if self.use_congestion else 0.0
                dl = self.net.link_delay(*key) / 0.05
                plr = -np.log10(self.net.link_plr(*key)) / 6.0
                tau = min(self.net.tau[key], 300) / 300
                valid = 1.0
                psi = (self.psi.get(j, 0.0) / self._psi_norm) if self.use_potential else 0.0
            else:
                util, dl, plr, tau, valid, psi = 1.0, 1.0, 0.0, 0.0, 0.0, 1.0
            if self.use_potential:
                feats += [util, dl, plr, tau, valid, psi]
            else:
                feats += [util, dl, plr, tau, valid]
        return np.array(list(dpos) + cls_oh + feats + [self.hop / H_MAX], dtype=np.float32)

    def action_mask(self):
        mask = np.zeros(4, bool)
        for a, j in enumerate(self.net.neighbors(self.cur)):
            key = (self.cur, j)
            if key in self.net.cap and self.net.alive[key] and j not in self.visited:
                mask[a] = True
        if not mask.any():
            for a, j in enumerate(self.net.neighbors(self.cur)):
                key = (self.cur, j)
                if key in self.net.cap and self.net.alive[key]:
                    mask[a] = True
        return mask

    def _grid_dist(self, a, b):
        """+Grid 拓扑上的曼哈顿跳数距离(环形)。"""
        pa, sa = a // self.net.c.S, a % self.net.c.S
        pb, sb = b // self.net.c.S, b % self.net.c.S
        P, S = self.net.c.P, self.net.c.S
        dp = min(abs(pa - pb), P - abs(pa - pb))
        ds = min(abs(sa - sb), S - abs(sa - sb))
        return dp + ds

    def step(self, action):
        j = self.net.neighbors(self.cur)[action]
        key = (self.cur, j)
        if key not in self.net.cap or not self.net.alive[key]:
            return self._obs(), -10.0, True, {'fail': True}
        # 单跳代价
        dl = self.net.link_delay(*key)
        util = self.net.link_util(*key)
        plr = self.net.link_plr(*key)
        # 网格跳数势能引导(比 ECI 欧氏距离更贴合 +Grid 拓扑)
        d_before = self._grid_dist(self.cur, self.dst)
        d_after = self._grid_dist(j, self.dst)
        rpd = float(d_before - d_after)   # +1 靠近, -1 远离

        self.cur = j
        self.visited.add(j)
        self.path.append(j)
        self.hop += 1

        w = BIZ[self.cls]
        # 奖励设计: 方向引导为主(保证到达), 拥塞/QoS 为辅(业务差异化)
        r = 6.0 * rpd - 0.2                       # 势能塑形方向引导 + 每跳惩罚
        if self.use_congestion:
            # 拥塞感知 + 业务差异化 QoS 惩罚(本文方法)
            if self.cls == 'A':                    # 时延敏感: 罚排队时延与拥塞
                r -= 2.5 * util
                r -= 5.0 * max(0, dl - self.net.delay0[key]) / 0.02
            elif self.cls == 'B':                  # 带宽密集: 罚高利用率(带宽被占)
                r -= 3.5 * util
            else:                                   # 可靠性: 罚丢包
                r -= 2.0 * util
                if plr > 1e-4:
                    r -= 2.5
            # 下游拥塞势惩罚(创新点): 规避通往高拥塞纵深的方向
            if self.use_potential:
                r -= 3.0 * (self.psi.get(j, 0.0) / self._psi_norm)
        else:
            # 拥塞无感知基线: 仅按静态传播时延(等价学习最短路)
            r -= 8.0 * self.net.delay0[key]

        done = False
        info = {'fail': False}
        if self.cur == self.dst:
            r += 15.0
            done = True
            info['delivered'] = True
        elif self.hop >= H_MAX:
            r -= 8.0
            done = True
            info['timeout'] = True
        return self._obs(), r, done, info
