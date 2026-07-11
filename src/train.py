"""
PyTorch Double DQN + 基线(随机/Dijkstra/ECMP) + 负载扫描评估。
核心: 批量注入多条流, 拥塞传导, 对比各方法在不同负载下的性能。
"""
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import json
import time
from collections import deque
from constellation import Constellation
from network import SatNetwork, RoutingMDP, BIZ, CLASSES, DIRS, H_MAX

DEV = torch.device('cpu')  # 小模型小batch, CPU 比 MPS 更快(避免搬运开销)
torch.manual_seed(0)


class QNet(nn.Module):
    """支持普通与 Dueling 两种结构的 Q 网络。
    Dueling: Q(s,a)=V(s)+(A(s,a)-mean_a A(s,a)), 将状态价值与动作优势解耦,
    在多个动作价值相近(如节点被拥塞包围)时能更稳定地估计状态价值。"""

    def __init__(self, sd, ad, dueling=False):
        super().__init__()
        self.dueling = dueling
        self.feature = nn.Sequential(
            nn.Linear(sd, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        if dueling:
            self.value = nn.Linear(128, 1)          # 状态价值流 V(s)
            self.adv = nn.Linear(128, ad)           # 动作优势流 A(s,a)
        else:
            self.head = nn.Linear(128, ad)

    def forward(self, x):
        h = self.feature(x)
        if self.dueling:
            v = self.value(h)
            a = self.adv(h)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.head(h)


class DQN:
    def __init__(self, sd, ad, lr=5e-4, gamma=0.95, dueling=False):
        self.q = QNet(sd, ad, dueling).to(DEV)
        self.tgt = QNet(sd, ad, dueling).to(DEV)
        self.tgt.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.buf = deque(maxlen=50000)
        self.ad = ad

    def act(self, obs, mask, eps):
        if np.random.rand() < eps:
            return int(np.random.choice(np.where(mask)[0]))
        with torch.no_grad():
            qv = self.q(torch.tensor(obs, dtype=torch.float32, device=DEV).unsqueeze(0))[0].cpu().numpy()
        qv[~mask] = -1e9
        return int(np.argmax(qv))

    def store(self, *tr):
        self.buf.append(tr)

    def train_step(self, batch=128):
        if len(self.buf) < batch:
            return
        idx = np.random.randint(0, len(self.buf), batch)
        S, A, R, S2, M2, D = zip(*[self.buf[i] for i in idx])
        S = torch.tensor(np.array(S), dtype=torch.float32, device=DEV)
        A = torch.tensor(np.array(A), dtype=torch.long, device=DEV)
        R = torch.tensor(np.array(R), dtype=torch.float32, device=DEV)
        S2 = torch.tensor(np.array(S2), dtype=torch.float32, device=DEV)
        M2 = torch.tensor(np.array(M2), dtype=torch.bool, device=DEV)
        D = torch.tensor(np.array(D), dtype=torch.float32, device=DEV)
        q = self.q(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q2 = self.q(S2)
            q2[~M2] = -1e9
            a_star = q2.argmax(1)
            q_tgt = self.tgt(S2).gather(1, a_star.unsqueeze(1)).squeeze(1)
            y = R + self.gamma * q_tgt * (1 - D)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.item())

    def sync(self):
        self.tgt.load_state_dict(self.q.state_dict())


def train_dqn(use_congestion=True, episodes=6000, load_factor=1.0, seed=1,
              log=True, dueling=None, use_potential=False, bg_hotspot=False):
    # 默认: 本文方法(拥塞感知)采用 Dueling 结构; 基线不采用
    if dueling is None:
        dueling = use_congestion
    con = Constellation(seed=42)
    rng = np.random.default_rng(seed)
    agent = None
    curve = []
    eps = 1.0
    for ep in range(episodes):
        # 每若干 episode 重建带随机背景负载的快照(制造拥塞环境)
        if ep % 1 == 0:
            net = SatNetwork(con, load_factor=load_factor, seed=int(rng.integers(1e6)))
            # 注入背景流量制造热点拥塞
            inject_background(net, load_factor, rng, hotspot=bg_hotspot)
        mdp = RoutingMDP(net, use_congestion=use_congestion,
                         use_potential=use_potential)
        if agent is None:
            agent = DQN(mdp.state_dim, mdp.action_dim, dueling=dueling)
        src = int(rng.integers(con.N)); dst = int(rng.integers(con.N))
        while dst == src:
            dst = int(rng.integers(con.N))
        cls = rng.choice(CLASSES, p=[BIZ[c]['prob'] for c in CLASSES])
        obs = mdp.start(src, dst, cls)
        done = False; ep_r = 0
        while not done:
            m = mdp.action_mask()
            a = agent.act(obs, m, eps)
            obs2, r, done, info = mdp.step(a)
            m2 = mdp.action_mask() if not done else np.ones(4, bool)
            agent.store(obs, a, r, obs2, m2, float(done))
            agent.train_step()
            obs = obs2; ep_r += r
        if ep % 20 == 0:
            agent.sync()
        eps = max(0.05, eps * 0.9995)
        curve.append(ep_r)
        if log and ep % 1000 == 0:
            print(f"  ep{ep} reward={np.mean(curve[-200:]):.2f} eps={eps:.2f}")
    return agent, curve


def inject_background(net, load_factor, rng, n_bg=None, hotspot=True):
    """注入背景流量制造拥塞环境。
    hotspot=True 时, 大部分背景流汇聚到少数热点节点(模拟地面信关站/人口
    密集区的流量汇聚), 形成空间聚集的成片拥塞区, 更贴近真实卫星网络,
    也使下游拥塞纵深成为影响路由质量的关键因素。"""
    con = net.c
    if n_bg is None:
        n_bg = int(rng.integers(120, 240))
    Gd = build_delay_graph(net)
    # 选取 3 个热点汇聚节点
    hotspots = [int(x) for x in rng.choice(con.N, size=3, replace=False)]
    for _ in range(n_bg):
        s = int(rng.integers(con.N))
        if hotspot and rng.random() < 0.7:
            d = hotspots[int(rng.integers(len(hotspots)))]
        else:
            d = int(rng.integers(con.N))
        if s == d:
            continue
        try:
            p = nx.shortest_path(Gd, s, d, weight='w')
            cls = rng.choice(CLASSES, p=[BIZ[c]['prob'] for c in CLASSES])
            net.add_flow_load(p, BIZ[cls]['rate'])
        except Exception:
            pass


def build_delay_graph(net):
    Gd = nx.DiGraph()
    for (u, v), c in net.cap.items():
        if net.alive[(u, v)]:
            Gd.add_edge(u, v, w=net.delay0[(u, v)])
    return Gd


# -------- 基线 --------
def route_dijkstra(net, src, dst, weight='delay'):
    Gd = nx.DiGraph()
    for (u, v) in net.cap:
        if not net.alive[(u, v)]:
            continue
        w = net.delay0[(u, v)] if weight == 'delay' else 1.0
        Gd.add_edge(u, v, w=w)
    try:
        return nx.shortest_path(Gd, src, dst, weight='w')
    except Exception:
        return None


def route_ecmp(net, src, dst, rng):
    """等价多路径: 在最短路长度附近随机选路以分散负载。"""
    Gd = nx.DiGraph()
    for (u, v) in net.cap:
        if not net.alive[(u, v)]:
            continue
        Gd.add_edge(u, v, w=net.delay0[(u, v)])
    try:
        # 随机扰动权重, 得到多样化近最短路
        for (u, v) in Gd.edges:
            Gd[u][v]['w'] *= (1 + rng.uniform(0, 0.3))
        return nx.shortest_path(Gd, src, dst, weight='w')
    except Exception:
        return None


def route_random(net, src, dst, rng):
    cur, path, vis = src, [src], {src}
    for _ in range(H_MAX):
        if cur == dst:
            return path
        cand = [j for j in net.neighbors(cur)
                if (cur, j) in net.cap and net.alive[(cur, j)] and j not in vis]
        if not cand:
            cand = [j for j in net.neighbors(cur)
                    if (cur, j) in net.cap and net.alive[(cur, j)]]
        if not cand:
            return None
        cur = int(rng.choice(cand)); path.append(cur); vis.add(cur)
    return None


def gen_flows(con, rng, n_flows, hotspot=True):
    """生成一批业务流, 部分汇聚到热点节点(与训练分布一致)。"""
    hotspots = [int(x) for x in rng.choice(con.N, size=3, replace=False)]
    flows = []
    for _ in range(n_flows):
        s = int(rng.integers(con.N))
        if hotspot and rng.random() < 0.7:
            d = hotspots[int(rng.integers(len(hotspots)))]
        else:
            d = int(rng.integers(con.N))
        while d == s:
            d = int(rng.integers(con.N))
        cls = rng.choice(CLASSES, p=[BIZ[c]['prob'] for c in CLASSES])
        flows.append((s, d, cls))
    return flows


def route_dqn(net, agent, src, dst, cls, use_congestion=True, use_potential=False):
    mdp = RoutingMDP(net, use_congestion=use_congestion, use_potential=use_potential)
    obs = mdp.start(src, dst, cls); done = False
    while not done:
        m = mdp.action_mask()
        a = agent.act(obs, m, 0.0)
        obs, r, done, info = mdp.step(a)
    return mdp.path if mdp.cur == dst else None


def qos_ok(cls, m):
    if m is None:
        return False
    w = BIZ[cls]
    return m['delay'] <= w['D_max'] and m['band'] >= w['B_min'] and m['plr'] <= w['P_max']


def evaluate_scan(agent_full, agent_van, loads, base_flows=160, n_trials=25, seed=123, hotspot=False, potential=False):
    """负载扫描: load 控制流数量(制造拥塞梯度), 拥塞传导, 统计各方法。"""
    con = Constellation(seed=42)
    methods = ['Random', 'Dijkstra', 'ECMP', 'Vanilla-DQN', 'Proposed']
    res = {m: {ld: {'delay': [], 'plr': [], 'deliv': [], 'sat': [],
                    'util': []} for ld in loads} for m in methods}
    for ld in loads:
        n_flows = int(base_flows * ld)
        for tr in range(n_trials):
            rng = np.random.default_rng(seed + tr * 100 + int(ld * 10))
            flows = gen_flows(con, rng, n_flows, hotspot=hotspot)
            for m in methods:
                net = SatNetwork(con, load_factor=1.0, seed=seed + tr)
                net.clear_load()
                deliv = 0; sat = 0; delays = []; plrs = []
                r2 = np.random.default_rng(seed + tr + 7)
                for (s, d, cls) in flows:
                    if m == 'Random':
                        p = route_random(net, s, d, r2)
                    elif m == 'Dijkstra':
                        p = route_dijkstra(net, s, d, 'delay')
                    elif m == 'ECMP':
                        p = route_ecmp(net, s, d, r2)
                    elif m == 'Vanilla-DQN':
                        p = route_dqn(net, agent_van, s, d, cls, use_congestion=False)
                    else:
                        p = route_dqn(net, agent_full, s, d, cls, use_congestion=True, use_potential=potential)
                    if p is not None and len(p) >= 2:
                        mq = net.path_metrics(p, BIZ[cls]['rate'])
                        if mq is not None:
                            deliv += 1
                            delays.append(mq['delay'] * 1000)
                            plrs.append(mq['plr'])
                            if qos_ok(cls, mq):
                                sat += 1
                            net.add_flow_load(p, BIZ[cls]['rate'])
                utils = [net.link_util(*k) for k in net.cap if net.alive[k]]
                res[m][ld]['deliv'].append(deliv / n_flows)
                res[m][ld]['sat'].append(sat / n_flows)
                res[m][ld]['delay'].append(np.mean(delays) if delays else 0)
                res[m][ld]['plr'].append(np.mean(plrs) if plrs else 0)
                res[m][ld]['util'].append(np.percentile(utils, 95) if utils else 0)
    return res


def summarize_scan(res, loads):
    out = {}
    for m in res:
        out[m] = {'loads': loads}
        for k in ['delay', 'plr', 'deliv', 'sat', 'util']:
            out[m][k] = [float(np.mean(res[m][ld][k])) for ld in loads]
    return out


def evaluate_perclass(agent_full, agent_van, n_flows=220, n_trials=30, seed=555, hotspot=False, potential=False):
    con = Constellation(seed=42)
    methods = ['Dijkstra', 'ECMP', 'Vanilla-DQN', 'Proposed']
    stat = {m: {c: [0, 0] for c in CLASSES} for m in methods}
    for tr in range(n_trials):
        rng = np.random.default_rng(seed + tr)
        flows = gen_flows(con, rng, n_flows, hotspot=hotspot)
        for m in methods:
            net = SatNetwork(con, load_factor=1.0, seed=seed + tr)
            net.clear_load()
            r2 = np.random.default_rng(seed + tr + 3)
            for (s, d, cls) in flows:
                if m == 'Dijkstra':
                    p = route_dijkstra(net, s, d, 'delay')
                elif m == 'ECMP':
                    p = route_ecmp(net, s, d, r2)
                elif m == 'Vanilla-DQN':
                    p = route_dqn(net, agent_van, s, d, cls, use_congestion=False)
                else:
                    p = route_dqn(net, agent_full, s, d, cls, use_congestion=True, use_potential=potential)
                if p is not None and len(p) >= 2:
                    mq = net.path_metrics(p, BIZ[cls]['rate'])
                    if mq is not None:
                        stat[m][cls][1] += 1
                        if qos_ok(cls, mq):
                            stat[m][cls][0] += 1
                        net.add_flow_load(p, BIZ[cls]['rate'])
                else:
                    stat[m][cls][1] += 1
    out = {}
    for m in methods:
        out[m] = {c: (stat[m][c][0] / stat[m][c][1] if stat[m][c][1] else 0)
                  for c in CLASSES}
    return out


if __name__ == '__main__':
    t0 = time.time()
    print("训练 Proposed = CA-MQR (拥塞感知+业务差异化+Dueling)...")
    agent_full, curve_full = train_dqn(use_congestion=True, dueling=True,
                                       episodes=5000, seed=1)
    print("训练 Vanilla-DQN (拥塞无感知, 普通DQN)...")
    agent_van, curve_van = train_dqn(use_congestion=False, dueling=False,
                                     episodes=5000, seed=1)

    loads = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    print("负载扫描评估...")
    scan = summarize_scan(evaluate_scan(agent_full, agent_van, loads, n_trials=20), loads)
    print("分业务 QoS 评估...")
    perclass = evaluate_perclass(agent_full, agent_van, n_flows=220, n_trials=30)

    # Dueling 消融: 多种子对比训练稳定性与收敛速度
    print("Dueling 消融(多种子)...")
    def conv_stats(dueling, seeds=(1, 2, 3)):
        finals, convs, curves = [], [], []
        for s in seeds:
            _, c = train_dqn(use_congestion=True, dueling=dueling,
                             episodes=3000, seed=s, log=False)
            c = np.array(c)
            finals.append(float(np.mean(c[-200:])))
            sm = np.convolve(c, np.ones(50) / 50, mode='valid')
            tgt = 0.8 * np.mean(c[-200:])
            idx = int(np.argmax(sm > tgt)) if (sm > tgt).any() else len(sm)
            convs.append(idx)
            curves.append(c.tolist())
        return {'final_mean': float(np.mean(finals)),
                'final_std': float(np.std(finals)),
                'conv_ep': float(np.mean(convs)),
                'curves': curves}
    abl_duel = conv_stats(True)
    abl_plain = conv_stats(False)

    # 下游拥塞势场增强实验(热点汇聚场景, 势场价值在此显现)
    print("下游拥塞势场增强实验(热点汇聚场景)...")
    agent_pot, _ = train_dqn(use_congestion=True, dueling=True,
                             use_potential=True, bg_hotspot=True,
                             episodes=4000, seed=1, log=False)
    agent_nopot, _ = train_dqn(use_congestion=True, dueling=True,
                               use_potential=False, bg_hotspot=True,
                               episodes=4000, seed=1, log=False)

    def eval_hotspot(agent, use_pot, loads_h, nfb=170, nt=20, seed0=555):
        con = Constellation(seed=42)
        out = {'delay': [], 'p95': [], 'sat': []}
        for ld in loads_h:
            nf = int(nfb * ld)
            ds, sat, tot = [], 0, 0
            for tr in range(nt):
                rng = np.random.default_rng(seed0 + tr + int(ld * 10))
                flows = gen_flows(con, rng, nf, hotspot=True)
                net = SatNetwork(con, seed=seed0 + tr)
                net.clear_load()
                for (s, dd, c) in flows:
                    p = route_dqn(net, agent, s, dd, c, use_congestion=True,
                                  use_potential=use_pot)
                    tot += 1
                    if p is not None and len(p) >= 2:
                        mq = net.path_metrics(p, BIZ[c]['rate'])
                        if mq is not None:
                            ds.append(mq['delay'] * 1000)
                            if qos_ok(c, mq):
                                sat += 1
                            net.add_flow_load(p, BIZ[c]['rate'])
            out['delay'].append(float(np.mean(ds)) if ds else 0)
            out['p95'].append(float(np.percentile(ds, 95)) if ds else 0)
            out['sat'].append(sat / tot if tot else 0)
        return out

    loads_h = [0.8, 1.0, 1.2, 1.4]
    pot_res = eval_hotspot(agent_pot, True, loads_h)
    nopot_res = eval_hotspot(agent_nopot, False, loads_h)

    json.dump({
        'curve_full': curve_full, 'curve_van': curve_van,
        'scan': scan, 'loads': loads, 'perclass': perclass,
        'abl_dueling': abl_duel, 'abl_plain': abl_plain,
        'potential': {'loads': loads_h, 'with': pot_res, 'without': nopot_res},
    }, open('../results/results.json', 'w'), indent=1)
    print(f"完成 用时{time.time()-t0:.0f}s")
    print("\n端到端时延(ms) 随负载:")
    for m in scan:
        print(f"  {m:12s}", [round(x, 0) for x in scan[m]['delay']])
    print("\n分业务QoS满足率:")
    for m in perclass:
        print(f"  {m:12s}", {c: round(perclass[m][c], 2) for c in CLASSES})
    print("\nDueling消融:")
    print(f"  Dueling : 终值{abl_duel['final_mean']:.1f} 标准差{abl_duel['final_std']:.2f} 收敛{abl_duel['conv_ep']:.0f}回合")
    print(f"  普通DQN : 终值{abl_plain['final_mean']:.1f} 标准差{abl_plain['final_std']:.2f} 收敛{abl_plain['conv_ep']:.0f}回合")
    print("\n势场增强(热点场景, load=0.8/1.0/1.2/1.4):")
    print(f"  有势场 时延{[round(x) for x in pot_res['delay']]} P95{[round(x) for x in pot_res['p95']]} QoS{[round(x,2) for x in pot_res['sat']]}")
    print(f"  无势场 时延{[round(x) for x in nopot_res['delay']]} P95{[round(x) for x in nopot_res['p95']]} QoS{[round(x,2) for x in nopot_res['sat']]}")
