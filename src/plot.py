"""生成论文插图(论文风格: 简洁, 低饱和, 专业)。"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from constellation import Constellation

rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Songti SC']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 10.5
rcParams['axes.linewidth'] = 0.8
rcParams['figure.dpi'] = 300

FIG = '../figures/'
# 论文风格配色: 灰阶 + 少量低饱和色, 不同 marker 区分
STYLE = {
    'Random':      dict(color='#999999', marker='v', ls=':'),
    'Dijkstra':    dict(color='#4C72B0', marker='o', ls='--'),
    'ECMP':        dict(color='#55A868', marker='s', ls='-.'),
    'Vanilla-DQN': dict(color='#C44E52', marker='^', ls='--'),
    'Proposed':    dict(color='#000000', marker='D', ls='-'),
}
LABELS = {'Random': 'Random', 'Dijkstra': 'Dijkstra', 'ECMP': 'ECMP',
          'Vanilla-DQN': 'Vanilla-DQN', 'Proposed': '本文方法'}

d = json.load(open('../results/results.json'))
loads = d['loads']


def fig_convergence():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    def smooth(x, k=50):
        x = np.array(x)
        return np.convolve(x, np.ones(k) / k, mode='valid')
    cf = smooth(d['curve_full'])
    cv = smooth(d['curve_van'])
    ax.plot(cf, color='#000000', lw=1.2, label='本文方法')
    ax.plot(cv, color='#C44E52', lw=1.0, ls='--', label='Vanilla-DQN')
    ax.set_xlabel('训练回合 (episode)')
    ax.set_ylabel('回合累计奖励')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_convergence.png', bbox_inches='tight')
    plt.close()


def fig_delay():
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for m in ['Random', 'Dijkstra', 'ECMP', 'Vanilla-DQN', 'Proposed']:
        ax.plot(loads, d['scan'][m]['delay'], label=LABELS[m], lw=1.3,
                markersize=5, **STYLE[m])
    ax.set_xlabel('归一化网络负载')
    ax.set_ylabel('平均端到端时延 (ms)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_delay.png', bbox_inches='tight')
    plt.close()


def fig_util():
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for m in ['Dijkstra', 'ECMP', 'Vanilla-DQN', 'Proposed']:
        ax.plot(loads, d['scan'][m]['util'], label=LABELS[m], lw=1.3,
                markersize=5, **STYLE[m])
    ax.set_xlabel('归一化网络负载')
    ax.set_ylabel('链路利用率 (95 分位)')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_util.png', bbox_inches='tight')
    plt.close()


def fig_perclass():
    pc = d['perclass']
    methods = ['Dijkstra', 'ECMP', 'Vanilla-DQN', 'Proposed']
    classes = ['A', 'B', 'C']
    cls_name = {'A': 'A类(时延敏感)', 'B': 'B类(带宽密集)', 'C': 'C类(可靠性)'}
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    x = np.arange(len(classes))
    w = 0.2
    hatches = ['', '//', '\\\\', 'xx']
    grays = ['#BBBBBB', '#888888', '#555555', '#000000']
    for i, m in enumerate(methods):
        vals = [pc[m][c] for c in classes]
        ax.bar(x + (i - 1.5) * w, vals, w, label=LABELS[m],
               color=grays[i], edgecolor='black', lw=0.6, hatch=hatches[i])
    ax.set_xticks(x)
    ax.set_xticklabels([cls_name[c] for c in classes], fontsize=9)
    ax.set_ylabel('QoS 满足率')
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper center')
    ax.grid(True, axis='y', ls=':', lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_perclass.png', bbox_inches='tight')
    plt.close()


def fig_constellation():
    """Walker 星座 + Grid 拓扑示意图(2D 投影)。"""
    con = Constellation(seed=42)
    pos = con.sat_positions(0.0)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    # 投影到 x-y 平面
    for i in range(con.N):
        nb = con.grid_neighbors(i)
        for d_, j in nb.items():
            ax.plot([pos[i, 0] / 1e6, pos[j, 0] / 1e6],
                    [pos[i, 1] / 1e6, pos[j, 1] / 1e6],
                    color='#CCCCCC', lw=0.4, zorder=1)
    # 按轨道面着色(灰阶)
    for p in range(con.P):
        idx = [p * con.S + s for s in range(con.S)]
        ax.scatter(pos[idx, 0] / 1e6, pos[idx, 1] / 1e6, s=18,
                   color=plt.cm.Greys(0.3 + 0.6 * p / con.P),
                   edgecolor='black', lw=0.4, zorder=2)
    # 地球
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(6.371 * np.cos(theta), 6.371 * np.sin(theta),
            color='#4C72B0', lw=1.0)
    ax.fill(6.371 * np.cos(theta), 6.371 * np.sin(theta),
            color='#E8F0F8', zorder=0)
    ax.text(0, 0, '地球', ha='center', va='center', fontsize=9, color='#4C72B0')
    ax.set_xlabel('X (×10³ km)')
    ax.set_ylabel('Y (×10³ km)')
    ax.set_aspect('equal')
    ax.grid(True, ls=':', lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_constellation.png', bbox_inches='tight')
    plt.close()


def fig_satrate_load():
    """整体 QoS 满足率随负载(用 scan 的 sat)。"""
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    for m in ['Random', 'Dijkstra', 'ECMP', 'Vanilla-DQN', 'Proposed']:
        ax.plot(loads, d['scan'][m]['sat'], label=LABELS[m], lw=1.3,
                markersize=5, **STYLE[m])
    ax.set_xlabel('归一化网络负载')
    ax.set_ylabel('总体 QoS 满足率')
    ax.legend(frameon=False, fontsize=8.5, loc='lower left')
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_satrate.png', bbox_inches='tight')
    plt.close()


def fig_dueling():
    """Dueling 消融: 多种子收敛曲线均值±标准差, 体现训练稳定性。"""
    def band(curves, k=80):
        arr = np.array([np.convolve(c, np.ones(k) / k, mode='valid')
                        for c in curves])
        L = min(a.shape[0] for a in arr)
        arr = np.array([a[:L] for a in arr])
        return arr.mean(0), arr.std(0)
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    md, sd = band(d['abl_dueling']['curves'])
    mp, sp = band(d['abl_plain']['curves'])
    xd = np.arange(len(md)); xp = np.arange(len(mp))
    ax.plot(xd, md, color='#000000', lw=1.2, label='Dueling Double DQN（本文）')
    ax.fill_between(xd, md - sd, md + sd, color='#000000', alpha=0.15)
    ax.plot(xp, mp, color='#C44E52', lw=1.0, ls='--', label='普通 Double DQN')
    ax.fill_between(xp, mp - sp, mp + sp, color='#C44E52', alpha=0.15)
    ax.set_xlabel('训练回合 (episode)')
    ax.set_ylabel('回合累计奖励')
    ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    ax.grid(True, ls=':', lw=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_dueling.png', bbox_inches='tight')
    plt.close()


def fig_potential():
    """下游拥塞势场增强: 热点汇聚场景下 有/无势场 的时延与QoS对比。"""
    p = d['potential']
    loads_h = p['loads']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # 左: P95 尾部时延
    ax1.plot(loads_h, p['with']['p95'], color='#000000', marker='D', lw=1.3,
             markersize=5, label='引入势场（本文）')
    ax1.plot(loads_h, p['without']['p95'], color='#C44E52', marker='^',
             ls='--', lw=1.2, markersize=5, label='未引入势场')
    ax1.set_xlabel('归一化网络负载')
    ax1.set_ylabel('P95 端到端时延 (ms)')
    ax1.legend(frameon=False, fontsize=8.5, loc='upper left')
    ax1.grid(True, ls=':', lw=0.5, alpha=0.6)
    ax1.set_title('(a) P95 尾部时延', fontsize=9.5)
    # 右: QoS 满足率
    ax2.plot(loads_h, p['with']['sat'], color='#000000', marker='D', lw=1.3,
             markersize=5, label='引入势场（本文）')
    ax2.plot(loads_h, p['without']['sat'], color='#C44E52', marker='^',
             ls='--', lw=1.2, markersize=5, label='未引入势场')
    ax2.set_xlabel('归一化网络负载')
    ax2.set_ylabel('QoS 满足率')
    ax2.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax2.grid(True, ls=':', lw=0.5, alpha=0.6)
    ax2.set_title('(b) QoS 满足率', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_potential.png', bbox_inches='tight')
    plt.close()


def fig_scenario():
    """仿真场景与业务流转示意: 卫星网格 + 源->逐跳中继->目的 + 热点汇聚。"""
    P, S = 6, 11
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    # 用规则网格表示 +Grid 拓扑(逻辑视图, 便于展示业务流转)
    xs = {}
    for p in range(P):
        for s in range(S):
            xs[(p, s)] = (s, p)
    # 画网格链路(横=同轨道面, 竖=相邻轨道面)
    for p in range(P):
        for s in range(S):
            if s + 1 < S:
                ax.plot([s, s + 1], [p, p], color='#DDDDDD', lw=0.6, zorder=1)
            if p + 1 < P:
                ax.plot([s, s], [p, p + 1], color='#DDDDDD', lw=0.6, zorder=1)
    # 画卫星节点
    for p in range(P):
        for s in range(S):
            ax.scatter(s, p, s=40, color='white', edgecolor='#888888',
                       lw=0.8, zorder=2)
    # 一条业务流: 源 -> 逐跳 -> 目的 (折线路径), 节点用(p,s)表示
    path = [(1, 0), (1, 1), (1, 2), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
    px = [xs[n][0] for n in path]
    py = [xs[n][1] for n in path]
    ax.plot(px, py, color='#000000', lw=1.8, zorder=3,
            marker='o', markersize=4, markerfacecolor='#000000')
    # 源、目的标注
    ax.scatter(*xs[path[0]], s=140, color='#4C72B0', edgecolor='black',
               lw=1, zorder=4, marker='s')
    ax.scatter(*xs[path[-1]], s=140, color='#55A868', edgecolor='black',
               lw=1, zorder=4, marker='*')
    ax.annotate('源卫星', xy=xs[path[0]], xytext=(-0.3, -0.6),
                fontsize=9, color='#4C72B0')
    ax.annotate('目的卫星', xy=xs[path[-1]], xytext=(5.3, 2.5),
                fontsize=9, color='#55A868')
    ax.annotate('逐跳中继转发', xy=(2.5, 1.6), fontsize=9, color='#000000')
    # 热点节点(业务汇聚, 拥塞)
    for hp in [(8, 3), (9, 4)]:
        ax.scatter(*hp, s=180, color='#C44E52', edgecolor='black',
                   lw=1, zorder=4, alpha=0.8)
    ax.annotate('热点\n(业务汇聚)', xy=(8.5, 3.5), xytext=(7.2, 4.6),
                fontsize=8.5, color='#C44E52', ha='center')
    ax.set_xlabel('面内卫星编号')
    ax.set_ylabel('轨道面编号')
    ax.set_xlim(-1, S)
    ax.set_ylim(-1.2, P)
    ax.set_xticks(range(S))
    ax.set_yticks(range(P))
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(FIG + 'fig_scenario.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    fig_constellation()
    fig_scenario()
    fig_convergence()
    fig_delay()
    fig_util()
    fig_perclass()
    fig_satrate_load()
    fig_dueling()
    fig_potential()
    print("图已生成到", FIG)
