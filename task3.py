"""
Task 3: 进化计算优化启发式权重
- 编码：5维实数向量 [w_empty, w_mono, w_smooth, w_corner, w_merge]
- 适应度：跑 N 局游戏的平均得分
- 选择：锦标赛选择
- 交叉：均匀交叉 + 算术交叉
- 变异：高斯扰动 + 边界裁剪
- 精英保留：每代保留最优 K 个个体
"""

import numpy as np
import time
import json
import os
from game import print_stats
from task2 import run_one_game, run_benchmark, DEFAULT_WEIGHTS

# ─────────────────────────────────────────────
#  权重向量编解码
# ─────────────────────────────────────────────

WEIGHT_KEYS = ['empty', 'mono', 'smooth', 'corner', 'merge']

# 每个维度的搜索范围 [low, high]
BOUNDS = np.array([
    [0.5,  5.0],   # empty
    [0.5,  4.0],   # mono
    [0.01, 1.0],   # smooth
    [0.5,  5.0],   # corner
    [0.1,  2.0],   # merge
])

def vec_to_weights(vec: np.ndarray) -> dict:
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, vec)}

def weights_to_vec(weights: dict) -> np.ndarray:
    return np.array([weights[k] for k in WEIGHT_KEYS])

def clip(vec: np.ndarray) -> np.ndarray:
    """将向量裁剪到各维度合法范围内"""
    return np.clip(vec, BOUNDS[:, 0], BOUNDS[:, 1])

def random_individual() -> np.ndarray:
    """在搜索范围内随机生成一个个体"""
    return BOUNDS[:, 0] + np.random.rand(5) * (BOUNDS[:, 1] - BOUNDS[:, 0])


# ─────────────────────────────────────────────
#  适应度函数
# ─────────────────────────────────────────────

def fitness(vec: np.ndarray,
            n_games: int = 5,
            depth: int = 2) -> float:
    """
    适应度 = 跑 n_games 局的平均得分。
    depth=2 加快评估速度（进化过程中精度/速度的折中）。
    最终验证时用 depth=3、n_games=20。
    """
    weights = vec_to_weights(vec)
    scores = []
    for _ in range(n_games):
        result = run_one_game(depth=depth, weights=weights)
        scores.append(result['score'])
    return float(np.mean(scores))


# ─────────────────────────────────────────────
#  遗传算子
# ─────────────────────────────────────────────

def tournament_select(population: list[np.ndarray],
                      fitnesses: list[float],
                      k: int = 3) -> np.ndarray:
    """
    锦标赛选择：随机抽 k 个个体，返回其中适应度最高的。
    k 越大选择压力越强，k=3 是常用默认值。
    """
    idx = np.random.choice(len(population), size=k, replace=False)
    best = idx[np.argmax([fitnesses[i] for i in idx])]
    return population[best].copy()


def crossover(p1: np.ndarray, p2: np.ndarray,
              alpha: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    均匀算术交叉：
      c1[i] = alpha * p1[i] + (1-alpha) * p2[i]
      c2[i] = alpha * p2[i] + (1-alpha) * p1[i]
    alpha 从 [0.3, 0.7] 随机取，增加多样性。
    """
    alpha = np.random.uniform(0.3, 0.7)
    c1 = alpha * p1 + (1 - alpha) * p2
    c2 = alpha * p2 + (1 - alpha) * p1
    return clip(c1), clip(c2)


def mutate(vec: np.ndarray,
           rate: float = 0.3,
           sigma: float = 0.3) -> np.ndarray:
    """
    高斯变异：每个维度以 rate 概率加高斯噪声。
    sigma 随代数衰减（调用时传入），实现"先粗后细"搜索。
    """
    child = vec.copy()
    for i in range(len(child)):
        if np.random.rand() < rate:
            child[i] += np.random.randn() * sigma * (BOUNDS[i, 1] - BOUNDS[i, 0])
    return clip(child)


# ─────────────────────────────────────────────
#  遗传算法主循环
# ─────────────────────────────────────────────

def genetic_algorithm(
    pop_size:    int   = 20,    # 种群大小
    n_gen:       int   = 15,    # 进化代数
    elite_k:     int   = 2,     # 精英保留数
    n_games_fit: int   = 5,     # 适应度评估局数
    depth_fit:   int   = 2,     # 适应度评估深度
    sigma_init:  float = 0.4,   # 初始变异幅度
    sigma_decay: float = 0.92,  # 变异幅度衰减系数
    save_path:   str   = 'evo_result.json',
    verbose:     bool  = True,
) -> dict:
    """
    遗传算法主函数，返回最优权重及进化历史。

    参数说明：
      pop_size    : 种群大小，越大搜索越充分但越慢
      n_gen       : 进化代数
      elite_k     : 每代直接保留的最优个体数（精英策略）
      n_games_fit : 每次适应度评估的游戏局数
      depth_fit   : 评估时的搜索深度（2=快，3=精）
      sigma_init  : 初始高斯变异幅度（相对于范围的比例）
      sigma_decay : 每代乘以该系数，实现变异幅度退火
    """
    t_start = time.time()

    # ── 初始种群 ──
    # 把当前默认权重也放入初始种群（热启动）
    population = [weights_to_vec(DEFAULT_WEIGHTS)]
    population += [random_individual() for _ in range(pop_size - 1)]

    history = []   # 每代最优适应度
    best_vec = population[0].copy()
    best_fit = -float('inf')
    sigma = sigma_init

    for gen in range(n_gen):
        gen_t = time.time()

        # ── 评估适应度 ──
        if verbose:
            print(f"\n  第 {gen+1}/{n_gen} 代  (σ={sigma:.3f})")
        fitnesses = []
        for i, ind in enumerate(population):
            f = fitness(ind, n_games=n_games_fit, depth=depth_fit)
            fitnesses.append(f)
            if verbose:
                print(f"    个体{i+1:>2}  权重={np.round(ind,2)}  适应度={f:.0f}")

        # ── 更新全局最优 ──
        gen_best_idx = int(np.argmax(fitnesses))
        gen_best_fit = fitnesses[gen_best_idx]
        gen_best_vec = population[gen_best_idx]

        if gen_best_fit > best_fit:
            best_fit = gen_best_fit
            best_vec = gen_best_vec.copy()

        history.append({
            'gen':      gen + 1,
            'best':     gen_best_fit,
            'mean':     float(np.mean(fitnesses)),
            'worst':    float(np.min(fitnesses)),
            'best_vec': gen_best_vec.tolist(),
        })

        if verbose:
            elapsed = time.time() - t_start
            print(f"  ── 代最优={gen_best_fit:.0f}  "
                  f"代均值={np.mean(fitnesses):.0f}  "
                  f"全局最优={best_fit:.0f}  "
                  f"耗时={time.time()-gen_t:.0f}s  累计={elapsed:.0f}s")

        # ── 构造下一代 ──
        # 1. 精英保留
        elite_idx = np.argsort(fitnesses)[-elite_k:]
        next_pop = [population[i].copy() for i in elite_idx]

        # 2. 选择 + 交叉 + 变异 填满种群
        while len(next_pop) < pop_size:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            next_pop.append(mutate(c1, sigma=sigma))
            if len(next_pop) < pop_size:
                next_pop.append(mutate(c2, sigma=sigma))

        population = next_pop[:pop_size]

        # 3. 变异幅度退火
        sigma *= sigma_decay

    # ── 保存结果 ──
    result = {
        'best_weights': vec_to_weights(best_vec),
        'best_fitness': best_fit,
        'history':      history,
        'total_time':   time.time() - t_start,
        'config': {
            'pop_size': pop_size, 'n_gen': n_gen,
            'n_games_fit': n_games_fit, 'depth_fit': depth_fit,
        }
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  ══ 进化结束 ══  总耗时={result['total_time']:.0f}s")
        print(f"  最优权重: {result['best_weights']}")
        print(f"  最优适应度: {best_fit:.0f}")
        print(f"  结果已保存至 {save_path}")

    return result


# ─────────────────────────────────────────────
#  进化策略（ES）备选方案
# ─────────────────────────────────────────────

def evolution_strategy(
    mu:          int   = 5,     # 父代数量
    lam:         int   = 20,    # 子代数量（μ+λ 策略）
    n_gen:       int   = 15,
    sigma_init:  float = 0.3,
    n_games_fit: int   = 5,
    depth_fit:   int   = 2,
    save_path:   str   = 'es_result.json',
    verbose:     bool  = True,
) -> dict:
    """
    (μ+λ) 进化策略。
    每代从 μ 个父代产生 λ 个子代，
    从 μ+λ 个体中选出最优 μ 个作为下一代父代。
    σ 自适应：若连续3代无改善则增大 σ 重新扩散。
    """
    t_start = time.time()

    # 初始父代
    parents = [random_individual() for _ in range(mu)]
    parents[0] = weights_to_vec(DEFAULT_WEIGHTS)   # 热启动

    best_vec = parents[0].copy()
    best_fit = -float('inf')
    history = []
    sigma = sigma_init
    no_improve = 0

    for gen in range(n_gen):
        # 生成子代：每个父代产生 lam//mu 个子代
        offspring = []
        for p in parents:
            for _ in range(lam // mu):
                child = mutate(p, rate=0.5, sigma=sigma)
                offspring.append(child)

        # 评估 μ+λ 个体
        pool = parents + offspring
        fits = [fitness(ind, n_games=n_games_fit, depth=depth_fit)
                for ind in pool]

        # 选出最优 μ 个
        top_idx = np.argsort(fits)[-mu:]
        parents = [pool[i].copy() for i in top_idx]
        top_fits = [fits[i] for i in top_idx]

        gen_best = max(top_fits)
        if gen_best > best_fit:
            best_fit = gen_best
            best_vec = parents[np.argmax(top_fits)].copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:       # 连续3代无改善 → 扩散
                sigma = min(sigma * 1.3, sigma_init * 2)
                no_improve = 0
                if verbose:
                    print(f"  ⚠ 连续3代无改善，σ 扩大至 {sigma:.3f}")

        sigma *= 0.93   # 整体退火

        history.append({'gen': gen+1, 'best': gen_best,
                         'mean': float(np.mean(top_fits))})
        if verbose:
            print(f"  ES 第{gen+1:>2}代  best={gen_best:.0f}  "
                  f"mean={np.mean(top_fits):.0f}  "
                  f"全局={best_fit:.0f}  σ={sigma:.3f}")

    result = {
        'best_weights': vec_to_weights(best_vec),
        'best_fitness': best_fit,
        'history':      history,
        'total_time':   time.time() - t_start,
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  ES 结束  最优权重={result['best_weights']}  "
              f"耗时={result['total_time']:.0f}s")
    return result


# ─────────────────────────────────────────────
#  进化结果可视化（文字版）
# ─────────────────────────────────────────────

def print_history(history: list[dict]):
    """用 ASCII 折线图展示进化曲线"""
    bests = [h['best'] for h in history]
    means = [h['mean'] for h in history]
    lo, hi = min(means) * 0.9, max(bests) * 1.05
    height = 12

    print("\n  进化曲线（● 最优  · 均值）")
    print(f"  {hi:>7.0f} ┐")
    for row in range(height, -1, -1):
        threshold = lo + (hi - lo) * row / height
        line = "         │"
        for h in history:
            b_here = h['best'] >= threshold
            m_here = h['mean'] >= threshold
            if b_here:
                line += '●'
            elif m_here:
                line += '·'
            else:
                line += ' '
        print(line)
    print(f"  {lo:>7.0f} └" + "─" * len(history))
    gen_labels = "          " + "".join(
        str(h['gen'] % 10) for h in history)
    print(gen_labels + "  代")


def compare_weights(original: dict, optimized: dict):
    """对比默认权重与优化权重"""
    print("\n  ══ 权重对比 ══")
    print(f"  {'参数':<8} {'默认':>8} {'优化后':>8} {'变化':>8}")
    print("  " + "─" * 36)
    for k in WEIGHT_KEYS:
        orig = original.get(k, 0)
        opt  = optimized.get(k, 0)
        arrow = '↑' if opt > orig else '↓'
        print(f"  {k:<8} {orig:>8.3f} {opt:>8.3f} {arrow}{abs(opt-orig):>6.3f}")


# ─────────────────────────────────────────────
#  验证：用优化权重跑完整基准
# ─────────────────────────────────────────────

def validate(weights: dict, n_games: int = 20, depth: int = 3):
    """用 depth=3 完整基准验证优化后权重"""
    print(f"\n  验证优化权重（depth={depth}, {n_games}局）...")
    print(f"  权重: {weights}\n")
    stats = run_benchmark(n_games=n_games, depth=depth,
                          weights=weights, verbose=True)
    print_stats(stats)

    # 同条件跑默认权重对比
    print(f"\n  对比默认权重（depth={depth}, {n_games}局）...")
    default_stats = run_benchmark(n_games=n_games, depth=depth,
                                  weights=None, verbose=False)
    print_stats(default_stats)

    print("  ── 提升 ──")
    print(f"  平均分: {default_stats['avg_score']:.0f} → {stats['avg_score']:.0f}"
          f"  (+{(stats['avg_score']/default_stats['avg_score']-1)*100:.1f}%)")
    print(f"  最大块: {default_stats['best_tile']} → {stats['best_tile']}")
    return stats


# ─────────────────────────────────────────────
#  入口
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'ga'

    if mode == 'ga':
        # 遗传算法（推荐）
        # 快速模式：pop=10, gen=8, 用于调试
        # 正式模式：pop=20, gen=15
        quick = '--quick' in sys.argv
        result = genetic_algorithm(
            pop_size    = 10 if quick else 20,
            n_gen       = 8  if quick else 15,
            elite_k     = 2,
            n_games_fit = 3  if quick else 5,
            depth_fit   = 2,
            save_path   = 'ga_result.json',
            verbose     = True,
        )
        print_history(result['history'])
        compare_weights(DEFAULT_WEIGHTS, result['best_weights'])

        # 询问是否验证
        ans = input("\n  是否用 depth=3 验证最优权重？(y/n) ").strip().lower()
        if ans == 'y':
            validate(result['best_weights'], n_games=20, depth=3)

    elif mode == 'es':
        # 进化策略备选
        result = evolution_strategy(
            mu=5, lam=20, n_gen=15,
            save_path='es_result.json', verbose=True
        )
        print_history(result['history'])
        compare_weights(DEFAULT_WEIGHTS, result['best_weights'])

    elif mode == 'validate':
        # 从已有结果文件验证
        path = sys.argv[2] if len(sys.argv) > 2 else 'ga_result.json'
        n    = int(sys.argv[3]) if len(sys.argv) > 3 else 20
        if not os.path.exists(path):
            print(f"  ✗ 找不到文件 {path}")
            sys.exit(1)
        with open(path) as f:
            result = json.load(f)
        print_history(result['history'])
        compare_weights(DEFAULT_WEIGHTS, result['best_weights'])
        validate(result['best_weights'], n_games=n, depth=3)

    else:
        print("  用法:")
        print("    python evolution.py ga [--quick]   # 遗传算法")
        print("    python evolution.py es             # 进化策略")
        print("    python evolution.py validate [文件] [局数]  # 验证已有结果")
