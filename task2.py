import numpy as np
import time
from typing import TypedDict
from game import (
    Game2048, move, spawn_tile, is_game_over,
    get_valid_moves, get_empty_cells,
    run_random_agent, print_stats, render
)


class GameResult(TypedDict):
    score: int
    max_tile: int
    steps: int


class BenchmarkStats(TypedDict):
    n_games: int
    avg_score: float
    max_score: float
    avg_max_tile: float
    best_tile: int
    avg_steps: float
    total_time: float
    tile_dist: dict[int, int]


def _empty_score(board: np.ndarray) -> float:
    n = int(np.sum(board == 0))
    return np.log(n + 1) * 10.0


def _monotonicity(board: np.ndarray) -> float:
    score = 0.0

    # 横向（每行）
    for r in range(4):
        row = board[r]
        # 从左到右递增 or 递减各自的"惩罚"取较小的
        inc = dec = 0.0
        for c in range(3):
            if row[c] > 0 and row[c + 1] > 0:
                diff = np.log2(row[c] + 1) - np.log2(row[c + 1] + 1)
                if diff > 0:
                    dec += diff
                else:
                    inc -= diff
        score += max(inc, dec)

    # 纵向（每列）
    for c in range(4):
        col = board[:, c]
        inc = dec = 0.0
        for r in range(3):
            if col[r] > 0 and col[r + 1] > 0:
                diff = np.log2(col[r] + 1) - np.log2(col[r + 1] + 1)
                if diff > 0:
                    dec += diff
                else:
                    inc -= diff
        score += max(inc, dec)

    return score


def _smoothness(board: np.ndarray) -> float:
    score = 0.0
    for r in range(4):
        for c in range(4):
            if board[r, c] == 0:
                continue
            val = np.log2(board[r, c])
            # 右邻
            if c + 1 < 4 and board[r, c + 1] != 0:
                score -= abs(val - np.log2(board[r, c + 1]))
            # 下邻
            if r + 1 < 4 and board[r + 1, c] != 0:
                score -= abs(val - np.log2(board[r + 1, c]))
    return score


def _corner_bonus(board: np.ndarray) -> float:

    WEIGHTS = np.array([
        [  0,  1,  2,  3],
        [  7,  6,  5,  4],
        [  8,  9, 10, 11],
        [ 15, 14, 13, 12],
    ], dtype=float)

    score = 0.0
    for r in range(4):
        for c in range(4):
            if board[r, c] > 0:
                score += np.log2(board[r, c]) * WEIGHTS[r, c]
    return score


def _merge_potential(board: np.ndarray) -> float:
    score = 0.0
    for r in range(4):
        for c in range(4):
            if board[r, c] == 0:
                continue
            if c + 1 < 4 and board[r, c] == board[r, c + 1]:
                score += np.log2(board[r, c])
            if r + 1 < 4 and board[r, c] == board[r + 1, c]:
                score += np.log2(board[r, c])
    return score

DEFAULT_WEIGHTS: dict[str, float] = {
    'empty':   2.7,
    'mono':    1.5,
    'smooth':  0.1,
    'corner':  2.5,
    'merge':   0.8,
}


def evaluate(board: np.ndarray, weights: dict[str, float] | None = None) -> float:

    if weights is None:
        weights = DEFAULT_WEIGHTS

    score = 0.0
    score += weights['empty']  * _empty_score(board)
    score += weights['mono']   * _monotonicity(board)
    score += weights['smooth'] * _smoothness(board)
    score += weights['corner'] * _corner_bonus(board)
    score += weights['merge']  * _merge_potential(board)
    return score

DIRECTIONS = ['left', 'right', 'up', 'down']
ExpectimaxCache = dict[tuple[bytes, int, bool], float]
MAX_CHANCE_CELLS = 6


def _select_chance_cells(board: np.ndarray,
                         empty: list[tuple[int, int]],
                         limit: int = MAX_CHANCE_CELLS) -> list[tuple[int, int]]:
    if len(empty) <= limit:
        return empty

    def cell_priority(cell: tuple[int, int]) -> tuple[int, int, int]:
        r, c = cell
        adjacent_tiles = 0
        adjacent_value = 0
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < 4 and 0 <= nc < 4 and board[nr, nc] > 0:
                adjacent_tiles += 1
                adjacent_value += int(board[nr, nc])
        edge_bonus = int(r in (0, 3)) + int(c in (0, 3))
        return adjacent_tiles, adjacent_value, edge_bonus

    return sorted(empty, key=cell_priority, reverse=True)[:limit]


def expectimax(board: np.ndarray,
               depth: int,
               is_player: bool,
               weights: dict[str, float] | None = None,
               cache: ExpectimaxCache | None = None) -> float:

    key = (board.tobytes(), depth, is_player)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return cached

    if depth == 0 or is_game_over(board):
        value = evaluate(board, weights)
        if cache is not None:
            cache[key] = value
        return value

    if is_player:
        best = -float('inf')
        for d in DIRECTIONS:
            new_b, _, changed = move(board, d)
            if not changed:
                continue
            val = expectimax(new_b, depth - 1, False, weights, cache)
            if val > best:
                best = val
        if best == -float('inf'):
            best = evaluate(board, weights)
        if cache is not None:
            cache[key] = best
        return best

    else:
        empty = get_empty_cells(board)
        if not empty:
            value = expectimax(board, depth - 1, True, weights, cache)
            if cache is not None:
                cache[key] = value
            return value

        chance_cells = _select_chance_cells(board, empty)
        total = 0.0
        for (r, c) in chance_cells:
            for val, prob in [(2, 0.9), (4, 0.1)]:
                new_b = board.copy()
                new_b[r, c] = val
                total += prob * expectimax(new_b, depth - 1, True, weights, cache)
        value = total / len(chance_cells)
        if cache is not None:
            cache[key] = value
        return value


def choose_action(board: np.ndarray,
                  depth: int = 3,
                  weights: dict[str, float] | None = None) -> str | None:

    empty_cnt = int(np.sum(board == 0))
    if empty_cnt >= 8:
        actual_depth = min(depth, 2)
    elif empty_cnt <= 4:
        actual_depth = depth + 1
    else:
        actual_depth = depth

    best_score = -float('inf')
    best_dir = None
    cache: ExpectimaxCache = {}

    for d in DIRECTIONS:
        new_b, _, changed = move(board, d)
        if not changed:
            continue
        score = expectimax(new_b, actual_depth - 1, False, weights, cache)
        if score > best_score:
            best_score = score
            best_dir = d

    return best_dir

def run_one_game(depth: int = 3,
                 weights: dict[str, float] | None = None,
                 verbose: bool = False) -> GameResult:
    game = Game2048()

    while not game.over:
        if verbose:
            render(game.board, game.score, game.steps)
            time.sleep(0.05)

        action = choose_action(game.board, depth=depth, weights=weights)
        if action is None:
            break
        game.step(action)

    if verbose:
        render(game.board, game.score, game.steps)

    return {
        'score':    game.score,
        'max_tile': game.max_tile(),
        'steps':    game.steps,
    }


def run_benchmark(n_games: int = 20,
                  depth: int = 3,
                  weights: dict[str, float] | None = None,
                  verbose: bool = True) -> BenchmarkStats:
    scores, max_tiles, steps_list = [], [], []
    t0 = time.time()

    for i in range(n_games):
        result = run_one_game(depth=depth, weights=weights)
        scores.append(result['score'])
        max_tiles.append(result['max_tile'])
        steps_list.append(result['steps'])

        if verbose:
            elapsed = time.time() - t0
            avg_t = elapsed / (i + 1)
            remaining = avg_t * (n_games - i - 1)
            print(f"  [{i+1:>3}/{n_games}] "
                  f"得分={result['score']:>7}  "
                  f"最大块={result['max_tile']:>5}  "
                  f"步数={result['steps']:>5}  "
                  f"剩余≈{remaining:.0f}s")

    stats: BenchmarkStats = {
        'n_games':      n_games,
        'avg_score':    float(np.mean(scores)),
        'max_score':    float(np.max(scores)),
        'avg_max_tile': float(np.mean(max_tiles)),
        'best_tile':    int(np.max(max_tiles)),
        'avg_steps':    float(np.mean(steps_list)),
        'total_time':   time.time() - t0,
        'tile_dist': {
            t: int(np.sum(np.array(max_tiles) >= t))
            for t in [128, 256, 512, 1024, 2048]
        },
    }
    return stats

def demo(depth: int = 3):
    import os
    clear_cmd = 'cls' if os.name == 'nt' else 'clear'
    game = Game2048()
    print(f"\n  ══ AI 演示（depth={depth}）══  Ctrl+C 中断\n")
    try:
        while not game.over:
            os.system(clear_cmd)
            render(game.board, game.score, game.steps)
            action = choose_action(game.board, depth=depth)
            if action is None:
                break
            gained, _ = game.step(action)
            print(f"  动作: {action:<6}  获得: {gained}")
            time.sleep(0.15)
        os.system(clear_cmd)
        render(game.board, game.score, game.steps)
        print(f"\n  游戏结束  得分={game.score}  最大块={game.max_tile()}")
    except KeyboardInterrupt:
        print("\n  已中断")

if __name__ == '__main__':
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else 'bench'

    if mode == 'demo':
        depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
        demo(depth)

    elif mode == 'bench':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        depth = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        print(f"\n  Expectimax 启发式 AI（depth={depth}, {n} 局）\n")
        stats = run_benchmark(n_games=n, depth=depth, verbose=True)
        print_stats(stats)
        print(f"  总耗时: {stats['total_time']:.1f}s  "
              f"均速: {stats['total_time']/n:.1f}s/局\n")

    elif mode == 'compare':
        # 与随机策略对比
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        print(f"\n随机策略 ({n}局)")
        rand_stats = run_random_agent(n_games=n)
        print_stats(rand_stats)

        print(f"\nExpectimax depth=3 ({n}局)")
        ai_stats = run_benchmark(n_games=n, depth=3, verbose=False)
        print_stats(ai_stats)

        print(f"\n提升对比")
        print(f"  平均分提升: {ai_stats['avg_score']/rand_stats['avg_score']:.1f}x")
        print(f"  最大块提升: {ai_stats['best_tile']} vs {rand_stats['best_tile']}\n")
