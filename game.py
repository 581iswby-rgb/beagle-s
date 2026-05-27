import numpy as np
import random
import copy
from typing import Any, Mapping

def new_board() -> np.ndarray:
    """创建全零 4×4 棋盘"""
    return np.zeros((4, 4), dtype=int)


def init_game() -> tuple[np.ndarray, int]:
    board = new_board()
    board = spawn_tile(board)
    board = spawn_tile(board)
    return board, 0

def spawn_tile(board: np.ndarray) -> np.ndarray:
    board = board.copy()
    empty_cells = list(zip(*np.where(board == 0)))
    if not empty_cells:
        return board
    row, col = random.choice(empty_cells)
    board[row, col] = 2 if random.random() < 0.9 else 4
    return board


def get_empty_cells(board: np.ndarray) -> list[tuple[int, int]]:
    return list(zip(*np.where(board == 0))) if np.any(board == 0) else []


def _merge_row_left(row: np.ndarray) -> tuple[np.ndarray, int]:

    tiles = row[row != 0]

    score = 0
    merged = []
    skip = False

    for i in range(len(tiles)):
        if skip:
            skip = False
            continue
        # 与下一个相同 → 合并
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            new_val = tiles[i] * 2
            merged.append(new_val)
            score += new_val
            skip = True
        else:
            merged.append(tiles[i])

    result = np.zeros(4, dtype=int)
    result[:len(merged)] = merged
    return result, score


def _move_left(board: np.ndarray) -> tuple[np.ndarray, int]:
    new_board = np.zeros((4, 4), dtype=int)
    total_score = 0
    for r in range(4):
        new_row, score = _merge_row_left(board[r])
        new_board[r] = new_row
        total_score += score
    return new_board, total_score


def move(board: np.ndarray, direction: str) -> tuple[np.ndarray, int, bool]:

    direction = direction.lower()
    rot_map = {
        'left':  0,
        'up':    1,   
        'right': 2,
        'down':  3,
    }
    if direction not in rot_map:
        raise ValueError(f"无效方向: {direction}，需为 left/right/up/down")

    k = rot_map[direction]
    rotated = np.rot90(board, k)          
    moved, score = _move_left(rotated)    
    result = np.rot90(moved, -k)          

    changed = not np.array_equal(result, board)
    return result, score, changed

def is_game_over(board: np.ndarray) -> bool:
    # 有空格 → 未结束
    if np.any(board == 0):
        return False
    # 尝试四个方向
    for d in ['left', 'right', 'up', 'down']:
        _, _, changed = move(board, d)
        if changed:
            return False
    return True


def get_max_tile(board: np.ndarray) -> int:
    return int(np.max(board))


def get_empty_count(board: np.ndarray) -> int:
    return int(np.sum(board == 0))


def get_valid_moves(board: np.ndarray) -> list[str]:
    valid = []
    for d in ['left', 'right', 'up', 'down']:
        _, _, changed = move(board, d)
        if changed:
            valid.append(d)
    return valid

class Game2048:

    def __init__(self):
        self.board, self.score = init_game()
        self.steps = 0
        self.over = False

    def step(self, direction: str) -> tuple[int, bool]:
 
        if self.over:
            return 0, True

        new_board, gained, changed = move(self.board, direction)

        if not changed:
            return 0, False 

        self.board = spawn_tile(new_board)
        self.score += gained
        self.steps += 1
        self.over = is_game_over(self.board)
        return gained, self.over

    def clone(self) -> 'Game2048':
        g = Game2048.__new__(Game2048)
        g.board = self.board.copy()
        g.score = self.score
        g.steps = self.steps
        g.over = self.over
        return g

    def valid_moves(self) -> list[str]:
        return get_valid_moves(self.board)

    def max_tile(self) -> int:
        return get_max_tile(self.board)

    def __repr__(self):
        return f"Game2048(score={self.score}, steps={self.steps}, max={self.max_tile()})"

_TILE_COLORS = {
    0:    '\033[90m',     
    2:    '\033[97m',     
    4:    '\033[93m',     
    8:    '\033[33m',     
    16:   '\033[91m',     
    32:   '\033[31m',     
    64:   '\033[35m',     
    128:  '\033[95m',     
    256:  '\033[94m',     
    512:  '\033[96m',     
    1024: '\033[92m',     
    2048: '\033[32m',     
}
_RESET = '\033[0m'


def render(board: np.ndarray, score: int, steps: int):
    print(f"\n  得分: {score:<8}  步数: {steps}")
    print("  ┌──────┬──────┬──────┬──────┐")
    for r in range(4):
        row_str = "  │"
        for c in range(4):
            v = board[r, c]
            color = _TILE_COLORS.get(v, '\033[32m')
            cell = f"{v:^6}" if v != 0 else "      "
            row_str += f"{color}{cell}{_RESET}│"
        print(row_str)
        if r < 3:
            print("  ├──────┼──────┼──────┼──────┤")
    print("  └──────┴──────┴──────┴──────┘")

def play_human():
    import sys

    KEY_MAP = {
        'w': 'up',    'k': 'up',
        'a': 'left',  'h': 'left',
        's': 'down',  'j': 'down',
        'd': 'right', 'l': 'right',
        'q': 'quit',
    }

    game = Game2048()
    print("       2048  —  终端版")
    print("  w/a/s/d 移动   q 退出")


    while not game.over:
        render(game.board, game.score, game.steps)
        valid = game.valid_moves()
        print(f"  可用方向: {valid}")

        try:
            key = input("  输入方向 > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  游戏退出")
            return

        if key == 'q':
            print("  游戏退出")
            return

        direction = KEY_MAP.get(key)
        if direction is None:
            print("无效按键，请重试")
            continue

        gained, done = game.step(direction)
        if gained == 0 and not done:
            # 判断是否真的无效（棋盘未变）
            # step() 返回 (0, False) 表示无效移动
            print("该方向无法移动")

    render(game.board, game.score, game.steps)
    print(f"\n  ══ 游戏结束 ══")
    print(f"  最终得分 : {game.score}")
    print(f"  最大方块 : {game.max_tile()}")
    print(f"  总步数   : {game.steps}")
    print()

def run_random_agent(n_games: int = 100, verbose: bool = False) -> dict:

    scores, max_tiles, steps_list = [], [], []

    for i in range(n_games):
        game = Game2048()
        while not game.over:
            valid = game.valid_moves()
            if not valid:
                break
            game.step(random.choice(valid))

        scores.append(game.score)
        max_tiles.append(game.max_tile())
        steps_list.append(game.steps)

        if verbose and (i + 1) % 10 == 0:
            print(f"  进度 {i+1}/{n_games} | "
                  f"均分={np.mean(scores):.0f} | "
                  f"最大块={max(max_tiles)}")

    stats = {
        'n_games':       n_games,
        'avg_score':     float(np.mean(scores)),
        'max_score':     float(np.max(scores)),
        'avg_max_tile':  float(np.mean(max_tiles)),
        'best_tile':     int(np.max(max_tiles)),
        'avg_steps':     float(np.mean(steps_list)),
        'tile_dist':     {
            t: int(np.sum(np.array(max_tiles) >= t))
            for t in [128, 256, 512, 1024, 2048]
        },
    }
    return stats


def print_stats(stats: Mapping[str, Any]):
    print("\n  ══════════════ 统计结果 ══════════════")
    print(f"  局数       : {stats['n_games']}")
    print(f"  平均得分   : {stats['avg_score']:.1f}")
    print(f"  最高得分   : {stats['max_score']:.0f}")
    print(f"  平均最大块 : {stats['avg_max_tile']:.1f}")
    print(f"  最大方块   : {stats['best_tile']}")
    print(f"  平均步数   : {stats['avg_steps']:.1f}")
    print("  到达率:")
    for tile, cnt in stats['tile_dist'].items():
        pct = cnt / stats['n_games'] * 100
        bar = '█' * int(pct / 5)
        print(f"    ≥{tile:>5} : {bar:<20} {pct:5.1f}%  ({cnt}/{stats['n_games']})")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'bench':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        print(f"\n  随机策略基准测试（{n} 局）...")
        stats = run_random_agent(n_games=n, verbose=True)
        print_stats(stats)
    else:
        # 默认：人工交互
        play_human()
