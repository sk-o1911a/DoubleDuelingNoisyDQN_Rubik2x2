import numpy as np
from typing import List, Optional, Tuple, Callable, Dict
from Rubik2x2Env import apply_move_idx, is_solved

def h_simple(cube: np.ndarray) -> int:
    return sum(0 if (face == face[0, 0]).all() else 1 for face in cube)

def default_inverse(move_idx: int) -> int:
    return move_idx ^ 1

# IDA* algorithm for solving the Rubik's Cube
def ida_star_solve(start_cube: np.ndarray,
                num_actions: int = 12,
                max_depth: int = 14,
                heuristic: Callable[[np.ndarray], int] = h_simple,
                inverse_fn: Callable[[int], int] = default_inverse,
                use_tt: bool = True,
                tt_capacity: int = 200_000,) -> Optional[List[int]]:

    # bound is maximum cost allowed in current iteration
    bound = heuristic(start_cube)
    # tt is transposition table to store already visited states, avoiding redundant searches
    tt: Optional[Dict[bytes, int]] = {} if use_tt else None

    def search(cube: np.ndarray,
            g: int,
            bound: int,
            last_move: Optional[int],
            path: List[int],) -> Tuple[int, Optional[List[int]]]:

        f = g + heuristic(cube)
        if f > bound:
            return f, None
        if is_solved(cube):
            return f, list(path)

        f = g + heuristic(cube)
        if f > bound:
            return f, None
        if is_solved(cube):
            return f, list(path)

        # Transposition Table prune
        if tt is not None:
            key = cube.tobytes()
            prev_g = tt.get(key)
            if prev_g is not None and prev_g <= g:
                return float("inf"), None
            if len(tt) < tt_capacity:
                tt[key] = g

        min_next = float("inf")

        for move_idx in range(num_actions):
            if last_move is not None and move_idx == inverse_fn(last_move):
                continue

            next_cube = apply_move_idx(cube, move_idx)
            path.append(move_idx)

            next_f, result_path = search(next_cube, g + 1, bound, move_idx, path)

            if result_path is not None:
                return next_f, result_path
            path.pop()

            if next_f < min_next:
                min_next = next_f

        return min_next, None

    while bound <= max_depth:
        if tt is not None:
            tt.clear()
        next_bound, result_path = search(start_cube, 0, bound, None, [])
        if result_path is not None:
            return result_path
        if next_bound == float("inf"):
            return None
        bound = next_bound
    return None


