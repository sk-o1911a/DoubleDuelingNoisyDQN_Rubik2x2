from __future__ import annotations
import collections
from typing import Dict, Tuple, List, Optional, Callable
import numpy as np
from Rubik2x2Env import Rubik2x2Env, apply_move_idx, solved_cube

_env_tmp = Rubik2x2Env()
def default_inverse(move_idx: int) -> int:
    return _env_tmp._inverse_move_idx(move_idx)

#def _face_of(a: int) -> int:
    #return a // 2

def _apply_seq(cube: np.ndarray, seq: List[int]) -> np.ndarray:
    for m in seq:
        cube = apply_move_idx(cube, m)
    return cube

def _reconstruct_path(end_key: bytes, parents_fwd: Dict[bytes, Tuple[bytes | None, int | None]]) -> List[int]:
    path: List[int] = []
    cur = end_key
    while True:
        parent, move = parents_fwd[cur]
        if parent is None:
            break
        path.append(move)
        cur = parent
    path.reverse()
    return path

def bfs_solve(start_cube: np.ndarray,
            num_actions: int = 12,
            max_depth: int = 14,
            inverse_fn: Callable[[int], int] = default_inverse,
        ) -> Optional[List[int]]:

    start_cube = np.asarray(start_cube, dtype=np.int8).reshape(6, 2, 2)
    start_key = start_cube.tobytes()
    goal_cube = solved_cube()
    goal_key = goal_cube.tobytes()

    if start_key == goal_key:
        return []

    # deque: (state_key, last_move_or_None, depth)
    q = collections.deque([(start_key, None, 0)])
    parents: Dict[bytes, Tuple[bytes | None, int | None]] = {start_key: (None, None)}
    visited: set[bytes] = {start_key}

    while q:
        state_key, last_move, depth = q.popleft()
        if depth >= max_depth:
            continue

        cube = np.frombuffer(state_key, dtype=np.int8).copy().reshape(6, 2, 2)

        for a in range(num_actions):
            if last_move is not None and a == inverse_fn(last_move):
                continue
            #if last_move is not None and _face_of(a) == _face_of(last_move):
                #continue

            next_cube = apply_move_idx(cube.copy(), a)
            next_key = next_cube.tobytes()
            if next_key in visited:
                continue

            parents[next_key] = (state_key, a)
            visited.add(next_key)


            if next_key == goal_key:
                path = _reconstruct_path(next_key, parents)
                if np.array_equal(_apply_seq(start_cube.copy(), path), goal_cube):
                    return path

            q.append((next_key, a, depth + 1))


    return None
