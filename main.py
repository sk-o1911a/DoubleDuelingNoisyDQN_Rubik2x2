import argparse
import os
import time
from collections import deque
import numpy as np
import torch
from Rubik2x2Env import Rubik2x2Env, MOVE_FUNCS
from DQN_Agent import DQNAgent
from Replay_Buffer import ReplayBuffer
from Search_BFS import bfs_solve
#from Search_IDA import ida_star_solve
torch.set_float32_matmul_precision("high")


MOVE_NAMES = [name for (name, _) in MOVE_FUNCS.values()]

def parse_args():
    p = argparse.ArgumentParser("Train DQN (Double+Dueling+NoisyNet) for 2x2 with optional BFS hints")
    # Train
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--start_scramble", type=int, default=2)
    p.add_argument("--max_scramble", type=int, default=8)
    p.add_argument("--curriculum_every", type=int, default=2000)
    p.add_argument("--buffer_capacity", type=int, default=300_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--n_step", type=int, default=3)
    # SR-gated curriculum (Cách A)
    p.add_argument("--sr_window", type=int, default=800, help="cửa sổ SR gần đây")
    p.add_argument("--sr_hi", type=float, default=0.85, help="ngưỡng tăng độ khó")
    p.add_argument("--sr_lo", type=float, default=0.40, help="ngưỡng giảm độ khó")
    # BFS hỗ trợ học
    p.add_argument("--plan_prob_start", type=float, default=0.20, help="xác suất dùng BFS cho bước hiện tại")
    p.add_argument("--plan_prob_min", type=float, default=0.05)
    p.add_argument("--plan_prob_decay", type=float, default=0.7, help="giảm khi tăng độ khó")
    p.add_argument("--bfs_depth", type=int, default=12)
    # Eval
    p.add_argument("--eval_trials", type=int, default=10)
    p.add_argument("--eval_agent_steps", type=int, default=30)
    p.add_argument("--eval_bfs_depth", type=int, default=14)
    # IO
    p.add_argument("--save_path", type=str, default="rubik2x2_best.pt")
    p.add_argument("--no_save", action="store_true")
    return p.parse_args()

def make_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def train(args):
    device = make_device()
    env = Rubik2x2Env(scramble_len=args.start_scramble, max_steps=100, use_action_mask=True)  # :contentReference[oaicite:5]{index=5}

    agent = DQNAgent(
        input_dim=144,
        num_actions=env.action_space.n,
        lr=1e-3,
        gamma=args.gamma,
        tau=0.01,
        dueling=True,
        noisy=True,
        sigma0=0.5,
        device=device,
    )

    buf = ReplayBuffer(capacity=args.buffer_capacity, n_step=args.n_step,
                       gamma=args.gamma)  # :contentReference[oaicite:7]{index=7}

    plan_p = args.plan_prob_start
    sr_window = deque(maxlen=args.sr_window)
    best_sr = 0.0
    t0 = time.time()
    total_ep = 0
    total_solved = 0
    print("=== Training start ===")
    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_loss = 0.0
        steps = 0

        while not (done or truncated):
            steps += 1
            action_mask = info.get("action_mask", None)

            use_plan = np.random.rand() < plan_p
            if use_plan:
                plan = bfs_solve(env.cube.copy(),
                                              num_actions=env.action_space.n,
                                              max_depth=args.bfs_depth)
                if plan is not None and len(plan) > 0:
                    a = plan[0]
                    if action_mask is not None and not action_mask[a]:
                        use_plan = False
                else:
                    use_plan = False

            if not use_plan:
                a = agent.act(obs, action_mask=action_mask, eval_mode=False)

            obs2, r, done, truncated, info = env.step(a)
            buf.add(obs, a, r, obs2, done)
            obs = obs2

            if len(buf) >= max(args.warmup, args.batch_size):
                batch = buf.sample(batch_size=args.batch_size, device=agent.device)
                ep_loss += agent.update(batch, double=True)

            if done or truncated:
                sr_window.append(1 if done else 0)
                buf.reset_episode()

            total_ep += 1
            total_solved += int(done)

# ---- SR-gated curriculum logic (Cách A) ----
        if len(sr_window) == sr_window.maxlen:
            sr = sum(sr_window) / len(sr_window)
            # tăng độ khó nếu SR cao, giảm nếu SR thấp (có hysteresis hi/lo)
            if sr >= args.sr_hi and env.scramble_len < args.max_scramble:
                env.scramble_len += 1
                # khi tăng khó: tạm tăng nhẹ plan_p để “đỡ gãy” rồi giảm dần lại
                plan_p = min(0.25, plan_p * (1.0 / max(1e-6, args.plan_prob_decay)))
                sr_window.clear()
            elif sr <= args.sr_lo and env.scramble_len > args.start_scramble:
                env.scramble_len -= 1
                # khi hạ khó: có thể giảm trợ giúp một chút
                plan_p = max(args.plan_prob_min, plan_p * args.plan_prob_decay)
                sr_window.clear()

        # ---- logging nhẹ mỗi 200 ep ----
        if ep % 10 == 0:
            window = len(sr_window)
            sr_recent = 100.0 * (sum(sr_window) / max(1, window)) if window else 0.0
            sr_life = 100.0 * (total_solved / max(1, total_ep))
            best_sr = max(best_sr, sr_recent)
            avg_loss = (ep_loss / max(1, steps))
            elapsed_m = (time.time() - t0) / 60.0
            print(
                f"[Ep {ep:>6}/{args.episodes}] "
                f"scr={env.scramble_len} "
                f"avg_loss/step={avg_loss:.4f} "
                f"SR_recent({window})={sr_recent:.1f}% "
                f"SR_all={sr_life:.1f}% "
                f"best_recent={best_sr:.1f}% "
                f"plan_p={plan_p:.2f} "
                f"elapsed={elapsed_m:.1f}m"
            )

    return agent, env

def eval_once(env, agent, max_agent_steps=30, bfs_depth=14):
    obs, info = env.reset()
    plan_taken = []
    agent.q.eval()  # tắt NoisyNet trong suy luận

    for _ in range(max_agent_steps):
        a = agent.act(obs, action_mask=info.get("action_mask"), eval_mode=True)
        obs, r, done, truncated, info = env.step(a)
        plan_taken.append(a)
        if done:
            return plan_taken, True

    # fallback: IDA* tối ưu nốt
    plan = bfs_solve(env.cube.copy(),
                                  num_actions=env.action_space.n,
                                  max_depth=bfs_depth)
    if plan is None:
        return plan_taken, False
    plan_taken.extend(plan)
    for a in plan:  # (tuỳ chọn) áp dụng plan còn lại để đảm bảo solved
        env.step(a)
    return plan_taken, True

def main():
    args = parse_args()
    agent, env = train(args)

    # lưu checkpoint (nếu hỗ trợ)
    if not args.no_save:
        try:
            agent.save(args.save_path)
            print(f"Saved to {os.path.abspath(args.save_path)}")
        except Exception as e:
            print("Skip saving:", repr(e))

    # đánh giá nhanh vài lần reset
    sr = 0
    moves_sum = 0
    for i in range(args.eval_trials):
        plan, ok = eval_once(env, agent, max_agent_steps=args.eval_agent_steps,
                             bfs_depth=args.eval_bfs_depth)
        sr += int(ok)
        moves_sum += len(plan)
        if i == 0:
            names = " ".join(MOVE_NAMES[idx] for idx in plan) if plan else "<empty>"
            print(f"[Eval#{i}] solved={ok} len={len(plan)} plan={names}")
    avg_len = moves_sum / max(1, args.eval_trials)
    print(f"[Eval] SR={sr}/{args.eval_trials}  Avg plan length={avg_len:.2f}")

if __name__ == "__main__":
    main()



