import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

import layout
import pacman
import textDisplay
from ghostAgents import DeterministicGhost
from multiAgents import AgentTracker, LearningReflexAgent, SarsaReflexAgent
from pacman import ClassicGameRules

RESULTS_ROOT = Path("results")
DEFAULT_LAYOUT = "mediumClassic"
DEFAULT_TIMEOUT = 30
DEFAULT_MAX_MOVES = 1500
DEFAULT_EVAL_EPISODES = 20
DEFAULT_TRAINING_EPISODES = [400]
DEFAULT_GHOST_COUNTS = [1, 2]

AGENT_SPECS = {
    "q_learning": LearningReflexAgent,
    "sarsa": SarsaReflexAgent,
}
AGENT_ORDER = list(AGENT_SPECS.keys())

TRAINING_PARAMS = {
    "strategy": "epsilon_greedy",
    "epsilon": 0.3,
    "epsilon_min": 0.05,
    "alpha": 0.006,
    "gamma": 0.99,
    "num_features": 11,
}

EVAL_PARAMS = {
    "strategy": "greedy",
    "epsilon": 0.0,
    "alpha": 0.0,
    "gamma": 0.0,
}

pacman.TRACK_EXPLORED_STATES = False
random.seed(0)
np.random.seed(0)


def log(msg: str):
    print(msg, flush=True)


def build_experiments(layout_name: str, episodes: List[int], ghost_counts: List[int], eval_eps: int, max_moves: int):
    runs = []
    for ep in episodes:
        for ghosts in ghost_counts:
            runs.append(
                {
                    "name": f"ep{ep}_ghosts{ghosts}",
                    "layout": layout_name,
                    "episodes": ep,
                    "numGhosts": ghosts,
                    "eval_episodes": eval_eps,
                    "max_moves": max_moves,
                }
            )
    return runs


def ensure_experiment_metadata(exp_dir: Path, config: Dict):
    exp_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = exp_dir / "experiment_config.json"
    metadata = dict(config)
    metadata["agent_types"] = AGENT_ORDER
    metadata["training_params"] = TRAINING_PARAMS
    metadata["evaluation_params"] = EVAL_PARAMS
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def reset_agent_outputs(agent_dir: Path, agent_name: str, mode: str):
    for suffix in ("logs.csv", "state_visits.npy", "config.json"):
        path = agent_dir / f"{agent_name}_{mode}_{suffix}"
        if path.exists():
            path.unlink()


def create_tracker(exp_dir: Path, agent_name: str, mode: str, config: Dict) -> AgentTracker:
    agent_dir = exp_dir / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)
    tracker_config = dict(TRAINING_PARAMS if mode == "train" else EVAL_PARAMS)
    tracker_config.update(
        {
            "layout": config["layout"],
            "numGhosts": config["numGhosts"],
            "episodes": config["episodes"] if mode == "train" else config["eval_episodes"],
            "strategy": tracker_config.get("strategy"),
            "epsilon": tracker_config.get("epsilon"),
            "alpha": tracker_config.get("alpha"),
            "gamma": tracker_config.get("gamma"),
        }
    )
    return AgentTracker(
        output_dir=agent_dir,
        experiment_name=config["name"],
        agent_label=agent_name,
        mode=mode,
        config=tracker_config,
        num_features=TRAINING_PARAMS.get("num_features", 11),
    )


def next_episode_index(exp_dir: Path, agent_name: str, mode: str) -> int:
    csv_path = exp_dir / agent_name / f"{agent_name}_{mode}_logs.csv"
    if not csv_path.exists():
        return 0
    last_episode = -1
    with csv_path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            try:
                ep = int(row.get("episode", -1))
                if ep > last_episode:
                    last_episode = ep
            except (TypeError, ValueError):
                continue
    return last_episode + 1 if last_episode >= 0 else 0


def run_single_episode(layout_name: str, agent, num_ghosts: int, max_moves: int, quiet: bool = True):
    if hasattr(pacman, "GameState"):
        pacman.GameState.explored.clear()

    layout_obj = layout.getLayout(layout_name)
    ghosts = [DeterministicGhost(i + 1) for i in range(num_ghosts)]
    rules = ClassicGameRules(DEFAULT_TIMEOUT, max_moves=max_moves)
    display = textDisplay.NullGraphics()
    game = rules.newGame(layout_obj, agent, ghosts, display, quiet, catchExceptions=False)
    game.run()
    return game


def train_agent(agent_name: str, config: Dict, exp_dir: Path, resume: bool):
    weights_path = exp_dir / agent_name / f"{agent_name}_weights.csv"
    agent_dir = exp_dir / agent_name
    if not resume:
        reset_agent_outputs(agent_dir, agent_name, "train")
        if weights_path.exists():
            weights_path.unlink()
    start_episode = next_episode_index(exp_dir, agent_name, "train") if resume else 0
    if start_episode >= config["episodes"]:
        log(f"[{config['name']}:{agent_name}] Training already complete.")
        return
    log(f"[{config['name']}:{agent_name}] Train episodes={config['episodes']} ghosts={config['numGhosts']} start={start_episode}")
    tracker = create_tracker(exp_dir, agent_name, mode="train", config=config)
    agent_class = AGENT_SPECS[agent_name]
    agent_params = dict(TRAINING_PARAMS)
    agent_params.update(
        {
            "total_episodes": config["episodes"],
            "epsilon_decay_steps": config["episodes"] * 3,
            "num_features": TRAINING_PARAMS.get("num_features", 11),
        }
    )
    agent = agent_class(
        weights=str(weights_path),
        tracker=tracker,
        learning=True,
        **agent_params,
    )
    for episode in range(start_episode, config["episodes"]):
        log(f"[{config['name']}:{agent_name}] Training ep {episode + 1}/{config['episodes']}")
        agent.set_episode_index(episode)
        game = run_single_episode(config["layout"], agent, config["numGhosts"], config["max_moves"])
        final_state = game.state
        outcome = "WIN" if final_state.isWin() else "LOSS"
        log(
            f"[{config['name']}:{agent_name}] Done outcome={outcome} "
            f"score={final_state.getScore():.1f} moves={len(game.moveHistory)}"
        )
    tracker.close()
    log(f"[{config['name']}:{agent_name}] Training complete.")


def evaluate_agent(agent_name: str, config: Dict, exp_dir: Path, resume: bool):
    weights_path = exp_dir / agent_name / f"{agent_name}_weights.csv"
    agent_dir = exp_dir / agent_name
    if not resume:
        reset_agent_outputs(agent_dir, agent_name, "test")
    start_episode = next_episode_index(exp_dir, agent_name, "test") if resume else 0
    if start_episode >= config["eval_episodes"]:
        log(f"[{config['name']}:{agent_name}] Evaluation already complete.")
        return
    tracker = create_tracker(exp_dir, agent_name, mode="test", config=config)
    agent_class = AGENT_SPECS[agent_name]
    agent = agent_class(
        weights=str(weights_path),
        tracker=tracker,
        learning=False,
        num_features=TRAINING_PARAMS.get("num_features", 11),
        **EVAL_PARAMS,
    )
    for episode in range(start_episode, config["eval_episodes"]):
        log(f"[{config['name']}:{agent_name}] Eval ep {episode + 1}/{config['eval_episodes']}")
        agent.set_episode_index(episode)
        game = run_single_episode(config["layout"], agent, config["numGhosts"], config["max_moves"])
        final_state = game.state
        outcome = "WIN" if final_state.isWin() else "LOSS"
        log(
            f"[{config['name']}:{agent_name}] Eval outcome={outcome} "
            f"score={final_state.getScore():.1f} moves={len(game.moveHistory)}"
        )
    tracker.close()
    log(f"[{config['name']}:{agent_name}] Evaluation complete.")


def run_experiment(config: Dict, skip_plots: bool, resume: bool):
    log(f"=== {config['name']} layout={config['layout']} ghosts={config['numGhosts']} train_eps={config['episodes']} ===")
    exp_dir = RESULTS_ROOT / config["name"]
    ensure_experiment_metadata(exp_dir, config)
    for agent_name in AGENT_ORDER:
        train_agent(agent_name, config, exp_dir, resume=resume)
        evaluate_agent(agent_name, config, exp_dir, resume=resume)
    if not skip_plots:
        generate_all_plots(exp_dir)
    log(f"=== Finished {config['name']} ===")


def evaluate_saved_experiment(exp_name: str, games: int, layout_override: str, ghosts_override: int, max_moves: int):
    exp_dir = RESULTS_ROOT / exp_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
    meta_path = exp_dir / "experiment_config.json"
    layout_name = layout_override
    ghost_count = ghosts_override
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        layout_name = layout_name or metadata.get("layout", DEFAULT_LAYOUT)
        ghost_count = ghost_count or metadata.get("numGhosts", DEFAULT_GHOST_COUNTS[0])
    if layout_name is None:
        layout_name = DEFAULT_LAYOUT
    if ghost_count is None:
        ghost_count = DEFAULT_GHOST_COUNTS[0]

    outcomes = {}
    for agent_name, agent_class in AGENT_SPECS.items():
        weights_path = exp_dir / agent_name / f"{agent_name}_weights.csv"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing weights for {agent_name}: {weights_path}")
        agent = agent_class(
            weights=str(weights_path),
            tracker=None,
            learning=False,
            num_features=TRAINING_PARAMS.get("num_features", 11),
            **EVAL_PARAMS,
        )
        results = {"wins": 0, "scores": [], "moves": [], "wins_sequence": []}
        for game_idx in range(games):
            win, score, moves = run_eval_game(layout_name, agent, ghost_count, max_moves)
            results["wins"] += int(win)
            results["wins_sequence"].append(1 if win else 0)
            results["scores"].append(score)
            results["moves"].append(moves)
            log(
                f"[{exp_name}:{agent_name}] Game {game_idx + 1}/{games} "
                f"{'WIN' if win else 'LOSS'} score={score} moves={moves}"
            )
        outcomes[agent_name] = results
    plot_eval_summary(exp_dir, games, outcomes)
    for agent, result in outcomes.items():
        log(f"[{exp_name}:{agent}] wins {result['wins']} / {games}")


def run_eval_game(layout_name: str, agent, num_ghosts: int, max_moves: int):
    if hasattr(pacman, "GameState"):
        pacman.GameState.explored.clear()

    layout_obj = layout.getLayout(layout_name)
    ghosts = [DeterministicGhost(i + 1) for i in range(num_ghosts)]
    rules = ClassicGameRules(DEFAULT_TIMEOUT, max_moves=max_moves)
    display = textDisplay.NullGraphics()
    game = rules.newGame(layout_obj, agent, ghosts, display, quiet=True, catchExceptions=False)
    game.run()
    final_state = game.state
    return final_state.isWin(), final_state.getScore(), len(game.moveHistory)


def plot_eval_summary(exp_dir: Path, game_count: int, outcomes: Dict[str, Dict]):
    import matplotlib.pyplot as plt

    agents = list(outcomes.keys())
    wins = [outcomes[a]["wins"] for a in agents]

    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.bar(agents, wins, color=["tab:blue", "tab:orange"])
    plt.title(f"Win counts over {game_count} games")
    plt.ylabel("Wins")
    plt.ylim(0, game_count)

    plt.subplot(2, 1, 2)
    for agent in agents:
        cumulative = []
        total = 0
        for w in outcomes[agent]["wins_sequence"]:
            total += w
            cumulative.append(total)
        plt.plot(range(1, game_count + 1), cumulative, marker="o", label=agent)
    plt.xlabel("Game number")
    plt.ylabel("Cumulative wins")
    plt.ylim(0, game_count)
    plt.legend()
    plt.tight_layout()
    out_path = exp_dir / "eval_win_rates.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
    log(f"Saved eval plot to {out_path}")


def generate_all_plots(exp_dir: Path):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    def _load_agent_logs(agent: str, mode: str = "train") -> pd.DataFrame:
        log_path = exp_dir / agent / f"{agent}_{mode}_logs.csv"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")
        return pd.read_csv(log_path)

    def _save_plot(filename: str):
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=200)
        plt.close()

    plt.figure(figsize=(10, 5))
    for agent in AGENT_ORDER:
        df = _load_agent_logs(agent)
        df = df.sort_values("episode")
        rolling = df["win"].rolling(window=50, min_periods=1).mean()
        cumulative = df["win"].expanding().mean()
        plt.plot(df["episode"], rolling, label=f"{agent} rolling (w=50)")
        plt.plot(df["episode"], cumulative, linestyle="--", alpha=0.7, label=f"{agent} cumulative")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate")
    plt.title("Win Rate vs Training Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("win_rate.png")

    plt.figure(figsize=(10, 5))
    for agent in AGENT_ORDER:
        df = _load_agent_logs(agent)
        plt.plot(df["episode"], df["score"], label=agent)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward / Score")
    plt.title("Average Reward (Score) per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("average_reward.png")

    plt.figure(figsize=(10, 5))
    for agent in AGENT_ORDER:
        df = _load_agent_logs(agent)
        df = df.sort_values("episode")
        rolling = df["score"].rolling(window=50, min_periods=1).mean()
        cumulative = df["score"].expanding().mean()
        plt.plot(df["episode"], rolling, label=f"{agent} rolling (w=50)")
        plt.plot(df["episode"], cumulative, linestyle="--", alpha=0.7, label=f"{agent} cumulative")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score Trend vs Training Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("score_trend.png")

    plt.figure(figsize=(10, 5))
    for agent in AGENT_ORDER:
        df = _load_agent_logs(agent)
        plt.plot(df["episode"], df["steps"], label=agent)
    plt.xlabel("Episode")
    plt.ylabel("Steps Survived")
    plt.title("Steps Survived per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("steps_survived.png")

    df_q = _load_agent_logs("q_learning")
    df_s = _load_agent_logs("sarsa")
    weight_cols = [col for col in df_q.columns if col.startswith("w")]
    rows = 2
    cols = int(np.ceil(len(weight_cols) / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    axes = axes.flatten()
    for idx, col in enumerate(weight_cols):
        axes[idx].plot(df_q["episode"], df_q[col], label="Q-Learning")
        axes[idx].plot(df_s["episode"], df_s[col], label="SARSA")
        axes[idx].set_title(f"{col.upper()} Convergence")
        axes[idx].set_xlabel("Episode")
        axes[idx].set_ylabel("Weight")
        axes[idx].grid(True, alpha=0.3)
        if idx == 0:
            axes[idx].legend()
    plt.tight_layout()
    _save_plot("weight_convergence.png")

    plt.figure(figsize=(10, 5))
    for agent in AGENT_ORDER:
        df = _load_agent_logs(agent)
        plt.plot(df["episode"], df["duration_sec"], label=agent)
    plt.xlabel("Episode")
    plt.ylabel("Duration (seconds)")
    plt.title("Training Time per Episode")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _save_plot("training_time.png")

    visit_maps = {}
    vmax = 0.0
    for agent in AGENT_ORDER:
        npy_path = exp_dir / agent / f"{agent}_train_state_visits.npy"
        visits = np.load(npy_path)
        visit_maps[agent] = visits
        vmax = max(vmax, visits.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, agent in zip(axes, AGENT_ORDER):
        sns.heatmap(
            np.rot90(visit_maps[agent]),
            ax=ax,
            cmap="magma",
            vmin=0,
            vmax=max(vmax, 1),
            cbar=True,
        )
        ax.set_title(f"{agent} State Visits")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    plt.tight_layout()
    _save_plot("state_visit_heatmaps.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate Pacman RL agents.")
    parser.add_argument("--layout", default=None, help="Layout name to load (default: mediumClassic).")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        help="Training episode counts to run (default: 400).",
    )
    parser.add_argument(
        "--ghosts",
        type=int,
        nargs="*",
        default=None,
        help="Ghost counts to test (default: 1 2).",
    )
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_EPISODES, help="Evaluation games per agent.")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES, help="Max moves before forcing a loss.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--resume", action="store_true", help="Resume training/eval instead of resetting logs.")
    parser.add_argument("--eval-only", metavar="EXPERIMENT", help="Skip training and evaluate saved weights in results/<name>.")
    return parser.parse_args()


def main():
    args = parse_args()
    layout_name = args.layout or DEFAULT_LAYOUT
    episode_options = args.episodes or DEFAULT_TRAINING_EPISODES
    ghost_options = args.ghosts or DEFAULT_GHOST_COUNTS

    if args.eval_only:
        evaluate_saved_experiment(
            exp_name=args.eval_only,
            games=args.eval_games,
            layout_override=args.layout,
            ghosts_override=ghost_options[0] if ghost_options else DEFAULT_GHOST_COUNTS[0],
            max_moves=args.max_moves,
        )
        return

    experiments = build_experiments(layout_name, episode_options, ghost_options, args.eval_games, args.max_moves)
    for config in experiments:
        run_experiment(config, skip_plots=args.skip_plots, resume=args.resume)
    log("All experiments completed.")


if __name__ == "__main__":
    main()
