# Pacman RL Experiments

Lightweight setup for comparing approximate Q-learning and SARSA agents on the classic Pacman grid world. The remaining code focuses on four things: running experiments, evaluating trained weights, playing the game yourself, and visualizing runs.

## What matters
- `src/multiAgents.py`: feature extractor, RL agents (Q-learning + SARSA), and the lightweight tracker used for logging.
- `src/run_experiments.py`: batch runner that trains, evaluates, and plots results for each experiment.
- Core game + rendering: `src/pacman.py`, `src/game.py`, `src/ghostAgents.py`, `src/keyboardAgents.py`, `src/graphicsDisplay.py`, `src/textDisplay.py`, `src/layout.py`, `src/layouts/`.

## Quick start
1. From the repo root: `cd ai_pacman/src`.
2. Install deps if needed: `pip install numpy pandas matplotlib seaborn`.
3. Run the default experiments (400 episodes, 1–2 ghosts, mediumClassic layout):
   - `python run_experiments.py`
4. Evaluate saved weights without retraining (use the experiment folder under `results/`):
   - `python run_experiments.py --eval-only ep400_ghosts2 --eval-games 20`
5. Watch a trained agent (quiet/text modes avoid the Tk window):
   - `python pacman.py -p LearningReflexAgent -a weights=results/ep400_ghosts2/q_learning/q_learning_weights.csv -q -n 5`
6. Play yourself:
   - Graphics: `python pacman.py -p KeyboardAgent -l mediumClassic`
   - Text mode: `python pacman.py -p KeyboardAgent -l mediumClassic --textGraphics`

## Experiment outputs
- Per-agent logs: `results/<exp>/<agent>/<agent>_train_logs.csv` and `_test_logs.csv`
- Weights: `results/<exp>/<agent>/<agent>_weights.csv`
- State visit heatmaps: `results/<exp>/<agent>/<agent>_train_state_visits.npy`
- Plots: `results/<exp>/plots/*.png` and evaluation plot `eval_win_rates.png`

## Notes
- Ghosts use deterministic movement to keep runs repeatable.
- The feature vector tracks ghost proximity (active + scared), on-food flag, inverse food distance, a bias term, food remaining fraction, inverse capsule distance, and min distances to active/scared ghosts.
- Use `--episodes`, `--ghosts`, `--eval-games`, `--max-moves`, and `--resume` flags on `run_experiments.py` to tweak runs for the professor’s review.
