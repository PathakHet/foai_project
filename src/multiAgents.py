import csv
import json
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from game import Agent, Directions
from util import Stack, manhattanDistance


class BaseStrategy:
    name = "base"

    def select_action(self, q_values, legal_moves):
        raise NotImplementedError

    def describe(self):
        return self.name


class GreedyStrategy(BaseStrategy):
    name = "greedy"

    def select_action(self, q_values, legal_moves):
        if not q_values:
            return 0
        max_q = max(q_values)
        best = [i for i, value in enumerate(q_values) if value == max_q]
        return random.choice(best)


class EpsilonGreedyStrategy(BaseStrategy):
    name = "epsilon_greedy"

    def __init__(self, epsilon: float = 0.1):
        self.set_epsilon(epsilon)

    def select_action(self, q_values, legal_moves):
        if not q_values:
            return 0
        if random.random() < self.epsilon:
            return random.randrange(len(legal_moves))
        return GreedyStrategy().select_action(q_values, legal_moves)

    def set_epsilon(self, epsilon: float):
        self.epsilon = max(0.0, min(1.0, float(epsilon)))

    def describe(self):
        return f"{self.name}(epsilon={self.epsilon:.3f})"


class SoftmaxStrategy(BaseStrategy):
    name = "softmax"

    def __init__(self, temperature: float = 1.0):
        self.temperature = max(1e-6, float(temperature))

    def select_action(self, q_values, legal_moves):
        if not q_values:
            return 0
        scaled = [q / self.temperature for q in q_values]
        max_q = max(scaled)
        exp_values = [np.exp(q - max_q) for q in scaled]
        total = float(np.sum(exp_values))
        if total <= 0.0:
            return GreedyStrategy().select_action(q_values, legal_moves)
        threshold = random.random() * total
        cumulative = 0.0
        for idx, weight in enumerate(exp_values):
            cumulative += weight
            if cumulative >= threshold:
                return idx
        return len(legal_moves) - 1

    def describe(self):
        return f"{self.name}(temperature={self.temperature:.3f})"


def build_strategy(name: str, **kwargs) -> BaseStrategy:
    normalized = (name or "greedy").lower()
    if normalized in ("epsilon", "epsilon_greedy"):
        return EpsilonGreedyStrategy(epsilon=kwargs.get("epsilon"))
    if normalized in ("softmax", "boltzmann"):
        return SoftmaxStrategy(temperature=kwargs.get("temperature"))
    if normalized in ("greedy", "deterministic"):
        return GreedyStrategy()
    raise ValueError(f"Unknown strategy '{name}'")


class AgentTracker:
    def __init__(
        self,
        output_dir: Path,
        experiment_name: str,
        agent_label: str,
        mode: str,
        config: dict,
        num_features: int = 6,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.agent_label = agent_label
        self.mode = mode
        self.config = dict(config)
        self.config.update({"experiment": experiment_name, "mode": mode, "agent": agent_label})
        self.num_features = num_features

        self.csv_path = self.output_dir / f"{agent_label}_{mode}_logs.csv"
        self.visit_path = self.output_dir / f"{agent_label}_{mode}_state_visits.npy"
        self.metadata_path = self.output_dir / f"{agent_label}_{mode}_config.json"

        if not self.metadata_path.exists():
            self.metadata_path.write_text(
                json.dumps(self.config, indent=2),
                encoding="utf-8",
            )

        self.csv_file = self.csv_path.open("a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.csv_file)
        if self.csv_path.stat().st_size == 0:
            header = [
                "experiment",
                "mode",
                "agent",
                "episode",
                "layout",
                "numGhosts",
                "trainingEpisodes",
                "win",
                "score",
                "reward",
                "steps",
                "duration_sec",
                "alpha",
                "gamma",
                "epsilon",
                "strategy",
                "timestamp",
            ]
            header.extend([f"w{i+1}" for i in range(self.num_features)])
            self.writer.writerow(header)

        self.aggregate_visits: Optional[np.ndarray] = None
        self.current_visits: Optional[np.ndarray] = None
        self.layout_shape: Optional[Tuple[int, int]] = None
        self.current_episode_index: Optional[int] = None
        self.episode_start_time: Optional[float] = None
        self.current_steps: int = 0
        self.current_reward: float = 0.0

    def register_layout(self, layout):
        if self.aggregate_visits is None:
            self.layout_shape = (layout.width, layout.height)
        if self.aggregate_visits is None:
            if self.visit_path.exists():
                try:
                    stored = np.load(self.visit_path)
                    if stored.shape == self.layout_shape:
                        self.aggregate_visits = stored
                    else:
                        self.aggregate_visits = np.zeros(self.layout_shape, dtype=np.float64)
                except Exception:
                    self.aggregate_visits = np.zeros(self.layout_shape, dtype=np.float64)
            else:
                self.aggregate_visits = np.zeros(self.layout_shape, dtype=np.float64)

    def start_episode(self, episode_index: int, start_position: Tuple[int, int]):
        self.current_episode_index = episode_index
        self.episode_start_time = time.time()
        if self.aggregate_visits is None:
            raise RuntimeError("Layout must be registered before starting episodes.")
        self.current_visits = np.zeros_like(self.aggregate_visits)
        self.current_steps = 0
        self.current_reward = 0.0
        self.record_state_visit(start_position)

    def record_step(self, position: Tuple[int, int], reward: float):
        if self.current_visits is None:
            return
        self.current_steps += 1
        self.current_reward += reward
        self.record_state_visit(position)

    def record_state_visit(self, position: Tuple[int, int]):
        if self.current_visits is None:
            return
        x = int(position[0])
        y = int(position[1])
        if 0 <= x < self.current_visits.shape[0] and 0 <= y < self.current_visits.shape[1]:
            self.current_visits[x, y] += 1
            self.aggregate_visits[x, y] += 1

    def complete_episode(self, final_state, weights: np.ndarray):
        if self.current_episode_index is None:
            return
        duration = time.time() - self.episode_start_time
        win_flag = 1 if final_state.isWin() else 0
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        row = [
            self.config.get("experiment"),
            self.mode,
            self.agent_label,
            self.current_episode_index,
            self.config.get("layout"),
            self.config.get("numGhosts"),
            self.config.get("episodes"),
            win_flag,
            final_state.getScore(),
            self.current_reward,
            self.current_steps,
            duration,
            self.config.get("alpha"),
            self.config.get("gamma"),
            self.config.get("epsilon"),
            self.config.get("strategy"),
            timestamp,
        ]
        weights = np.array(np.atleast_1d(weights), dtype=np.float64)
        if len(weights) < self.num_features:
            padded = np.zeros(self.num_features)
            padded[: len(weights)] = weights
            weights = padded
        row.extend([f"{w:.6f}" for w in weights])
        self.writer.writerow(row)
        self.csv_file.flush()

    def save_state_visits(self):
        if self.aggregate_visits is not None:
            np.save(self.visit_path, self.aggregate_visits)

    def close(self):
        self.save_state_visits()
        self.csv_file.close()


class Node:
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __hash__(self):
        return hash(self.state)

    def __eq__(self, other):
        return self.state == other.state

    def __ne__(self, other):
        return self.state != other.state


class ReflexAgent(Agent):
    def __init__(self, weights="weights.csv", tracker: AgentTracker = None, num_features: int = 11):
        self.weights_path = weights
        self.num_features = int(num_features)
        self._ensure_weights_file()
        self.tracker = tracker
        self._episode_index = 0
        self._timer_start = None

    def goalTest(self, gs, pos, flag):
        if flag == 0:
            return gs.hasFood(pos[0], pos[1])
        if flag == 1:
            gpos = gs.getGhostPositions()
            return any(gp == pos for gp in gpos)
        return False

    def DLS(self, currentNode, stack, explored, layer, limit, found, flag):
        explored.append(currentNode)
        if self.goalTest(currentNode.parent.state, currentNode.state.getPacmanPosition(), flag):
            stack.push(currentNode)
            return stack, explored, True
        if layer == limit:
            return stack, explored, False
        stack.push(currentNode)
        actions = currentNode.state.getLegalActions()
        for a in actions:
            newState = currentNode.state.generatePacmanSuccessor(a)
            newNode = Node(newState, currentNode, a, 1)
            if newNode in explored:
                continue
            stack, explored, found = self.DLS(newNode, stack, explored, layer + 1, limit, found, flag)
            if found:
                return stack, explored, True
        stack.pop()
        return stack, explored, False

    def IDS(self, sgs, limit, flag):
        found = False
        current_limit = 0
        while not found and current_limit <= limit:
            current_limit += 1
            startNode = Node(sgs, None, None, 0)
            startNode.parent = startNode
            stack = Stack()
            explored = []
            stack, explored, found = self.DLS(startNode, stack, explored, 1, current_limit, False, flag)

        actions = []
        while not stack.isEmpty():
            node = stack.pop()
            actions.append(node.action)

        if not actions:
            return actions, found

        actions.reverse()
        actions.pop(0)

        return actions, found

    def _ensure_weights_file(self):
        try:
            np.loadtxt(self.weights_path, delimiter=",")
        except Exception:
            initial = np.zeros(self.num_features)
            np.savetxt(self.weights_path, initial, delimiter=",", fmt="%4.8f")

    def _load_weights(self):
        data = np.loadtxt(self.weights_path, delimiter=",")
        weights = np.array(np.atleast_1d(data), dtype=float)
        if weights.shape[0] < self.num_features:
            padded = np.zeros(self.num_features)
            padded[: weights.shape[0]] = weights
            weights = padded
        elif weights.shape[0] > self.num_features:
            weights = weights[: self.num_features]
        return weights

    def _save_weights(self, weights):
        np.savetxt(self.weights_path, weights, delimiter=",", fmt="%4.8f")

    def set_episode_index(self, index: int):
        self._episode_index = index

    def set_tracker(self, tracker: AgentTracker):
        self.tracker = tracker

    def registerInitialState(self, state):
        self._timer_start = time.time()
        if self.tracker is not None:
            self.tracker.register_layout(state.data.layout)
            self.tracker.start_episode(self._episode_index, state.getPacmanPosition())

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        if not legalMoves:
            return Directions.STOP

        weights = self._load_weights()
        scores = []
        for action in legalMoves:
            successorGameState = gameState.generatePacmanSuccessor(action)
            features = self.build_feature_vector(gameState, successorGameState)
            scores.append(float(np.dot(weights, np.transpose(features))))

        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        return legalMoves[chosenIndex]

    def final(self, state):
        if self.tracker is not None:
            self.tracker.complete_episode(state, self._load_weights())

    def CalcGhostPos(self, cgs, actions):
        for a in actions:
            cgs = cgs.generatePacmanSuccessor(a)
        return cgs.getPacmanPosition()

    def findAllGhosts(self, cgs):
        f1 = 0
        f2 = 0
        f3 = 0
        f4 = 0
        actions, found = self.IDS(cgs, 3, 1)
        if not found:
            return f1, f2, f3, f4
        ghosts = cgs.getGhostStates()
        ghostPos = self.CalcGhostPos(cgs, actions)
        ghost = None
        for g in ghosts:
            if ghostPos == g.configuration.pos:
                ghost = g
                break

        if ghost is None:
            return f1, f2, f3, f4

        if ghost.scaredTimer > 0:
            if len(actions) <= 1:
                f3 = 1
            if len(actions) == 2:
                f4 = 1
        else:
            if len(actions) <= 1:
                f1 = 1
            if len(actions) == 2:
                f2 = 1

        return f1, f2, f3, f4

    def getFeatureFive(self, cgs, sgs):
        return 1 if self.goalTest(cgs, sgs.getPacmanPosition(), 0) else 0

    def getFeatureSix(self, cgs):
        food = cgs.getFood()
        pacPos = cgs.getPacmanPosition()
        dist = []
        x_size = food.width
        y_size = food.height
        for x in range(0, x_size):
            for y in range(0, y_size):
                if food[x][y] is True:
                    dist.append(manhattanDistance(pacPos, (x, y)))
        if not dist:
            return 0
        closestFood = min(dist)
        return 1 / closestFood if closestFood > 0 else 1.0

    def build_feature_vector(self, currentGameState, successorGameState):
        f1, f2, f3, f4 = self.findAllGhosts(successorGameState)
        f5 = self.getFeatureFive(currentGameState, successorGameState)
        f6 = self.getFeatureSix(successorGameState)
        bias = 1.0
        total_food = max(1, currentGameState.getNumFood())
        food_remaining = successorGameState.getNumFood() / float(total_food)
        cap_positions = successorGameState.getCapsules()
        pacPos = successorGameState.getPacmanPosition()
        if cap_positions:
            cap_dists = [manhattanDistance(pacPos, c) for c in cap_positions]
            inv_cap_dist = 1.0 / max(1.0, min(cap_dists))
        else:
            inv_cap_dist = 0.0
        ghosts = successorGameState.getGhostStates()
        active_dists = []
        scared_dists = []
        for g in ghosts:
            d = manhattanDistance(pacPos, g.getPosition())
            if g.scaredTimer > 0:
                scared_dists.append(d)
            else:
                active_dists.append(d)
        min_active = 1.0 / max(1.0, min(active_dists)) if active_dists else 0.0
        min_scared = 1.0 / max(1.0, min(scared_dists)) if scared_dists else 0.0

        return np.array(
            [
                f1,
                f2,
                f3,
                f4,
                f5,
                f6,
                bias,
                food_remaining,
                inv_cap_dist,
                min_active,
                min_scared,
            ],
            dtype=float,
        )

    def getReward(self, cgs, sgs):
        pacPos = sgs.getPacmanPosition()
        gpos = cgs.getGhostPositions()
        ghosts = cgs.getGhostStates()
        active_ghost = ghosts[0] if ghosts else None
        for pos in gpos:
            if active_ghost is not None:
                if pacPos == pos and active_ghost.scaredTimer == 0:
                    return -250
                if pacPos == pos and active_ghost.scaredTimer > 1:
                    return 100
        if cgs.hasFood(pacPos[0], pacPos[1]):
            if cgs.getNumFood() <= 1:
                return 250
            return 1
        return -1


class LearningReflexAgent(ReflexAgent):
    def __init__(
        self,
        strategy="greedy",
        epsilon=0.1,
        temperature=1.0,
        alpha=0.00001,
        gamma=0.9,
        weights="weights.csv",
        tracker: AgentTracker = None,
        learning: bool = True,
        **kwargs,
    ):
        feature_count = int(kwargs.pop("num_features", 11))
        super().__init__(weights=weights, tracker=tracker, num_features=feature_count)
        self.strategy = build_strategy(strategy, epsilon=epsilon, temperature=temperature)
        self.strategy_name = strategy
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.learning = learning
        self.total_episodes = int(kwargs.get("total_episodes", 0))
        self.epsilon_min = float(kwargs.get("epsilon_min", epsilon))
        self.epsilon_decay_steps = int(kwargs.get("epsilon_decay_steps", self.total_episodes))
        self.initial_epsilon = epsilon
        self._last_features = None
        self._last_score = 0.0
        self._last_food = None
        self._last_capsules = None
        self._initial_food = None

    def registerInitialState(self, state):
        self._last_features = None
        self._last_score = state.getScore()
        self._last_food = state.getNumFood()
        self._last_capsules = len(state.getCapsules())
        self._initial_food = max(1, state.getNumFood())
        super().registerInitialState(state)

    def set_episode_index(self, index: int):
        super().set_episode_index(index)
        if hasattr(self.strategy, "set_epsilon") and self.epsilon_decay_steps > 0:
            frac = min(1.0, max(0.0, float(index) / float(self.epsilon_decay_steps)))
            decayed = self.initial_epsilon - frac * (self.initial_epsilon - self.epsilon_min)
            self.strategy.set_epsilon(decayed)

    def _evaluate_actions(self, gameState, weights):
        q_values = []
        successor_cache = []
        for action in gameState.getLegalActions():
            successorState = gameState.generatePacmanSuccessor(action)
            features = self.build_feature_vector(gameState, successorState)
            q_values.append(float(np.dot(weights, np.transpose(features))))
            successor_cache.append((successorState, features))
        return q_values, successor_cache

    def _compute_step_reward(self, next_state):
        score_delta = float(next_state.getScore() - self._last_score)
        food_now = next_state.getNumFood()
        capsules_now = len(next_state.getCapsules())
        food_eaten = max(0, (self._last_food or 0) - food_now)
        capsules_eaten = max(0, (self._last_capsules or 0) - capsules_now)
        progress_bonus = food_eaten * 5.0 + capsules_eaten * 20.0
        step_penalty = -0.2
        reward = score_delta + progress_bonus + step_penalty
        return max(-3000.0, min(3000.0, reward))

    def _update_from_transition(self, reward, next_state, weights, next_q=None):
        if not self.learning or self._last_features is None:
            return weights
        if next_q is None:
            next_q = self._estimate_future_q(next_state, weights)
        prediction = float(np.dot(weights, np.transpose(self._last_features)))
        target = reward + self.gamma * next_q
        new_weights = weights + self.alpha * (target - prediction) * self._last_features
        self._save_weights(new_weights)
        return new_weights

    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        if not legalMoves:
            return Directions.STOP

        weights = self._load_weights()
        if self._last_features is not None:
            reward = self._compute_step_reward(gameState)
            if self.tracker is not None:
                self.tracker.record_step(gameState.getPacmanPosition(), reward)
            weights = self._update_from_transition(reward, gameState, weights)

        q_values, successor_cache = self._evaluate_actions(gameState, weights)
        action_index = self.strategy.select_action(q_values, legalMoves)
        chosen_action = legalMoves[action_index]
        _, chosen_features = successor_cache[action_index]

        self._last_features = chosen_features
        self._last_score = gameState.getScore()
        self._last_food = gameState.getNumFood()
        self._last_capsules = len(gameState.getCapsules())
        return chosen_action

    def _estimate_future_q(self, state, weights):
        legalMoves = state.getLegalActions()
        if not legalMoves:
            return 0.0
        q_vals = []
        for action in legalMoves:
            successor = state.generatePacmanSuccessor(action)
            features = self.build_feature_vector(state, successor)
            q_vals.append(float(np.dot(weights, np.transpose(features))))
        return max(q_vals) if q_vals else 0.0

    def final(self, state):
        weights = self._load_weights()
        if self._last_features is not None:
            reward = self._compute_step_reward(state) + self._terminal_shaping(state)
            if self.tracker is not None:
                self.tracker.record_step(state.getPacmanPosition(), reward)
            weights = self._update_from_transition(reward, state, weights, next_q=0.0)
            self._last_features = None
        if self.tracker is not None:
            self.tracker.complete_episode(state, weights)

    def _terminal_shaping(self, state):
        food_left = state.getNumFood()
        food_fraction = food_left / float(self._initial_food or max(1, food_left))
        if state.isWin():
            return 2000.0 + 500.0 * (1.0 - food_fraction)
        if state.isLose():
            return -2000.0 - 500.0 * (1.0 - food_fraction)
        return -800.0 * food_fraction


class SarsaReflexAgent(LearningReflexAgent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions()
        if not legalMoves:
            return Directions.STOP

        weights = self._load_weights()
        q_values, successor_cache = self._evaluate_actions(gameState, weights)

        action_index = self.strategy.select_action(q_values, legalMoves)
        chosen_action = legalMoves[action_index]
        _, chosen_features = successor_cache[action_index]

        if self._last_features is not None:
            reward = self._compute_step_reward(gameState)
            if self.tracker is not None:
                self.tracker.record_step(gameState.getPacmanPosition(), reward)
            weights = self._update_from_transition(reward, gameState, weights, next_q=q_values[action_index])

        self._last_features = chosen_features
        self._last_score = gameState.getScore()
        self._last_food = gameState.getNumFood()
        self._last_capsules = len(gameState.getCapsules())
        return chosen_action

    def _update_from_transition(self, reward, next_state, weights, next_q=None):
        if not self.learning or self._last_features is None:
            return weights
        if next_q is None:
            next_q = 0.0
        prediction = float(np.dot(weights, np.transpose(self._last_features)))
        target = reward + self.gamma * next_q
        new_weights = weights + self.alpha * (target - prediction) * self._last_features
        self._save_weights(new_weights)
        return new_weights
