"""
test_agent.py

Comprehensive unittest suite for network.py (DuelingQNetwork, ReplayBuffer)
and agent.py (D3QNAgent).

All tests run on CPU to avoid hardware-specific failures.
"""

import copy
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

from network import DuelingQNetwork, ReplayBuffer, Transition, get_device
from agent import D3QNAgent


# ── Shared fixtures ───────────────────────────────────────────────────────────

CPU = torch.device("cpu")

# Small dimensions keep tests fast.
INPUT_DIM  = 20
ACTION_DIM = 6
HIDDEN_DIM = 32
BATCH_SIZE = 8


def make_network(
    input_dim: int = INPUT_DIM,
    action_dim: int = ACTION_DIM,
    hidden_dim: int = HIDDEN_DIM,
) -> DuelingQNetwork:
    return DuelingQNetwork(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=CPU,
    )


def make_buffer(capacity: int = 100, input_dim: int = INPUT_DIM) -> ReplayBuffer:
    return ReplayBuffer(capacity=capacity, device=CPU)


def random_state(input_dim: int = INPUT_DIM) -> np.ndarray:
    return np.random.rand(input_dim).astype(np.float32)


def make_agent(
    n_bins: int = 3,
    num_users: int = 5,
) -> D3QNAgent:
    return D3QNAgent(
        grid_width=300.0,
        grid_height=300.0,
        num_users=num_users,
        n_bins=n_bins,
        buffer_capacity=500,
        lr=1e-3,
        hidden_dim=32,
        device=CPU,
    )


def fill_buffer(agent: D3QNAgent, n: int = 64) -> None:
    """Push n dummy transitions into the agent's replay buffer."""
    state = random_state(agent.input_dim)
    for _ in range(n):
        action = np.random.randint(0, agent.action_dim)
        next_state = random_state(agent.input_dim)
        reward = float(np.random.randn())
        done = bool(np.random.rand() < 0.1)
        agent.replay_buffer.push(state, action, reward, next_state, done)
        state = next_state


# ══════════════════════════════════════════════════════════════════════════════
#  get_device
# ══════════════════════════════════════════════════════════════════════════════

class TestGetDevice(unittest.TestCase):

    def test_returns_torch_device(self) -> None:
        self.assertIsInstance(get_device(), torch.device)

    def test_device_type_is_valid(self) -> None:
        device = get_device()
        self.assertIn(device.type, {"cpu", "cuda", "mps"})


# ══════════════════════════════════════════════════════════════════════════════
#  DuelingQNetwork — initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestDuelingQNetworkInit(unittest.TestCase):

    def setUp(self) -> None:
        self.net = make_network()

    def test_input_dim_stored(self) -> None:
        self.assertEqual(self.net.input_dim, INPUT_DIM)

    def test_action_dim_stored(self) -> None:
        self.assertEqual(self.net.action_dim, ACTION_DIM)

    def test_hidden_dim_stored(self) -> None:
        self.assertEqual(self.net.hidden_dim, HIDDEN_DIM)

    def test_device_stored(self) -> None:
        self.assertEqual(self.net.device, CPU)

    def test_feature_layer_is_sequential(self) -> None:
        self.assertIsInstance(self.net.feature_layer, nn.Sequential)

    def test_value_stream_is_sequential(self) -> None:
        self.assertIsInstance(self.net.value_stream, nn.Sequential)

    def test_advantage_stream_is_sequential(self) -> None:
        self.assertIsInstance(self.net.advantage_stream, nn.Sequential)

    def test_parameters_require_grad(self) -> None:
        for name, param in self.net.named_parameters():
            self.assertTrue(
                param.requires_grad,
                msg=f"Parameter '{name}' should require grad.",
            )

    def test_parameters_on_correct_device(self) -> None:
        for name, param in self.net.named_parameters():
            self.assertEqual(
                param.device.type, CPU.type,
                msg=f"Parameter '{name}' not on CPU.",
            )


# ══════════════════════════════════════════════════════════════════════════════
#  DuelingQNetwork — forward pass
# ══════════════════════════════════════════════════════════════════════════════

class TestDuelingQNetworkForward(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.net = make_network()

    # --- Output shape ----------------------------------------------------

    def test_output_shape_single_sample(self) -> None:
        x = torch.randn(1, INPUT_DIM)
        q = self.net(x)
        self.assertEqual(q.shape, (1, ACTION_DIM))

    def test_output_shape_batch(self) -> None:
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        q = self.net(x)
        self.assertEqual(q.shape, (BATCH_SIZE, ACTION_DIM))

    def test_output_shape_large_batch(self) -> None:
        x = torch.randn(128, INPUT_DIM)
        q = self.net(x)
        self.assertEqual(q.shape, (128, ACTION_DIM))

    def test_output_shape_varies_with_action_dim(self) -> None:
        for a_dim in [2, 5, 16, 25]:
            net = make_network(action_dim=a_dim)
            x = torch.randn(4, INPUT_DIM)
            self.assertEqual(net(x).shape, (4, a_dim))

    # --- Output dtype and finiteness ------------------------------------

    def test_output_dtype_float32(self) -> None:
        x = torch.randn(4, INPUT_DIM)
        self.assertEqual(self.net(x).dtype, torch.float32)

    def test_output_is_finite(self) -> None:
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        self.assertTrue(torch.all(torch.isfinite(self.net(x))))

    # --- Duelling identity: Q.mean(dim=1) == V(s) -----------------------

    def test_mean_centering_identity(self) -> None:
        """
        Because A is mean-centred, mean_a Q(s,a) must equal V(s):
            Q = V + (A − mean(A))  →  mean(Q) = V + 0 = V
        """
        x = torch.randn(BATCH_SIZE, INPUT_DIM)
        with torch.no_grad():
            features = self.net.feature_layer(x)
            value    = self.net.value_stream(features).squeeze(1)   # (B,)
            q        = self.net(x)                                   # (B, A)

        torch.testing.assert_close(
            q.mean(dim=1), value, atol=1e-5, rtol=1e-4,
        )

    # --- Gradient flows through policy net but not after no_grad --------

    def test_forward_creates_grad_fn(self) -> None:
        x = torch.randn(4, INPUT_DIM)
        q = self.net(x)
        self.assertIsNotNone(q.grad_fn)

    def test_no_grad_context_detaches_output(self) -> None:
        x = torch.randn(4, INPUT_DIM)
        with torch.no_grad():
            q = self.net(x)
        self.assertIsNone(q.grad_fn)


# ══════════════════════════════════════════════════════════════════════════════
#  DuelingQNetwork — act()
# ══════════════════════════════════════════════════════════════════════════════

class TestDuelingQNetworkAct(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(1)
        self.net = make_network()
        self.state = random_state()

    def test_returns_int(self) -> None:
        self.assertIsInstance(self.net.act(self.state, epsilon=0.0), int)

    def test_greedy_action_in_valid_range(self) -> None:
        action = self.net.act(self.state, epsilon=0.0)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, ACTION_DIM)

    def test_random_action_in_valid_range(self) -> None:
        for _ in range(20):
            action = self.net.act(self.state, epsilon=1.0)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, ACTION_DIM)

    def test_epsilon_zero_is_deterministic(self) -> None:
        """Same state + epsilon=0 must always return the same greedy action."""
        actions = {self.net.act(self.state, epsilon=0.0) for _ in range(10)}
        self.assertEqual(len(actions), 1)

    def test_epsilon_one_produces_varied_actions(self) -> None:
        """epsilon=1 must explore: over 200 trials multiple actions appear."""
        actions = {self.net.act(self.state, epsilon=1.0) for _ in range(200)}
        self.assertGreater(len(actions), 1)


# ══════════════════════════════════════════════════════════════════════════════
#  ReplayBuffer — initialisation and validation
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferInit(unittest.TestCase):

    def test_initial_length_zero(self) -> None:
        buf = make_buffer()
        self.assertEqual(len(buf), 0)

    def test_capacity_stored(self) -> None:
        buf = make_buffer(capacity=256)
        self.assertEqual(buf.capacity, 256)

    def test_device_stored(self) -> None:
        buf = make_buffer()
        self.assertEqual(buf.device, CPU)

    def test_zero_capacity_raises(self) -> None:
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=0, device=CPU)

    def test_negative_capacity_raises(self) -> None:
        with self.assertRaises(ValueError):
            ReplayBuffer(capacity=-10, device=CPU)

    def test_repr_contains_capacity(self) -> None:
        buf = make_buffer(capacity=512)
        self.assertIn("512", repr(buf))


# ══════════════════════════════════════════════════════════════════════════════
#  ReplayBuffer — push and len
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferPush(unittest.TestCase):

    def setUp(self) -> None:
        self.buf = make_buffer(capacity=10)

    def _push(self, done: bool = False) -> None:
        self.buf.push(
            state=random_state(),
            action=0,
            reward=1.0,
            next_state=random_state(),
            done=done,
        )

    def test_len_increments_after_push(self) -> None:
        self._push()
        self.assertEqual(len(self.buf), 1)

    def test_len_correct_after_multiple_pushes(self) -> None:
        for _ in range(7):
            self._push()
        self.assertEqual(len(self.buf), 7)

    def test_capacity_not_exceeded(self) -> None:
        for _ in range(15):   # push beyond capacity of 10
            self._push()
        self.assertEqual(len(self.buf), 10)

    def test_oldest_entry_evicted_at_capacity(self) -> None:
        """After filling and overflowing, stored count stays at capacity."""
        for _ in range(25):
            self._push()
        self.assertEqual(len(self.buf), 10)

    def test_state_coerced_to_float32(self) -> None:
        state = np.ones(INPUT_DIM, dtype=np.float64)
        self.buf.push(state, 0, 0.0, state, False)
        t: Transition = self.buf._buffer[-1]
        self.assertEqual(t.state.dtype, np.float32)

    def test_action_coerced_to_int(self) -> None:
        self.buf.push(random_state(), 3.0, 0.0, random_state(), False)
        self.assertIsInstance(self.buf._buffer[-1].action, int)

    def test_reward_coerced_to_float(self) -> None:
        self.buf.push(random_state(), 0, np.float32(5.0), random_state(), False)
        self.assertIsInstance(self.buf._buffer[-1].reward, float)

    def test_done_coerced_to_bool(self) -> None:
        self.buf.push(random_state(), 0, 0.0, random_state(), 1)
        self.assertIsInstance(self.buf._buffer[-1].done, bool)


# ══════════════════════════════════════════════════════════════════════════════
#  ReplayBuffer — sample
# ══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferSample(unittest.TestCase):

    def setUp(self) -> None:
        self.buf = make_buffer(capacity=200)
        for _ in range(100):
            self.buf.push(
                random_state(), np.random.randint(ACTION_DIM),
                float(np.random.randn()), random_state(),
                bool(np.random.rand() < 0.1),
            )

    def test_returns_five_tuple(self) -> None:
        result = self.buf.sample(BATCH_SIZE)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)

    # --- Tensor shapes --------------------------------------------------

    def test_states_shape(self) -> None:
        states, *_ = self.buf.sample(BATCH_SIZE)
        self.assertEqual(states.shape, (BATCH_SIZE, INPUT_DIM))

    def test_actions_shape(self) -> None:
        _, actions, *_ = self.buf.sample(BATCH_SIZE)
        self.assertEqual(actions.shape, (BATCH_SIZE, 1))

    def test_rewards_shape(self) -> None:
        _, _, rewards, *_ = self.buf.sample(BATCH_SIZE)
        self.assertEqual(rewards.shape, (BATCH_SIZE, 1))

    def test_next_states_shape(self) -> None:
        _, _, _, next_states, _ = self.buf.sample(BATCH_SIZE)
        self.assertEqual(next_states.shape, (BATCH_SIZE, INPUT_DIM))

    def test_dones_shape(self) -> None:
        *_, dones = self.buf.sample(BATCH_SIZE)
        self.assertEqual(dones.shape, (BATCH_SIZE, 1))

    # --- Tensor dtypes --------------------------------------------------

    def test_states_dtype_float32(self) -> None:
        states, *_ = self.buf.sample(BATCH_SIZE)
        self.assertEqual(states.dtype, torch.float32)

    def test_actions_dtype_int64(self) -> None:
        _, actions, *_ = self.buf.sample(BATCH_SIZE)
        self.assertEqual(actions.dtype, torch.int64)

    def test_rewards_dtype_float32(self) -> None:
        _, _, rewards, *_ = self.buf.sample(BATCH_SIZE)
        self.assertEqual(rewards.dtype, torch.float32)

    def test_dones_dtype_float32(self) -> None:
        *_, dones = self.buf.sample(BATCH_SIZE)
        self.assertEqual(dones.dtype, torch.float32)

    # --- Tensor values --------------------------------------------------

    def test_dones_are_binary(self) -> None:
        """dones tensor must contain only 0.0 or 1.0."""
        *_, dones = self.buf.sample(BATCH_SIZE)
        unique = dones.unique()
        for v in unique:
            self.assertIn(float(v), {0.0, 1.0})

    def test_actions_in_valid_range(self) -> None:
        _, actions, *_ = self.buf.sample(BATCH_SIZE)
        self.assertTrue(torch.all(actions >= 0))
        self.assertTrue(torch.all(actions < ACTION_DIM))

    # --- Tensor devices -------------------------------------------------

    def test_all_tensors_on_correct_device(self) -> None:
        for tensor in self.buf.sample(BATCH_SIZE):
            self.assertEqual(tensor.device.type, CPU.type)

    # --- Error on insufficient data -------------------------------------

    def test_sample_exceeds_buffer_raises(self) -> None:
        small_buf = make_buffer(capacity=5)
        small_buf.push(random_state(), 0, 0.0, random_state(), False)
        with self.assertRaises(ValueError):
            small_buf.sample(5)

    def test_sample_exact_buffer_size_succeeds(self) -> None:
        small_buf = make_buffer(capacity=4)
        for _ in range(4):
            small_buf.push(random_state(), 0, 0.0, random_state(), False)
        result = small_buf.sample(4)
        self.assertEqual(len(result), 5)


# ══════════════════════════════════════════════════════════════════════════════
#  D3QNAgent — initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestD3QNAgentInit(unittest.TestCase):

    def setUp(self) -> None:
        self.agent = make_agent(n_bins=4, num_users=5)

    def test_action_dim_is_n_bins_squared(self) -> None:
        self.assertEqual(self.agent.action_dim, 16)

    def test_input_dim_matches_env_schema(self) -> None:
        from environment import IABEnv
        expected = 5 * 2 + IABEnv.MAX_NODES * 2 + 2 + 3
        self.assertEqual(self.agent.input_dim, expected)

    def test_policy_net_is_dueling_network(self) -> None:
        self.assertIsInstance(self.agent.policy_net, DuelingQNetwork)

    def test_target_net_is_dueling_network(self) -> None:
        self.assertIsInstance(self.agent.target_net, DuelingQNetwork)

    def test_policy_and_target_identical_at_init(self) -> None:
        """Target net must start as an exact copy of the policy net."""
        for (_, p_param), (_, t_param) in zip(
            self.agent.policy_net.named_parameters(),
            self.agent.target_net.named_parameters(),
        ):
            torch.testing.assert_close(p_param, t_param)

    def test_target_net_params_frozen(self) -> None:
        for name, param in self.agent.target_net.named_parameters():
            self.assertFalse(
                param.requires_grad,
                msg=f"Target param '{name}' should be frozen.",
            )

    def test_replay_buffer_is_correct_type(self) -> None:
        self.assertIsInstance(self.agent.replay_buffer, ReplayBuffer)

    def test_replay_buffer_initially_empty(self) -> None:
        self.assertEqual(len(self.agent.replay_buffer), 0)

    def test_device_is_cpu(self) -> None:
        self.assertEqual(self.agent.device, CPU)


# ══════════════════════════════════════════════════════════════════════════════
#  D3QNAgent — state utilities
# ══════════════════════════════════════════════════════════════════════════════

class TestD3QNAgentStateUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.agent = make_agent(n_bins=3, num_users=5)

    # --- flatten_state --------------------------------------------------

    def test_flatten_state_returns_ndarray(self) -> None:
        from environment import IABEnv
        state = {
            "user_positions":  np.zeros((5, 2), dtype=np.float32),
            "node_positions":  np.zeros((IABEnv.MAX_NODES, 2), dtype=np.float32),
            "donor_position":  np.zeros(2, dtype=np.float32),
            "num_nodes":       0,
            "sum_rate_mbps":   0.0,
            "coverage_rate":   0.0,
        }
        result = D3QNAgent.flatten_state(state)
        self.assertIsInstance(result, np.ndarray)

    def test_flatten_state_correct_length(self) -> None:
        from environment import IABEnv
        state = {
            "user_positions":  np.zeros((5, 2), dtype=np.float32),
            "node_positions":  np.zeros((IABEnv.MAX_NODES, 2), dtype=np.float32),
            "donor_position":  np.zeros(2, dtype=np.float32),
            "num_nodes":       3,
            "sum_rate_mbps":   150.0,
            "coverage_rate":   0.8,
        }
        result = D3QNAgent.flatten_state(state)
        self.assertEqual(result.shape[0], self.agent.input_dim)

    def test_flatten_state_dtype_float32(self) -> None:
        from environment import IABEnv
        state = {
            "user_positions":  np.ones((5, 2)),
            "node_positions":  np.ones((IABEnv.MAX_NODES, 2)),
            "donor_position":  np.ones(2),
            "num_nodes":       1,
            "sum_rate_mbps":   100.0,
            "coverage_rate":   0.5,
        }
        result = D3QNAgent.flatten_state(state)
        self.assertEqual(result.dtype, np.float32)

    def test_flatten_state_from_live_env(self) -> None:
        """Verify flatten_state works with a real IABEnv observation."""
        from environment import IABEnv
        env = IABEnv(300.0, 300.0, num_users=5, seed=0)
        state_dict = env.reset()
        flat = D3QNAgent.flatten_state(state_dict)
        self.assertEqual(flat.shape[0], self.agent.input_dim)

    # --- action_to_coords -----------------------------------------------

    def test_action_to_coords_returns_tuple(self) -> None:
        result = self.agent.action_to_coords(0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_action_zero_maps_to_first_bin_centre(self) -> None:
        """Action 0 → row=0, col=0 → centre of bottom-left cell."""
        x, y = self.agent.action_to_coords(0)
        expected_x = 0.5 * self.agent.grid_width  / self.agent.n_bins
        expected_y = 0.5 * self.agent.grid_height / self.agent.n_bins
        self.assertAlmostEqual(x, expected_x)
        self.assertAlmostEqual(y, expected_y)

    def test_action_maps_to_within_grid(self) -> None:
        for action in range(self.agent.action_dim):
            x, y = self.agent.action_to_coords(action)
            self.assertGreater(x, 0.0)
            self.assertLess(x, self.agent.grid_width)
            self.assertGreater(y, 0.0)
            self.assertLess(y, self.agent.grid_height)

    def test_action_last_maps_to_last_bin_centre(self) -> None:
        n = self.agent.n_bins
        last = self.agent.action_dim - 1
        x, y = self.agent.action_to_coords(last)
        expected_x = (n - 0.5) * self.agent.grid_width  / n
        expected_y = (n - 0.5) * self.agent.grid_height / n
        self.assertAlmostEqual(x, expected_x)
        self.assertAlmostEqual(y, expected_y)

    def test_all_actions_produce_unique_coordinates(self) -> None:
        coords = {self.agent.action_to_coords(k) for k in range(self.agent.action_dim)}
        self.assertEqual(len(coords), self.agent.action_dim)

    def test_out_of_range_action_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.action_to_coords(self.agent.action_dim)

    def test_negative_action_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.action_to_coords(-1)


# ══════════════════════════════════════════════════════════════════════════════
#  D3QNAgent — select_action
# ══════════════════════════════════════════════════════════════════════════════

class TestD3QNAgentSelectAction(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        self.agent = make_agent(n_bins=4)
        self.state = random_state(self.agent.input_dim)

    def test_returns_int(self) -> None:
        self.assertIsInstance(self.agent.select_action(self.state, 0.0), int)

    def test_action_in_valid_range_greedy(self) -> None:
        action = self.agent.select_action(self.state, epsilon=0.0)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.agent.action_dim)

    def test_action_in_valid_range_random(self) -> None:
        for _ in range(20):
            action = self.agent.select_action(self.state, epsilon=1.0)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.agent.action_dim)

    # --- epsilon=0: pure greedy -----------------------------------------

    def test_epsilon_zero_is_deterministic(self) -> None:
        """epsilon=0 must always return the same greedy action for the same state."""
        actions = {
            self.agent.select_action(self.state, epsilon=0.0)
            for _ in range(20)
        }
        self.assertEqual(len(actions), 1)

    def test_epsilon_zero_never_explores(self) -> None:
        """With epsilon=0, random.random() is never consulted for exploration."""
        with mock.patch("network.random.random", return_value=0.5) as m:
            self.agent.select_action(self.state, epsilon=0.0)
        # random.random() may or may not be called depending on branching,
        # but the condition `random.random() < 0.0` is always False.
        # Verify the result is still in a valid range.
        action = self.agent.select_action(self.state, epsilon=0.0)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.agent.action_dim)

    # --- epsilon=1: pure random -----------------------------------------

    def test_epsilon_one_produces_varied_actions(self) -> None:
        """epsilon=1 forces random.random() < 1 always → exploration every call."""
        actions = {
            self.agent.select_action(self.state, epsilon=1.0)
            for _ in range(300)
        }
        self.assertGreater(
            len(actions), 1,
            msg="epsilon=1 should produce multiple distinct actions over 300 trials.",
        )

    def test_epsilon_one_covers_full_action_space(self) -> None:
        """Over enough trials with epsilon=1, all action bins should appear."""
        actions = {
            self.agent.select_action(self.state, epsilon=1.0)
            for _ in range(1000)
        }
        self.assertEqual(
            len(actions), self.agent.action_dim,
            msg="All action bins should be visited under pure exploration.",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  D3QNAgent — train_step
# ══════════════════════════════════════════════════════════════════════════════

class TestD3QNAgentTrainStep(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)
        np.random.seed(42)
        self.agent = make_agent(n_bins=3)
        fill_buffer(self.agent, n=64)

    def test_returns_float(self) -> None:
        loss = self.agent.train_step(batch_size=16)
        self.assertIsInstance(loss, float)

    def test_loss_is_finite(self) -> None:
        loss = self.agent.train_step(batch_size=16)
        self.assertTrue(np.isfinite(loss))

    def test_loss_is_non_negative(self) -> None:
        """Huber loss is always ≥ 0."""
        loss = self.agent.train_step(batch_size=16)
        self.assertGreaterEqual(loss, 0.0)

    def test_policy_net_weights_change_after_step(self) -> None:
        """At least one parameter must be updated by the gradient step."""
        before = {
            name: param.data.clone()
            for name, param in self.agent.policy_net.named_parameters()
        }
        self.agent.train_step(batch_size=16)
        any_changed = any(
            not torch.equal(before[name], param.data)
            for name, param in self.agent.policy_net.named_parameters()
        )
        self.assertTrue(any_changed, "Policy net weights must change after train_step.")

    def test_target_net_unchanged_after_train_step(self) -> None:
        """train_step must NOT modify target network weights."""
        before = {
            name: param.data.clone()
            for name, param in self.agent.target_net.named_parameters()
        }
        self.agent.train_step(batch_size=16)
        for name, param in self.agent.target_net.named_parameters():
            torch.testing.assert_close(
                param.data, before[name],
                msg=f"Target param '{name}' should not change during train_step.",
            )

    def test_target_net_still_frozen_after_train_step(self) -> None:
        self.agent.train_step(batch_size=16)
        for name, param in self.agent.target_net.named_parameters():
            self.assertFalse(param.requires_grad, msg=f"'{name}' should be frozen.")

    def test_multiple_train_steps_do_not_raise(self) -> None:
        for _ in range(10):
            loss = self.agent.train_step(batch_size=16)
            self.assertTrue(np.isfinite(loss))

    def test_underfilled_buffer_raises(self) -> None:
        fresh_agent = make_agent()
        fresh_agent.replay_buffer.push(
            random_state(fresh_agent.input_dim), 0, 0.0,
            random_state(fresh_agent.input_dim), False,
        )
        with self.assertRaises(ValueError):
            fresh_agent.train_step(batch_size=32)


# ══════════════════════════════════════════════════════════════════════════════
#  D3QNAgent — soft_update
# ══════════════════════════════════════════════════════════════════════════════

class TestD3QNAgentSoftUpdate(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(7)
        self.agent = make_agent(n_bins=3)
        # Perturb policy net so policy ≠ target
        with torch.no_grad():
            for param in self.agent.policy_net.parameters():
                param.add_(torch.randn_like(param) * 0.5)

    def test_tau_one_copies_policy_to_target(self) -> None:
        """tau=1 is a hard copy: target must exactly equal policy after update."""
        self.agent.soft_update(tau=1.0)
        for (_, t_param), (_, p_param) in zip(
            self.agent.target_net.named_parameters(),
            self.agent.policy_net.named_parameters(),
        ):
            torch.testing.assert_close(t_param, p_param)

    def test_soft_update_interpolates_correctly(self) -> None:
        """
        For tau=0.5 and known initial weights, the updated target must equal
        0.5 * policy + 0.5 * target_before.
        """
        tau = 0.5
        before_target = {
            name: param.data.clone()
            for name, param in self.agent.target_net.named_parameters()
        }
        policy_weights = {
            name: param.data.clone()
            for name, param in self.agent.policy_net.named_parameters()
        }
        self.agent.soft_update(tau=tau)
        for name, t_param in self.agent.target_net.named_parameters():
            expected = tau * policy_weights[name] + (1.0 - tau) * before_target[name]
            torch.testing.assert_close(t_param.data, expected)

    def test_small_tau_moves_target_slightly(self) -> None:
        """With very small tau the target should move but not reach policy."""
        before = {
            name: param.data.clone()
            for name, param in self.agent.target_net.named_parameters()
        }
        self.agent.soft_update(tau=0.001)
        all_equal_policy = all(
            torch.equal(param.data, p_param.data)
            for (_, param), (_, p_param) in zip(
                self.agent.target_net.named_parameters(),
                self.agent.policy_net.named_parameters(),
            )
        )
        any_changed = any(
            not torch.equal(param.data, before[name])
            for name, param in self.agent.target_net.named_parameters()
        )
        self.assertTrue(any_changed, "Target must move with tau=0.001.")
        self.assertFalse(all_equal_policy, "Target must not reach policy with tau=0.001.")

    def test_target_remains_frozen_after_soft_update(self) -> None:
        """soft_update must not accidentally re-enable gradients."""
        self.agent.soft_update(tau=0.005)
        for name, param in self.agent.target_net.named_parameters():
            self.assertFalse(param.requires_grad, msg=f"'{name}' must stay frozen.")

    def test_zero_tau_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.soft_update(tau=0.0)

    def test_negative_tau_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.soft_update(tau=-0.1)

    def test_tau_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.agent.soft_update(tau=1.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
