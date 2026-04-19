"""
test_environment.py

Comprehensive unittest suite for IABEnv (environment.py).
Uses unittest.mock to isolate reward components so that stochastic channel
draws never cause spurious failures.
"""

import unittest
from unittest import mock

import numpy as np

from entities import IABNode, User
from environment import IABEnv


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_env(
    width: float = 500.0,
    height: float = 500.0,
    num_users: int = 10,
    seed: int = 42,
) -> IABEnv:
    """Return a seeded IABEnv for deterministic tests."""
    return IABEnv(width=width, height=height, num_users=num_users, seed=seed)


# ══════════════════════════════════════════════════════════════════════════════
#  reset()
# ══════════════════════════════════════════════════════════════════════════════

class TestReset(unittest.TestCase):
    """Tests for IABEnv.reset()."""

    def setUp(self) -> None:
        self.env = make_env()

    # --- Return value schema ---------------------------------------------

    def test_returns_dict(self) -> None:
        state = self.env.reset()
        self.assertIsInstance(state, dict)

    def test_state_contains_required_keys(self) -> None:
        state = self.env.reset()
        required = {
            "user_positions", "node_positions", "donor_position",
            "num_nodes", "sum_rate_mbps", "coverage_rate",
        }
        self.assertTrue(required.issubset(state.keys()))

    # --- Relay nodes cleared ---------------------------------------------

    def test_num_nodes_zero_after_reset(self) -> None:
        self.env.step((100.0, 100.0))
        state = self.env.reset()
        self.assertEqual(state["num_nodes"], 0)

    def test_relay_list_empty_after_reset(self) -> None:
        self.env.step((100.0, 100.0))
        self.env.reset()
        self.assertEqual(len(self.env._relay_nodes), 0)

    def test_node_positions_all_zero_after_reset(self) -> None:
        self.env.step((200.0, 200.0))
        state = self.env.reset()
        self.assertTrue(np.all(state["node_positions"] == 0.0))

    # --- Donor node ------------------------------------------------------

    def test_donor_is_iabnode_instance(self) -> None:
        self.env.reset()
        self.assertIsInstance(self.env._donor, IABNode)

    def test_donor_flagged_as_donor(self) -> None:
        self.env.reset()
        self.assertTrue(self.env._donor.is_donor)

    def test_donor_placed_at_grid_centre_x(self) -> None:
        self.env.reset()
        self.assertAlmostEqual(self.env._donor.x, self.env.grid.width / 2.0)

    def test_donor_placed_at_grid_centre_y(self) -> None:
        self.env.reset()
        self.assertAlmostEqual(self.env._donor.y, self.env.grid.height / 2.0)

    def test_donor_flow_in_matches_class_constant(self) -> None:
        self.env.reset()
        self.assertAlmostEqual(
            self.env._donor.flow_in_capacity,
            IABEnv.DONOR_FLOW_CAPACITY_MBPS,
        )

    def test_donor_flow_out_is_zero_after_reset(self) -> None:
        self.env.reset()
        self.assertAlmostEqual(self.env._donor.flow_out_demand, 0.0)

    def test_donor_position_in_state_matches_grid_centre(self) -> None:
        state = self.env.reset()
        cx = self.env.grid.width / 2.0
        cy = self.env.grid.height / 2.0
        np.testing.assert_array_almost_equal(
            state["donor_position"], np.array([cx, cy])
        )

    # --- User population -------------------------------------------------

    def test_correct_number_of_users_generated(self) -> None:
        self.env.reset()
        self.assertEqual(len(self.env.grid.users), self.env.num_users)

    def test_user_positions_shape(self) -> None:
        state = self.env.reset()
        self.assertEqual(
            state["user_positions"].shape, (self.env.num_users, 2)
        )

    def test_user_positions_within_grid_bounds(self) -> None:
        state = self.env.reset()
        xs = state["user_positions"][:, 0]
        ys = state["user_positions"][:, 1]
        self.assertTrue(np.all(xs >= 0.0))
        self.assertTrue(np.all(xs < self.env.grid.width))
        self.assertTrue(np.all(ys >= 0.0))
        self.assertTrue(np.all(ys < self.env.grid.height))

    # --- Fixed-size node_positions padding ------------------------------

    def test_node_positions_shape_is_max_nodes_x_2(self) -> None:
        state = self.env.reset()
        self.assertEqual(
            state["node_positions"].shape, (IABEnv.MAX_NODES, 2)
        )

    # --- Scalar metrics zeroed ------------------------------------------

    def test_sum_rate_zero_on_reset(self) -> None:
        state = self.env.reset()
        self.assertAlmostEqual(state["sum_rate_mbps"], 0.0)

    def test_coverage_rate_zero_on_reset(self) -> None:
        state = self.env.reset()
        self.assertAlmostEqual(state["coverage_rate"], 0.0)

    # --- Idempotence -----------------------------------------------------

    def test_consecutive_resets_produce_zero_nodes(self) -> None:
        for _ in range(3):
            state = self.env.reset()
            self.assertEqual(state["num_nodes"], 0)


# ══════════════════════════════════════════════════════════════════════════════
#  step() — node placement and state updates
# ══════════════════════════════════════════════════════════════════════════════

class TestStepNodePlacement(unittest.TestCase):
    """Tests for relay node placement behaviour in step()."""

    def setUp(self) -> None:
        self.env = make_env(seed=7)

    # --- Return type and schema -----------------------------------------

    def test_returns_three_element_tuple(self) -> None:
        result = self.env.step((250.0, 250.0))
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_first_element_is_dict(self) -> None:
        state, _, _ = self.env.step((250.0, 250.0))
        self.assertIsInstance(state, dict)

    def test_second_element_is_float(self) -> None:
        _, reward, _ = self.env.step((250.0, 250.0))
        self.assertIsInstance(reward, float)

    def test_third_element_is_bool(self) -> None:
        _, _, done = self.env.step((250.0, 250.0))
        self.assertIsInstance(done, bool)

    def test_state_contains_required_keys(self) -> None:
        state, _, _ = self.env.step((100.0, 100.0))
        required = {
            "user_positions", "node_positions", "donor_position",
            "num_nodes", "sum_rate_mbps", "coverage_rate",
        }
        self.assertTrue(required.issubset(state.keys()))

    # --- Relay instantiation --------------------------------------------

    def test_step_adds_exactly_one_relay(self) -> None:
        self.assertEqual(len(self.env._relay_nodes), 0)
        self.env.step((200.0, 200.0))
        self.assertEqual(len(self.env._relay_nodes), 1)

    def test_relay_is_iabnode_instance(self) -> None:
        self.env.step((200.0, 200.0))
        self.assertIsInstance(self.env._relay_nodes[0], IABNode)

    def test_relay_is_not_flagged_as_donor(self) -> None:
        self.env.step((200.0, 200.0))
        self.assertFalse(self.env._relay_nodes[0].is_donor)

    def test_relay_placed_at_given_x(self) -> None:
        self.env.step((123.4, 56.7))
        self.assertAlmostEqual(self.env._relay_nodes[0].x, 123.4, places=4)

    def test_relay_placed_at_given_y(self) -> None:
        self.env.step((123.4, 56.7))
        self.assertAlmostEqual(self.env._relay_nodes[0].y, 56.7, places=4)

    # --- State updates --------------------------------------------------

    def test_num_nodes_increments_after_step(self) -> None:
        state, _, _ = self.env.step((200.0, 200.0))
        self.assertEqual(state["num_nodes"], 1)

    def test_num_nodes_increments_on_each_step(self) -> None:
        for expected in range(1, 4):
            state, _, _ = self.env.step((100.0 * expected, 100.0))
            self.assertEqual(state["num_nodes"], expected)

    def test_node_position_in_state_matches_placed_coords(self) -> None:
        state, _, _ = self.env.step((175.0, 325.0))
        np.testing.assert_array_almost_equal(
            state["node_positions"][0], np.array([175.0, 325.0])
        )

    def test_second_node_position_in_state(self) -> None:
        self.env.step((100.0, 100.0))
        state, _, _ = self.env.step((300.0, 400.0))
        np.testing.assert_array_almost_equal(
            state["node_positions"][1], np.array([300.0, 400.0])
        )

    def test_node_positions_padded_shape_unchanged_after_step(self) -> None:
        state, _, _ = self.env.step((200.0, 200.0))
        self.assertEqual(
            state["node_positions"].shape, (IABEnv.MAX_NODES, 2)
        )

    def test_padding_rows_remain_zero_after_one_step(self) -> None:
        state, _, _ = self.env.step((200.0, 200.0))
        # Rows 1 through MAX_NODES-1 must still be zero.
        np.testing.assert_array_equal(
            state["node_positions"][1:], np.zeros((IABEnv.MAX_NODES - 1, 2))
        )

    def test_relay_flow_in_set_after_step(self) -> None:
        """Backhaul Shannon capacity must be a finite positive float after step."""
        self.env.step((250.0, 250.0))
        relay = self.env._relay_nodes[0]
        self.assertGreater(relay.flow_in_capacity, 0.0)
        self.assertTrue(np.isfinite(relay.flow_in_capacity))

    def test_relay_flow_out_set_after_step(self) -> None:
        """flow_out_demand must be non-negative after step."""
        self.env.step((250.0, 250.0))
        relay = self.env._relay_nodes[0]
        self.assertGreaterEqual(relay.flow_out_demand, 0.0)

    # --- Coordinate clamping --------------------------------------------

    def test_x_above_width_clamped_to_boundary(self) -> None:
        self.env.step((9999.0, 250.0))
        self.assertLess(self.env._relay_nodes[0].x, self.env.grid.width)

    def test_y_above_height_clamped_to_boundary(self) -> None:
        self.env.step((250.0, 9999.0))
        self.assertLess(self.env._relay_nodes[0].y, self.env.grid.height)

    def test_negative_x_clamped_to_zero(self) -> None:
        self.env.step((-100.0, 250.0))
        self.assertGreaterEqual(self.env._relay_nodes[0].x, 0.0)

    def test_negative_y_clamped_to_zero(self) -> None:
        self.env.step((250.0, -100.0))
        self.assertGreaterEqual(self.env._relay_nodes[0].y, 0.0)

    # --- Termination ----------------------------------------------------

    def test_done_false_before_max_nodes(self) -> None:
        for _ in range(IABEnv.MAX_NODES - 1):
            _, _, done = self.env.step((250.0, 250.0))
            if done:
                # All demands met early — acceptable; skip remainder.
                return
        # If we reach here without all demands met, done must be False.
        self.assertFalse(done)

    def test_done_true_at_max_nodes(self) -> None:
        done = False
        for _ in range(IABEnv.MAX_NODES):
            _, _, done = self.env.step((250.0, 250.0))
        self.assertTrue(done)

    def test_done_is_bool(self) -> None:
        _, _, done = self.env.step((250.0, 250.0))
        self.assertIsInstance(done, bool)

    def test_sum_rate_positive_after_step(self) -> None:
        state, _, _ = self.env.step((250.0, 250.0))
        self.assertGreater(state["sum_rate_mbps"], 0.0)

    def test_coverage_rate_in_unit_interval(self) -> None:
        state, _, _ = self.env.step((250.0, 250.0))
        self.assertGreaterEqual(state["coverage_rate"], 0.0)
        self.assertLessEqual(state["coverage_rate"], 1.0)

    # --- Reset after steps clears relay list ----------------------------

    def test_reset_after_steps_clears_relay_nodes(self) -> None:
        for _ in range(3):
            self.env.step((200.0, 200.0))
        state = self.env.reset()
        self.assertEqual(state["num_nodes"], 0)
        self.assertEqual(len(self.env._relay_nodes), 0)


# ══════════════════════════════════════════════════════════════════════════════
#  Reward — CAPEX and backhaul penalties
# ══════════════════════════════════════════════════════════════════════════════

class TestRewardCapexPenalty(unittest.TestCase):
    """
    Tests isolating the CAPEX (-50/node) penalty component.

    ``IABNode.check_backhaul_constraint`` is patched to return ``True`` so that
    zero backhaul penalties are incurred, leaving only the sum-rate and CAPEX
    terms in the reward formula:

        R = sum_rate  −  50 · num_nodes
    """

    def setUp(self) -> None:
        self.env = make_env(seed=0)

    def _step_no_backhaul_penalty(
        self, coords: tuple[float, float]
    ) -> tuple[dict, float, bool]:
        """Run step() with backhaul constraint always satisfied."""
        with mock.patch.object(
            IABNode, "check_backhaul_constraint", return_value=True
        ):
            return self.env.step(coords)

    # --- Single node CAPEX ----------------------------------------------

    def test_capex_penalty_50_for_one_node(self) -> None:
        state, reward, _ = self._step_no_backhaul_penalty((250.0, 250.0))
        expected = state["sum_rate_mbps"] - IABEnv.CAPEX_PENALTY * 1
        self.assertAlmostEqual(reward, expected, places=4)

    def test_reward_less_than_sum_rate_by_capex(self) -> None:
        state, reward, _ = self._step_no_backhaul_penalty((250.0, 250.0))
        self.assertAlmostEqual(
            state["sum_rate_mbps"] - reward,
            IABEnv.CAPEX_PENALTY * 1,
            places=4,
        )

    # --- Two nodes CAPEX ------------------------------------------------

    def test_capex_penalty_100_for_two_nodes(self) -> None:
        self._step_no_backhaul_penalty((100.0, 100.0))
        state, reward, _ = self._step_no_backhaul_penalty((300.0, 300.0))
        expected = state["sum_rate_mbps"] - IABEnv.CAPEX_PENALTY * 2
        self.assertAlmostEqual(reward, expected, places=4)

    def test_capex_scales_linearly_with_node_count(self) -> None:
        """
        After N steps (all constraints satisfied), the CAPEX deduction must
        equal CAPEX_PENALTY * N exactly.
        """
        for n in range(1, 5):
            self.env.reset()
            for k in range(1, n + 1):
                state, reward, done = self._step_no_backhaul_penalty(
                    (50.0 * k, 50.0 * k)
                )
                if done:
                    break
            deduction = state["sum_rate_mbps"] - reward
            self.assertAlmostEqual(
                deduction,
                IABEnv.CAPEX_PENALTY * state["num_nodes"],
                places=4,
                msg=f"CAPEX deduction wrong at {n} node(s)",
            )

    # --- Reward is float ------------------------------------------------

    def test_reward_is_float(self) -> None:
        _, reward, _ = self._step_no_backhaul_penalty((250.0, 250.0))
        self.assertIsInstance(reward, float)


class TestRewardBackhaulPenalty(unittest.TestCase):
    """
    Tests isolating the backhaul penalty (-100/violated node) component.

    Two complementary fixtures are used:
    - ``check_backhaul_constraint`` patched to ``False``  → penalty is applied.
    - ``check_backhaul_constraint`` patched to ``True``   → penalty is absent.
    """

    def setUp(self) -> None:
        self.env = make_env(seed=1)

    # --- Penalty applied when constraint is violated --------------------

    def test_backhaul_penalty_100_applied_for_one_violation(self) -> None:
        """
        With 1 relay node and check_backhaul_constraint() forced False,
        reward must equal  sum_rate − CAPEX·1 − BACKHAUL·1.
        """
        with mock.patch.object(
            IABNode, "check_backhaul_constraint", return_value=False
        ):
            state, reward, _ = self.env.step((250.0, 250.0))

        expected = (
            state["sum_rate_mbps"]
            - IABEnv.CAPEX_PENALTY * 1
            - IABEnv.BACKHAUL_PENALTY * 1
        )
        self.assertAlmostEqual(reward, expected, places=4)

    def test_backhaul_penalty_200_for_two_violated_nodes(self) -> None:
        """With 2 relay nodes both violating, deduction = 2 × BACKHAUL + 2 × CAPEX."""
        with mock.patch.object(
            IABNode, "check_backhaul_constraint", return_value=False
        ):
            self.env.step((100.0, 100.0))
            state, reward, _ = self.env.step((300.0, 300.0))

        expected = (
            state["sum_rate_mbps"]
            - IABEnv.CAPEX_PENALTY * 2
            - IABEnv.BACKHAUL_PENALTY * 2
        )
        self.assertAlmostEqual(reward, expected, places=4)

    def test_reward_lower_with_violation_than_without(self) -> None:
        """Violated reward must be strictly less than satisfied reward for same action."""
        self.env.reset()
        with mock.patch.object(
            IABNode, "check_backhaul_constraint", return_value=True
        ):
            _, reward_ok, _ = self.env.step((250.0, 250.0))

        self.env.reset()
        with mock.patch.object(
            IABNode, "check_backhaul_constraint", return_value=False
        ):
            _, reward_violated, _ = self.env.step((250.0, 250.0))

        self.assertLess(reward_violated, reward_ok)

    def test_reward_difference_equals_backhaul_penalty(self) -> None:
        """
        The difference between satisfied and violated reward for the same
        single-relay scenario must equal exactly BACKHAUL_PENALTY.
        Because sum_rate is stochastic, each trial uses an independent seeded
        env so both scenarios start from the same RNG state.
        """
        # Both envs share the same seed so user layout and LoS draws are
        # identical up to the constraint check.
        env_ok = IABEnv(500, 500, 10, seed=5)
        env_bad = IABEnv(500, 500, 10, seed=5)

        with mock.patch.object(
            IABNode, "check_backhaul_constraint", return_value=True
        ):
            state_ok, reward_ok, _ = env_ok.step((250.0, 250.0))

        with mock.patch.object(
            IABNode, "check_backhaul_constraint", return_value=False
        ):
            state_bad, reward_bad, _ = env_bad.step((250.0, 250.0))

        self.assertAlmostEqual(
            reward_ok - reward_bad,
            IABEnv.BACKHAUL_PENALTY,
            places=4,
        )

    # --- No penalty when constraint is satisfied ------------------------

    def test_no_backhaul_penalty_when_constraint_satisfied(self) -> None:
        with mock.patch.object(
            IABNode, "check_backhaul_constraint", return_value=True
        ):
            state, reward, _ = self.env.step((250.0, 250.0))

        expected = state["sum_rate_mbps"] - IABEnv.CAPEX_PENALTY * 1
        self.assertAlmostEqual(reward, expected, places=4)

    # --- Algebraic identity always holds --------------------------------

    def test_reward_formula_identity_holds_for_natural_step(self) -> None:
        """
        Without any mocking, verify:
            reward == sum_rate − CAPEX·N − BACKHAUL·violations
        regardless of whether violations actually occur.
        """
        state, reward, _ = self.env.step((250.0, 250.0))
        violations = sum(
            1 for r in self.env._relay_nodes
            if not r.check_backhaul_constraint()
        )
        expected = (
            state["sum_rate_mbps"]
            - IABEnv.CAPEX_PENALTY * state["num_nodes"]
            - IABEnv.BACKHAUL_PENALTY * violations
        )
        self.assertAlmostEqual(reward, expected, places=4)

    def test_reward_formula_identity_holds_over_multiple_steps(self) -> None:
        """The reward identity must hold at each individual step."""
        for k in range(1, 5):
            state, reward, done = self.env.step((80.0 * k, 80.0 * k))
            violations = sum(
                1 for r in self.env._relay_nodes
                if not r.check_backhaul_constraint()
            )
            expected = (
                state["sum_rate_mbps"]
                - IABEnv.CAPEX_PENALTY * state["num_nodes"]
                - IABEnv.BACKHAUL_PENALTY * violations
            )
            self.assertAlmostEqual(
                reward, expected, places=4,
                msg=f"Reward identity failed at step {k}",
            )
            if done:
                break


if __name__ == "__main__":
    unittest.main(verbosity=2)
