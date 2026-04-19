"""
test_entities.py

Comprehensive unittest suite for entities.py (User, IABNode, CityGrid).
"""

import unittest

import numpy as np

from entities import CityGrid, IABNode, User


# ══════════════════════════════════════════════════════════════════════════════
#  User
# ══════════════════════════════════════════════════════════════════════════════

class TestUserInit(unittest.TestCase):
    """Tests for User initialisation and attribute correctness."""

    def test_coordinates_stored_correctly(self) -> None:
        u = User(x=10.5, y=20.3)
        self.assertAlmostEqual(u.x, 10.5)
        self.assertAlmostEqual(u.y, 20.3)

    def test_default_demand_is_100_mbps(self) -> None:
        u = User(x=0.0, y=0.0)
        self.assertAlmostEqual(u.data_demand_mbps, 100.0)

    def test_custom_demand_stored_correctly(self) -> None:
        u = User(x=5.0, y=5.0, data_demand_mbps=250.0)
        self.assertAlmostEqual(u.data_demand_mbps, 250.0)

    def test_attributes_are_floats(self) -> None:
        u = User(x=3, y=7, data_demand_mbps=50)
        self.assertIsInstance(u.x, float)
        self.assertIsInstance(u.y, float)
        self.assertIsInstance(u.data_demand_mbps, float)

    def test_zero_coordinates_accepted(self) -> None:
        u = User(x=0.0, y=0.0)
        self.assertAlmostEqual(u.x, 0.0)
        self.assertAlmostEqual(u.y, 0.0)

    def test_negative_coordinates_accepted(self) -> None:
        """Negative coords are geometrically valid outside a grid context."""
        u = User(x=-5.0, y=-10.0)
        self.assertAlmostEqual(u.x, -5.0)
        self.assertAlmostEqual(u.y, -10.0)

    def test_large_coordinates_accepted(self) -> None:
        u = User(x=1e6, y=1e6)
        self.assertAlmostEqual(u.x, 1e6)

    def test_zero_demand_raises(self) -> None:
        with self.assertRaises(ValueError):
            User(x=1.0, y=1.0, data_demand_mbps=0.0)

    def test_negative_demand_raises(self) -> None:
        with self.assertRaises(ValueError):
            User(x=1.0, y=1.0, data_demand_mbps=-50.0)

    def test_repr_contains_key_info(self) -> None:
        u = User(x=1.0, y=2.0, data_demand_mbps=150.0)
        r = repr(u)
        self.assertIn("User", r)
        self.assertIn("150.0", r)


# ══════════════════════════════════════════════════════════════════════════════
#  IABNode — initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestIABNodeInit(unittest.TestCase):
    """Tests for IABNode initialisation and attribute correctness."""

    def _make_donor(self, **kwargs) -> IABNode:
        defaults = dict(x=50.0, y=50.0, is_donor=True,
                        flow_in_capacity=1000.0, flow_out_demand=500.0)
        defaults.update(kwargs)
        return IABNode(**defaults)

    def _make_relay(self, **kwargs) -> IABNode:
        defaults = dict(x=100.0, y=200.0, is_donor=False,
                        flow_in_capacity=400.0, flow_out_demand=300.0)
        defaults.update(kwargs)
        return IABNode(**defaults)

    def test_donor_coordinates_stored_correctly(self) -> None:
        node = self._make_donor(x=12.3, y=45.6)
        self.assertAlmostEqual(node.x, 12.3)
        self.assertAlmostEqual(node.y, 45.6)

    def test_relay_coordinates_stored_correctly(self) -> None:
        node = self._make_relay(x=77.7, y=88.8)
        self.assertAlmostEqual(node.x, 77.7)
        self.assertAlmostEqual(node.y, 88.8)

    def test_is_donor_true_stored(self) -> None:
        node = self._make_donor()
        self.assertTrue(node.is_donor)
        self.assertIsInstance(node.is_donor, bool)

    def test_is_donor_false_stored(self) -> None:
        node = self._make_relay()
        self.assertFalse(node.is_donor)
        self.assertIsInstance(node.is_donor, bool)

    def test_flow_in_capacity_stored(self) -> None:
        node = self._make_donor(flow_in_capacity=750.0)
        self.assertAlmostEqual(node.flow_in_capacity, 750.0)

    def test_flow_out_demand_stored(self) -> None:
        node = self._make_donor(flow_out_demand=300.0)
        self.assertAlmostEqual(node.flow_out_demand, 300.0)

    def test_attributes_are_floats(self) -> None:
        node = IABNode(x=1, y=2, is_donor=True,
                       flow_in_capacity=500, flow_out_demand=200)
        self.assertIsInstance(node.x, float)
        self.assertIsInstance(node.y, float)
        self.assertIsInstance(node.flow_in_capacity, float)
        self.assertIsInstance(node.flow_out_demand, float)

    def test_zero_flow_values_accepted(self) -> None:
        node = IABNode(x=0.0, y=0.0, is_donor=True,
                       flow_in_capacity=0.0, flow_out_demand=0.0)
        self.assertAlmostEqual(node.flow_in_capacity, 0.0)
        self.assertAlmostEqual(node.flow_out_demand, 0.0)

    def test_negative_flow_in_raises(self) -> None:
        with self.assertRaises(ValueError):
            IABNode(x=0.0, y=0.0, is_donor=True,
                    flow_in_capacity=-1.0, flow_out_demand=0.0)

    def test_negative_flow_out_raises(self) -> None:
        with self.assertRaises(ValueError):
            IABNode(x=0.0, y=0.0, is_donor=False,
                    flow_in_capacity=100.0, flow_out_demand=-1.0)

    def test_repr_contains_key_info(self) -> None:
        node = self._make_donor()
        r = repr(node)
        self.assertIn("IABNode", r)
        self.assertIn("Donor", r)

    def test_repr_relay_label(self) -> None:
        node = self._make_relay()
        self.assertIn("Relay", repr(node))


# ══════════════════════════════════════════════════════════════════════════════
#  IABNode — backhaul constraint
# ══════════════════════════════════════════════════════════════════════════════

class TestIABNodeBackhaulConstraint(unittest.TestCase):
    """Tests for IABNode.check_backhaul_constraint()."""

    def _node(self, flow_in: float, flow_out: float) -> IABNode:
        return IABNode(x=0.0, y=0.0, is_donor=False,
                       flow_in_capacity=flow_in, flow_out_demand=flow_out)

    # --- True cases -------------------------------------------------------

    def test_returns_true_when_flow_in_greater_than_flow_out(self) -> None:
        self.assertTrue(self._node(500.0, 300.0).check_backhaul_constraint())

    def test_returns_true_when_flow_in_equals_flow_out(self) -> None:
        self.assertTrue(self._node(400.0, 400.0).check_backhaul_constraint())

    def test_returns_true_when_both_zero(self) -> None:
        self.assertTrue(self._node(0.0, 0.0).check_backhaul_constraint())

    def test_returns_true_large_surplus(self) -> None:
        self.assertTrue(self._node(10_000.0, 1.0).check_backhaul_constraint())

    def test_returns_true_just_above_demand(self) -> None:
        self.assertTrue(self._node(300.001, 300.0).check_backhaul_constraint())

    def test_donor_node_returns_true_when_satisfied(self) -> None:
        node = IABNode(x=0.0, y=0.0, is_donor=True,
                       flow_in_capacity=10_000.0, flow_out_demand=5_000.0)
        self.assertTrue(node.check_backhaul_constraint())

    # --- False cases ------------------------------------------------------

    def test_returns_false_when_flow_in_less_than_flow_out(self) -> None:
        self.assertFalse(self._node(200.0, 500.0).check_backhaul_constraint())

    def test_returns_false_zero_in_nonzero_out(self) -> None:
        self.assertFalse(self._node(0.0, 100.0).check_backhaul_constraint())

    def test_returns_false_just_below_demand(self) -> None:
        self.assertFalse(self._node(299.999, 300.0).check_backhaul_constraint())

    def test_returns_false_large_deficit(self) -> None:
        self.assertFalse(self._node(1.0, 9_999.0).check_backhaul_constraint())

    # --- Return type ------------------------------------------------------

    def test_return_type_is_bool(self) -> None:
        result = self._node(100.0, 50.0).check_backhaul_constraint()
        self.assertIsInstance(result, bool)


# ══════════════════════════════════════════════════════════════════════════════
#  CityGrid — initialisation
# ══════════════════════════════════════════════════════════════════════════════

class TestCityGridInit(unittest.TestCase):
    """Tests for CityGrid initialisation."""

    def test_dimensions_stored_correctly(self) -> None:
        g = CityGrid(width=500.0, height=300.0)
        self.assertAlmostEqual(g.width, 500.0)
        self.assertAlmostEqual(g.height, 300.0)

    def test_users_list_empty_on_construction(self) -> None:
        g = CityGrid(500.0, 300.0)
        self.assertEqual(len(g.users), 0)
        self.assertIsInstance(g.users, list)

    def test_dimensions_are_floats(self) -> None:
        g = CityGrid(400, 400)
        self.assertIsInstance(g.width, float)
        self.assertIsInstance(g.height, float)

    def test_zero_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            CityGrid(width=0.0, height=500.0)

    def test_zero_height_raises(self) -> None:
        with self.assertRaises(ValueError):
            CityGrid(width=500.0, height=0.0)

    def test_negative_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            CityGrid(width=-100.0, height=500.0)

    def test_negative_height_raises(self) -> None:
        with self.assertRaises(ValueError):
            CityGrid(width=500.0, height=-100.0)

    def test_repr_contains_dimensions(self) -> None:
        g = CityGrid(500.0, 300.0)
        r = repr(g)
        self.assertIn("CityGrid", r)
        self.assertIn("500.0", r)
        self.assertIn("300.0", r)


# ══════════════════════════════════════════════════════════════════════════════
#  CityGrid — generate_users
# ══════════════════════════════════════════════════════════════════════════════

class TestCityGridGenerateUsers(unittest.TestCase):
    """Tests for CityGrid.generate_users()."""

    WIDTH: float = 500.0
    HEIGHT: float = 400.0

    def setUp(self) -> None:
        self.grid = CityGrid(self.WIDTH, self.HEIGHT)

    # --- Count correctness -----------------------------------------------

    def test_generates_exact_number_of_users(self) -> None:
        users = self.grid.generate_users(10, seed=0)
        self.assertEqual(len(users), 10)

    def test_generates_exact_number_large(self) -> None:
        users = self.grid.generate_users(1000, seed=1)
        self.assertEqual(len(users), 1000)

    def test_generates_single_user(self) -> None:
        users = self.grid.generate_users(1, seed=42)
        self.assertEqual(len(users), 1)

    def test_self_users_updated_to_match_return(self) -> None:
        returned = self.grid.generate_users(20, seed=7)
        self.assertEqual(len(self.grid.users), 20)
        self.assertIs(returned, self.grid.users)

    def test_second_call_replaces_users(self) -> None:
        self.grid.generate_users(50, seed=0)
        self.grid.generate_users(10, seed=1)
        self.assertEqual(len(self.grid.users), 10)

    # --- Boundary: x in [0, width) ---------------------------------------

    def test_all_x_coordinates_geq_zero(self) -> None:
        self.grid.generate_users(500, seed=0)
        for u in self.grid.users:
            self.assertGreaterEqual(
                u.x, 0.0,
                msg=f"x={u.x} is below 0.0",
            )

    def test_all_x_coordinates_strictly_less_than_width(self) -> None:
        self.grid.generate_users(500, seed=0)
        for u in self.grid.users:
            self.assertLess(
                u.x, self.WIDTH,
                msg=f"x={u.x} >= width={self.WIDTH}",
            )

    # --- Boundary: y in [0, height) --------------------------------------

    def test_all_y_coordinates_geq_zero(self) -> None:
        self.grid.generate_users(500, seed=0)
        for u in self.grid.users:
            self.assertGreaterEqual(
                u.y, 0.0,
                msg=f"y={u.y} is below 0.0",
            )

    def test_all_y_coordinates_strictly_less_than_height(self) -> None:
        self.grid.generate_users(500, seed=0)
        for u in self.grid.users:
            self.assertLess(
                u.y, self.HEIGHT,
                msg=f"y={u.y} >= height={self.HEIGHT}",
            )

    # --- Boundary with non-square grid -----------------------------------

    def test_boundaries_respected_non_square_grid(self) -> None:
        grid = CityGrid(width=1000.0, height=200.0)
        grid.generate_users(300, seed=5)
        for u in grid.users:
            self.assertGreaterEqual(u.x, 0.0)
            self.assertLess(u.x, 1000.0)
            self.assertGreaterEqual(u.y, 0.0)
            self.assertLess(u.y, 200.0)

    # --- Returned objects are User instances -----------------------------

    def test_all_elements_are_user_instances(self) -> None:
        users = self.grid.generate_users(15, seed=3)
        for u in users:
            self.assertIsInstance(u, User)

    # --- Demand propagation ----------------------------------------------

    def test_default_demand_applied_to_all_users(self) -> None:
        self.grid.generate_users(20, seed=0)
        for u in self.grid.users:
            self.assertAlmostEqual(u.data_demand_mbps, 100.0)

    def test_custom_demand_applied_to_all_users(self) -> None:
        self.grid.generate_users(20, data_demand_mbps=200.0, seed=0)
        for u in self.grid.users:
            self.assertAlmostEqual(u.data_demand_mbps, 200.0)

    # --- Reproducibility with seed ---------------------------------------

    def test_same_seed_produces_identical_layout(self) -> None:
        g1 = CityGrid(self.WIDTH, self.HEIGHT)
        g2 = CityGrid(self.WIDTH, self.HEIGHT)
        users1 = g1.generate_users(50, seed=99)
        users2 = g2.generate_users(50, seed=99)
        for u1, u2 in zip(users1, users2):
            self.assertAlmostEqual(u1.x, u2.x, places=10)
            self.assertAlmostEqual(u1.y, u2.y, places=10)

    def test_different_seeds_produce_different_layouts(self) -> None:
        g1 = CityGrid(self.WIDTH, self.HEIGHT)
        g2 = CityGrid(self.WIDTH, self.HEIGHT)
        users1 = g1.generate_users(50, seed=1)
        users2 = g2.generate_users(50, seed=2)
        xs1 = [u.x for u in users1]
        xs2 = [u.x for u in users2]
        self.assertFalse(
            all(abs(a - b) < 1e-9 for a, b in zip(xs1, xs2)),
            "Different seeds produced identical x-coordinates.",
        )

    # --- Distribution coverage (vectorised check) ------------------------

    def test_users_span_meaningful_portion_of_grid(self) -> None:
        """1000 uniform samples should span at least 80 % of each axis."""
        self.grid.generate_users(1000, seed=0)
        xs = np.array([u.x for u in self.grid.users])
        ys = np.array([u.y for u in self.grid.users])
        self.assertGreater(xs.max() - xs.min(), 0.8 * self.WIDTH)
        self.assertGreater(ys.max() - ys.min(), 0.8 * self.HEIGHT)

    # --- Invalid input ---------------------------------------------------

    def test_zero_num_users_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.grid.generate_users(0)

    def test_negative_num_users_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.grid.generate_users(-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
