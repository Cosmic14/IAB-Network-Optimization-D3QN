"""
test_channel_model.py

Comprehensive unittest suite for ChannelModel.
"""

import math
import unittest

import numpy as np

from channel_model import ChannelModel


class TestCalculateLoSProb(unittest.TestCase):
    """Tests for ChannelModel.calculate_los_prob."""

    def setUp(self) -> None:
        self.model = ChannelModel()

    # --- Return type and range -------------------------------------------

    def test_returns_float(self) -> None:
        result = self.model.calculate_los_prob(50.0)
        self.assertIsInstance(result, float)

    def test_probability_lower_bound_short_distance(self) -> None:
        """Very short distances should give P_LoS close to 1, never < 0."""
        result = self.model.calculate_los_prob(1.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_probability_upper_bound_short_distance(self) -> None:
        result = self.model.calculate_los_prob(5.0)
        self.assertLessEqual(result, 1.0)

    def test_probability_range_medium_distance(self) -> None:
        result = self.model.calculate_los_prob(100.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_probability_range_long_distance(self) -> None:
        result = self.model.calculate_los_prob(500.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_probability_range_very_long_distance(self) -> None:
        result = self.model.calculate_los_prob(2000.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    # --- Monotonicity ----------------------------------------------------

    def test_probability_decreases_with_distance(self) -> None:
        """P_LoS must be monotonically non-increasing as distance grows."""
        distances = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
        probs = [self.model.calculate_los_prob(d) for d in distances]
        for i in range(len(probs) - 1):
            self.assertGreaterEqual(
                probs[i], probs[i + 1],
                msg=f"P_LoS not decreasing between {distances[i]} m and {distances[i+1]} m",
            )

    # --- Known analytical values ----------------------------------------

    def test_very_short_distance_near_unity(self) -> None:
        """At d=1 m, min(18/1,1)=1; result ≈ 1*(1-exp(-1/36)) + exp(-1/36) = 1."""
        result = self.model.calculate_los_prob(1.0)
        self.assertAlmostEqual(result, 1.0, places=4)

    def test_known_value_at_18m(self) -> None:
        """At d=18 m, min(18/18,1)=1; verify against manual calculation."""
        d = 18.0
        expected = 1.0 * (1.0 - math.exp(-d / 36.0)) + math.exp(-d / 36.0)
        result = self.model.calculate_los_prob(d)
        self.assertAlmostEqual(result, expected, places=6)

    def test_known_value_at_100m(self) -> None:
        d = 100.0
        expected = (
            min(18.0 / d, 1.0) * (1.0 - math.exp(-d / 36.0))
            + math.exp(-d / 36.0)
        )
        result = self.model.calculate_los_prob(d)
        self.assertAlmostEqual(result, expected, places=6)

    # --- Edge / boundary cases ------------------------------------------

    def test_zero_distance_guard(self) -> None:
        """d=0 must not raise and must return a valid probability."""
        result = self.model.calculate_los_prob(0.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_negative_distance_guard(self) -> None:
        """Negative distances should be treated as the guard minimum (1 m)."""
        result = self.model.calculate_los_prob(-10.0)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestCalculatePathloss(unittest.TestCase):
    """Tests for ChannelModel.calculate_pathloss."""

    def setUp(self) -> None:
        self.model = ChannelModel()

    # --- Return type and sign --------------------------------------------

    def test_returns_float_los(self) -> None:
        result = self.model.calculate_pathloss(100.0, is_los=True)
        self.assertIsInstance(result, float)

    def test_returns_float_nlos(self) -> None:
        result = self.model.calculate_pathloss(100.0, is_los=False)
        self.assertIsInstance(result, float)

    def test_pathloss_is_positive(self) -> None:
        for d in [10.0, 100.0, 500.0]:
            with self.subTest(distance=d):
                self.assertGreater(self.model.calculate_pathloss(d, True), 0.0)
                self.assertGreater(self.model.calculate_pathloss(d, False), 0.0)

    # --- Atmospheric attenuation penalty ---------------------------------

    def test_atmospheric_penalty_1km_los(self) -> None:
        """
        Over exactly 1 km the atmospheric penalty must equal
        (12.6 + 16.0) = 28.6 dB.  Compare LoS pathloss at 1000 m vs the
        base LoS formula without atmospheric terms.
        """
        d_m = 1000.0
        fc = ChannelModel.CARRIER_FREQ_GHZ

        base_los = 32.4 + 21.0 * math.log10(d_m) + 20.0 * math.log10(fc)
        expected_atmospheric = (
            ChannelModel.RAIN_ATTENUATION_DB_PER_KM
            + ChannelModel.OXYGEN_ABSORPTION_DB_PER_KM
        ) * (d_m / 1000.0)  # = 28.6 dB at 1 km

        result = self.model.calculate_pathloss(d_m, is_los=True)
        self.assertAlmostEqual(result, base_los + expected_atmospheric, places=4)

    def test_atmospheric_penalty_500m_los(self) -> None:
        """Over 500 m the atmospheric penalty must equal 28.6 * 0.5 = 14.3 dB."""
        d_m = 500.0
        fc = ChannelModel.CARRIER_FREQ_GHZ

        base_los = 32.4 + 21.0 * math.log10(d_m) + 20.0 * math.log10(fc)
        expected_atmospheric = (
            ChannelModel.RAIN_ATTENUATION_DB_PER_KM
            + ChannelModel.OXYGEN_ABSORPTION_DB_PER_KM
        ) * (d_m / 1000.0)  # = 14.3 dB at 500 m

        result = self.model.calculate_pathloss(d_m, is_los=True)
        self.assertAlmostEqual(result, base_los + expected_atmospheric, places=4)

    def test_atmospheric_penalty_scales_linearly_with_distance(self) -> None:
        """
        The difference in atmospheric loss between two LoS paths should
        scale exactly with distance ratio, verifying the dB/km formulation.
        """
        d1, d2 = 200.0, 400.0
        fc = ChannelModel.CARRIER_FREQ_GHZ

        base1 = 32.4 + 21.0 * math.log10(d1) + 20.0 * math.log10(fc)
        base2 = 32.4 + 21.0 * math.log10(d2) + 20.0 * math.log10(fc)

        atm_rate = (
            ChannelModel.RAIN_ATTENUATION_DB_PER_KM
            + ChannelModel.OXYGEN_ABSORPTION_DB_PER_KM
        ) / 1000.0  # dB/m

        pl1 = self.model.calculate_pathloss(d1, is_los=True)
        pl2 = self.model.calculate_pathloss(d2, is_los=True)

        delta_actual = pl2 - pl1
        delta_expected = (base2 + atm_rate * d2) - (base1 + atm_rate * d1)
        self.assertAlmostEqual(delta_actual, delta_expected, places=4)

    def test_rain_constant_isolated(self) -> None:
        """Verify rain contribution (12.6 dB/km) is correctly embedded."""
        d_m = 1000.0
        result = self.model.calculate_pathloss(d_m, is_los=True)
        fc = ChannelModel.CARRIER_FREQ_GHZ
        base_los = 32.4 + 21.0 * math.log10(d_m) + 20.0 * math.log10(fc)
        rain_contribution = ChannelModel.RAIN_ATTENUATION_DB_PER_KM * 1.0  # 1 km
        oxygen_contribution = ChannelModel.OXYGEN_ABSORPTION_DB_PER_KM * 1.0
        self.assertAlmostEqual(
            result, base_los + rain_contribution + oxygen_contribution, places=4
        )

    def test_oxygen_constant_isolated(self) -> None:
        """Verify oxygen contribution (16.0 dB/km) is correctly embedded."""
        d_m = 1000.0
        result = self.model.calculate_pathloss(d_m, is_los=True)
        fc = ChannelModel.CARRIER_FREQ_GHZ
        base_los = 32.4 + 21.0 * math.log10(d_m) + 20.0 * math.log10(fc)
        total_atm = (
            ChannelModel.RAIN_ATTENUATION_DB_PER_KM
            + ChannelModel.OXYGEN_ABSORPTION_DB_PER_KM
        )
        # result - base_los should equal total atmospheric loss
        self.assertAlmostEqual(result - base_los, total_atm, places=4)

    # --- LoS vs NLoS ordering --------------------------------------------

    def test_nlos_geq_los_at_long_distance(self) -> None:
        """NLoS pathloss must be >= LoS pathloss (enforced by max() in model)."""
        for d in [50.0, 100.0, 200.0, 500.0, 1000.0]:
            with self.subTest(distance=d):
                pl_los = self.model.calculate_pathloss(d, is_los=True)
                pl_nlos = self.model.calculate_pathloss(d, is_los=False)
                self.assertGreaterEqual(pl_nlos, pl_los)

    # --- Monotonicity with distance --------------------------------------

    def test_pathloss_increases_with_distance_los(self) -> None:
        distances = [10.0, 50.0, 100.0, 300.0, 1000.0]
        losses = [self.model.calculate_pathloss(d, True) for d in distances]
        for i in range(len(losses) - 1):
            self.assertLess(losses[i], losses[i + 1])

    def test_pathloss_increases_with_distance_nlos(self) -> None:
        distances = [10.0, 50.0, 100.0, 300.0, 1000.0]
        losses = [self.model.calculate_pathloss(d, False) for d in distances]
        for i in range(len(losses) - 1):
            self.assertLess(losses[i], losses[i + 1])


class TestCalculateSnr(unittest.TestCase):
    """Tests for ChannelModel.calculate_snr."""

    def setUp(self) -> None:
        self.model = ChannelModel()

    def test_returns_float(self) -> None:
        result = self.model.calculate_snr(30.0, 80.0, -90.0)
        self.assertIsInstance(result, float)

    def test_snr_positive_good_link(self) -> None:
        """High Tx power, low pathloss => positive SNR."""
        result = self.model.calculate_snr(
            tx_power_dbm=30.0, pathloss_db=60.0, noise_figure_dbm=-90.0
        )
        self.assertGreater(result, 0.0)

    def test_snr_less_than_one_bad_link(self) -> None:
        """Heavy pathloss should yield SNR < 1 (negative dB)."""
        result = self.model.calculate_snr(
            tx_power_dbm=10.0, pathloss_db=150.0, noise_figure_dbm=-80.0
        )
        self.assertLess(result, 1.0)

    def test_known_snr_value(self) -> None:
        """
        P_rx = 20 - 100 = -80 dBm; N = -100 dBm
        SNR_dB = 20 dB => SNR_linear = 100.0
        """
        result = self.model.calculate_snr(
            tx_power_dbm=20.0, pathloss_db=100.0, noise_figure_dbm=-100.0
        )
        self.assertAlmostEqual(result, 100.0, places=4)

    def test_snr_increases_with_tx_power(self) -> None:
        base = self.model.calculate_snr(20.0, 80.0, -90.0)
        higher = self.model.calculate_snr(30.0, 80.0, -90.0)
        self.assertGreater(higher, base)

    def test_snr_decreases_with_pathloss(self) -> None:
        low_loss = self.model.calculate_snr(30.0, 70.0, -90.0)
        high_loss = self.model.calculate_snr(30.0, 100.0, -90.0)
        self.assertGreater(low_loss, high_loss)


class TestCalculateShannonCapacity(unittest.TestCase):
    """Tests for ChannelModel.calculate_shannon_capacity."""

    BANDWIDTH_100MHZ: float = 100e6

    def setUp(self) -> None:
        self.model = ChannelModel()

    # --- Return type and positivity -------------------------------------

    def test_returns_float(self) -> None:
        result = self.model.calculate_shannon_capacity(10.0, self.BANDWIDTH_100MHZ)
        self.assertIsInstance(result, float)

    def test_capacity_positive_100mhz(self) -> None:
        """100 MHz bandwidth with SNR=10 must yield a positive Mbps value."""
        result = self.model.calculate_shannon_capacity(10.0, self.BANDWIDTH_100MHZ)
        self.assertGreater(result, 0.0)

    def test_capacity_unit_is_mbps_not_bps(self) -> None:
        """
        For 100 MHz BW and SNR=1, C = 100e6 * log2(2) = 100e6 bps = 100 Mbps.
        Confirms the /1e6 conversion is applied.
        """
        result = self.model.calculate_shannon_capacity(1.0, self.BANDWIDTH_100MHZ)
        self.assertAlmostEqual(result, 100.0, places=3)

    # --- Known analytical values ----------------------------------------

    def test_known_capacity_snr_1_100mhz(self) -> None:
        """SNR=1 → C = B*log2(2) = B Mbps = 100 Mbps for 100 MHz."""
        result = self.model.calculate_shannon_capacity(1.0, self.BANDWIDTH_100MHZ)
        self.assertAlmostEqual(result, 100.0, places=3)

    def test_known_capacity_snr_3_100mhz(self) -> None:
        """SNR=3 → C = 100e6 * log2(4) = 200 Mbps."""
        result = self.model.calculate_shannon_capacity(3.0, self.BANDWIDTH_100MHZ)
        self.assertAlmostEqual(result, 200.0, places=3)

    def test_known_capacity_snr_7_100mhz(self) -> None:
        """SNR=7 → C = 100e6 * log2(8) = 300 Mbps."""
        result = self.model.calculate_shannon_capacity(7.0, self.BANDWIDTH_100MHZ)
        self.assertAlmostEqual(result, 300.0, places=3)

    # --- Monotonicity and scaling ---------------------------------------

    def test_capacity_increases_with_snr(self) -> None:
        snr_values = [0.1, 1.0, 5.0, 10.0, 100.0]
        capacities = [
            self.model.calculate_shannon_capacity(s, self.BANDWIDTH_100MHZ)
            for s in snr_values
        ]
        for i in range(len(capacities) - 1):
            self.assertLess(capacities[i], capacities[i + 1])

    def test_capacity_scales_linearly_with_bandwidth(self) -> None:
        """Doubling bandwidth must double capacity (SNR held constant)."""
        snr = 10.0
        bw1, bw2 = 100e6, 200e6
        c1 = self.model.calculate_shannon_capacity(snr, bw1)
        c2 = self.model.calculate_shannon_capacity(snr, bw2)
        self.assertAlmostEqual(c2 / c1, 2.0, places=6)

    # --- Guard behaviour ------------------------------------------------

    def test_zero_snr_guard_does_not_raise(self) -> None:
        """SNR=0 must not raise; capacity should be near zero but finite."""
        result = self.model.calculate_shannon_capacity(0.0, self.BANDWIDTH_100MHZ)
        self.assertGreaterEqual(result, 0.0)
        self.assertTrue(math.isfinite(result))

    def test_negative_snr_guard_does_not_raise(self) -> None:
        result = self.model.calculate_shannon_capacity(-5.0, self.BANDWIDTH_100MHZ)
        self.assertGreaterEqual(result, 0.0)
        self.assertTrue(math.isfinite(result))


class TestChannelModelIntegration(unittest.TestCase):
    """
    End-to-end integration tests that chain all four methods together,
    simulating a realistic small-cell link budget.
    """

    def setUp(self) -> None:
        self.model = ChannelModel()

    def test_full_link_budget_short_range(self) -> None:
        """
        50 m LoS link at 60 GHz, 20 dBm Tx, -85 dBm noise, 200 MHz BW.
        Verify the pipeline produces a finite, positive capacity in Mbps.
        """
        distance_m = 50.0
        tx_power_dbm = 20.0
        noise_figure_dbm = -85.0
        bandwidth_hz = 200e6

        p_los = self.model.calculate_los_prob(distance_m)
        self.assertGreater(p_los, 0.5, "Short-range LoS probability should be > 0.5")

        pl = self.model.calculate_pathloss(distance_m, is_los=True)
        snr = self.model.calculate_snr(tx_power_dbm, pl, noise_figure_dbm)
        capacity = self.model.calculate_shannon_capacity(snr, bandwidth_hz)

        self.assertGreater(capacity, 0.0)
        self.assertTrue(math.isfinite(capacity))

    def test_full_link_budget_long_range_nlos(self) -> None:
        """
        500 m NLoS link; capacity must still be positive and finite.
        """
        distance_m = 500.0
        tx_power_dbm = 30.0
        noise_figure_dbm = -80.0
        bandwidth_hz = 100e6

        pl = self.model.calculate_pathloss(distance_m, is_los=False)
        snr = self.model.calculate_snr(tx_power_dbm, pl, noise_figure_dbm)
        capacity = self.model.calculate_shannon_capacity(snr, bandwidth_hz)

        self.assertGreaterEqual(capacity, 0.0)
        self.assertTrue(math.isfinite(capacity))

    def test_nlos_capacity_less_than_los_capacity(self) -> None:
        """At the same distance, NLoS must yield lower capacity than LoS."""
        d = 200.0
        tx, noise, bw = 23.0, -88.0, 100e6

        pl_los = self.model.calculate_pathloss(d, is_los=True)
        pl_nlos = self.model.calculate_pathloss(d, is_los=False)

        cap_los = self.model.calculate_shannon_capacity(
            self.model.calculate_snr(tx, pl_los, noise), bw
        )
        cap_nlos = self.model.calculate_shannon_capacity(
            self.model.calculate_snr(tx, pl_nlos, noise), bw
        )

        self.assertGreaterEqual(cap_los, cap_nlos)


if __name__ == "__main__":
    unittest.main(verbosity=2)
