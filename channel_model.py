"""
channel_model.py

mmWave channel model for 60 GHz small-cell access point placement simulation.
Implements LoS/NLoS pathloss equations based on Zhang et al. (2024) and
3GPP TR 38.901 Urban Micro-cell (UMi) propagation models.
"""

import numpy as np


class ChannelModel:
    """
    Models mmWave propagation characteristics at 60 GHz for an urban
    micro-cell (UMi) deployment scenario.

    All pathloss calculations follow the 3GPP TR 38.901 UMi Street Canyon
    model, augmented with atmospheric impairments (rain and oxygen absorption)
    specific to the 60 GHz band as characterised by Zhang et al. (2024).

    Class Constants
    ---------------
    CARRIER_FREQ_GHZ : float
        Carrier frequency in GHz (60 GHz mmWave band).
    RAIN_ATTENUATION_DB_PER_KM : float
        Rain-induced specific attenuation at 60 GHz: 12.6 dB/km.
    OXYGEN_ABSORPTION_DB_PER_KM : float
        Oxygen absorption specific attenuation at 60 GHz: 16.0 dB/km.
    NOISE_FLOOR_DBM : float
        Thermal noise floor at room temperature (290 K) over 1 Hz bandwidth:
        -174 dBm/Hz.
    """

    CARRIER_FREQ_GHZ: float = 60.0
    RAIN_ATTENUATION_DB_PER_KM: float = 12.6
    OXYGEN_ABSORPTION_DB_PER_KM: float = 16.0
    NOISE_FLOOR_DBM: float = -174.0

    # ------------------------------------------------------------------ #
    #  LoS Probability                                                     #
    # ------------------------------------------------------------------ #

    def calculate_los_prob(self, distance_m: float) -> float:
        """
        Calculate the Line-of-Sight (LoS) probability for a UMi deployment.

        Implements the 3GPP TR 38.901 Table 7.4.2-1 UMi Street Canyon model:

            P_LoS(d) = min(18 / d, 1) * (1 - exp(-d / 36)) + exp(-d / 36)

        where ``d`` is the 2-D separation between transmitter and receiver in
        metres.  The first term captures the probability that no obstacle
        blocks the direct path over short distances; the second exponential
        term models the increasing likelihood that the receiver lies in the
        shadow of a building cluster as distance grows.

        A guard value of 1 m is applied so that the expression remains finite
        when ``distance_m`` is zero or negative.

        Parameters
        ----------
        distance_m : float
            2-D Euclidean distance between transmitter and receiver [m].

        Returns
        -------
        float
            LoS probability in the range [0, 1].

        References
        ----------
        3GPP TR 38.901 V17.0.0, Table 7.4.2-1 (UMi Street Canyon).
        """
        d: float = np.maximum(distance_m, 1.0)

        term_a: float = np.minimum(18.0 / d, 1.0) * (1.0 - np.exp(-d / 36.0))
        term_b: float = np.exp(-d / 36.0)

        p_los: float = term_a + term_b
        return float(np.clip(p_los, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Pathloss                                                            #
    # ------------------------------------------------------------------ #

    def calculate_pathloss(self, distance_m: float, is_los: bool) -> float:
        """
        Calculate total pathloss at 60 GHz including atmospheric impairments.

        **Base pathloss** follows the 3GPP TR 38.901 UMi Street Canyon model
        (Zhang et al., 2024):

        *LoS condition* (free-space-like, breakpoint distance >> link range):

            PL_LoS = 32.4 + 21 * log10(d) + 20 * log10(f_c)   [dB]

        where ``d`` [m] is the 3-D link distance and ``f_c`` [GHz] is the
        carrier frequency.  This is derived from the Friis free-space model
        with a distance exponent of 2.1 that accounts for mild waveguiding
        along street canyons.

        *NLoS condition* (dominant single-reflection / diffraction paths):

            PL_NLoS = max(PL_LoS, 35.3 * log10(d) + 22.4 + 21.3 * log10(f_c))   [dB]

        The NLoS exponent of 3.53 reflects the additional loss from scattering
        and diffraction in cluttered urban environments.

        **Atmospheric attenuation** specific to 60 GHz is appended:

            A_atm = (alpha_rain + alpha_O2) * d_km   [dB]

            alpha_rain = 12.6 dB/km  (ITU-R P.838 heavy rain, ~50 mm/h)
            alpha_O2   = 16.0 dB/km  (ITU-R P.676 oxygen resonance at 60 GHz)

        **Total pathloss**:

            PL_total = PL_base + A_atm   [dB]

        Parameters
        ----------
        distance_m : float
            3-D link distance between transmitter and receiver [m].
        is_los : bool
            ``True`` to apply the LoS model; ``False`` for the NLoS model.

        Returns
        -------
        float
            Total pathloss in dB (always positive).

        References
        ----------
        3GPP TR 38.901 V17.0.0, Table 7.4.1-1 (UMi Street Canyon).
        ITU-R P.838-3 (rain attenuation).
        ITU-R P.676-12 (oxygen absorption).
        Zhang et al. (2024), mmWave small-cell propagation characterisation.
        """
        d: float = np.maximum(distance_m, 1.0)
        fc: float = self.CARRIER_FREQ_GHZ

        # --- Base LoS pathloss -------------------------------------------
        pl_los: float = 32.4 + 21.0 * np.log10(d) + 20.0 * np.log10(fc)

        # --- Base NLoS pathloss (lower-bounded by LoS) -------------------
        pl_nlos_candidate: float = (
            35.3 * np.log10(d) + 22.4 + 21.3 * np.log10(fc)
        )
        pl_nlos: float = float(np.maximum(pl_los, pl_nlos_candidate))

        pl_base: float = pl_los if is_los else pl_nlos

        # --- Atmospheric attenuation [dB/km] applied over link distance --
        d_km: float = d / 1000.0
        attenuation_db_per_km: float = (
            self.RAIN_ATTENUATION_DB_PER_KM + self.OXYGEN_ABSORPTION_DB_PER_KM
        )
        atmospheric_loss_db: float = attenuation_db_per_km * d_km

        return float(pl_base + atmospheric_loss_db)

    # ------------------------------------------------------------------ #
    #  SNR                                                                 #
    # ------------------------------------------------------------------ #

    def calculate_snr(
        self,
        tx_power_dbm: float,
        pathloss_db: float,
        noise_figure_dbm: float,
    ) -> float:
        """
        Calculate the received Signal-to-Noise Ratio (SNR) in linear scale.

        The received signal power at the receiver is:

            P_rx [dBm] = P_tx [dBm] - PL [dB]

        The effective noise power (thermal noise floor plus receiver noise
        figure) is given directly as ``noise_figure_dbm``, which represents
        the total in-band noise power referenced to the receiver input:

            N_eff [dBm] = noise_floor [dBm/Hz] + 10*log10(B) + NF [dB]

        This quantity must be pre-computed by the caller for the specific
        bandwidth and hardware noise figure.  The SNR in dB is:

            SNR_dB = P_rx - N_eff

        Converting to linear scale:

            SNR_linear = 10 ^ (SNR_dB / 10)

        Parameters
        ----------
        tx_power_dbm : float
            Transmit power of the access point [dBm].
        pathloss_db : float
            Total link pathloss as returned by ``calculate_pathloss`` [dB].
        noise_figure_dbm : float
            Effective noise power at the receiver input [dBm].  This must
            include thermal noise, bandwidth scaling, and hardware noise
            figure contributions.

        Returns
        -------
        float
            SNR in linear (power) scale [dimensionless, >= 0].
        """
        rx_power_dbm: float = tx_power_dbm - pathloss_db
        snr_db: float = rx_power_dbm - noise_figure_dbm
        snr_linear: float = float(10.0 ** (snr_db / 10.0))
        return snr_linear

    # ------------------------------------------------------------------ #
    #  Shannon Capacity                                                    #
    # ------------------------------------------------------------------ #

    def calculate_shannon_capacity(
        self,
        snr_linear: float,
        bandwidth_hz: float,
    ) -> float:
        """
        Compute the theoretical maximum data rate using the Shannon-Hartley
        theorem, expressed in Megabits per second (Mbps).

        The Shannon-Hartley theorem gives the channel capacity of an additive
        white Gaussian noise (AWGN) channel as:

            C = B * log2(1 + SNR)   [bits/s]

        where:
            ``B``   = channel bandwidth [Hz]
            ``SNR`` = signal-to-noise ratio [linear, dimensionless]

        Converted to Mbps:

            C_Mbps = C / 1e6   [Mbps]

        This represents an upper bound on the achievable spectral efficiency
        and does not account for practical modulation and coding overhead.

        Parameters
        ----------
        snr_linear : float
            Received SNR in linear (power) scale as returned by
            ``calculate_snr``.  Values <= 0 are clipped to a minimum of
            1e-9 to avoid numerical errors in the logarithm.
        bandwidth_hz : float
            Channel bandwidth [Hz] (e.g., 100e6 for 100 MHz).

        Returns
        -------
        float
            Achievable data rate upper bound [Mbps].
        """
        snr_safe: float = float(np.maximum(snr_linear, 1e-9))
        capacity_bps: float = bandwidth_hz * np.log2(1.0 + snr_safe)
        capacity_mbps: float = capacity_bps / 1e6
        return float(capacity_mbps)
