"""
entities.py

Physical environment entities for the mmWave IAB small-cell placement
simulation.  Defines the three core domain objects: User, IABNode, and
CityGrid.
"""

from __future__ import annotations

from typing import List

import numpy as np


class User:
    """
    Represents a mobile user terminal (UE) located on the simulation grid.

    A User is a passive entity that occupies a fixed (x, y) position within
    the CityGrid and declares a minimum downlink throughput requirement.  The
    DRL environment uses this demand to evaluate whether an IAB deployment
    satisfies per-user quality-of-service (QoS) constraints.

    Attributes
    ----------
    x : float
        Horizontal position of the user on the grid [m].
    y : float
        Vertical position of the user on the grid [m].
    data_demand_mbps : float
        Minimum required downlink data rate [Mbps].  Defaults to 100.0 Mbps,
        consistent with a typical enhanced Mobile Broadband (eMBB) target for
        60 GHz mmWave deployments.
    """

    def __init__(
        self,
        x: float,
        y: float,
        data_demand_mbps: float = 100.0,
    ) -> None:
        """
        Initialise a User at grid coordinates (x, y).

        Parameters
        ----------
        x : float
            Horizontal position [m].  Must be a finite real number.
        y : float
            Vertical position [m].  Must be a finite real number.
        data_demand_mbps : float, optional
            Required downlink throughput [Mbps].  Must be strictly positive.
            Defaults to 100.0 Mbps.

        Raises
        ------
        ValueError
            If ``data_demand_mbps`` is not strictly positive.
        """
        if data_demand_mbps <= 0.0:
            raise ValueError(
                f"data_demand_mbps must be strictly positive, got {data_demand_mbps}."
            )

        self.x: float = float(x)
        self.y: float = float(y)
        self.data_demand_mbps: float = float(data_demand_mbps)

    def __repr__(self) -> str:
        return (
            f"User(x={self.x:.2f}, y={self.y:.2f}, "
            f"demand={self.data_demand_mbps:.1f} Mbps)"
        )


class IABNode:
    """
    Represents an Integrated Access and Backhaul (IAB) small-cell node.

    An IABNode is either a **donor** node — connected to the core network
    via a fibre backhaul link and therefore not capacity-constrained by a
    wireless hop — or a **relay** node that receives its backhaul wirelessly
    from an upstream IAB donor or relay.

    The backhaul feasibility constraint requires that the total flow arriving
    at the node (``flow_in_capacity``) is sufficient to satisfy the aggregate
    downlink demand it must serve (``flow_out_demand``).  This constraint is
    central to the DRL reward signal.

    Attributes
    ----------
    x : float
        Horizontal position of the node on the grid [m].
    y : float
        Vertical position of the node on the grid [m].
    is_donor : bool
        ``True`` if this node has a fibre backhaul (donor); ``False`` if it
        relies on a wireless IAB backhaul link (relay).
    flow_in_capacity : float
        Maximum downlink throughput that can be delivered to this node from
        its upstream link [Mbps].  For donor nodes this represents the fibre
        capacity; for relay nodes it is the wireless backhaul capacity.
    flow_out_demand : float
        Aggregate downlink throughput that this node must deliver to all
        associated users and downstream relay nodes [Mbps].
    """

    def __init__(
        self,
        x: float,
        y: float,
        is_donor: bool,
        flow_in_capacity: float,
        flow_out_demand: float,
    ) -> None:
        """
        Initialise an IABNode at grid coordinates (x, y).

        Parameters
        ----------
        x : float
            Horizontal position [m].
        y : float
            Vertical position [m].
        is_donor : bool
            Set ``True`` for a fibre-connected donor node, ``False`` for a
            wireless relay node.
        flow_in_capacity : float
            Upstream capacity available to this node [Mbps].  Must be
            non-negative.
        flow_out_demand : float
            Total outgoing flow demand this node must serve [Mbps].  Must be
            non-negative.

        Raises
        ------
        ValueError
            If ``flow_in_capacity`` or ``flow_out_demand`` is negative.
        """
        if flow_in_capacity < 0.0:
            raise ValueError(
                f"flow_in_capacity must be non-negative, got {flow_in_capacity}."
            )
        if flow_out_demand < 0.0:
            raise ValueError(
                f"flow_out_demand must be non-negative, got {flow_out_demand}."
            )

        self.x: float = float(x)
        self.y: float = float(y)
        self.is_donor: bool = bool(is_donor)
        self.flow_in_capacity: float = float(flow_in_capacity)
        self.flow_out_demand: float = float(flow_out_demand)

    def check_backhaul_constraint(self) -> bool:
        """
        Evaluate whether the backhaul capacity constraint is satisfied.

        The IAB backhaul feasibility condition requires that the incoming
        capacity of this node is at least equal to the total outgoing demand:

            flow_in_capacity >= flow_out_demand

        For donor nodes this constraint is almost always satisfied by design
        (fibre link); for relay nodes it is the critical bottleneck that the
        DRL agent must respect during AP placement.

        Returns
        -------
        bool
            ``True`` if the backhaul constraint is satisfied
            (``flow_in_capacity >= flow_out_demand``); ``False`` otherwise.
        """
        return self.flow_in_capacity >= self.flow_out_demand

    def __repr__(self) -> str:
        node_type: str = "Donor" if self.is_donor else "Relay"
        constraint_ok: bool = self.check_backhaul_constraint()
        return (
            f"IABNode(type={node_type}, x={self.x:.2f}, y={self.y:.2f}, "
            f"flow_in={self.flow_in_capacity:.1f} Mbps, "
            f"flow_out={self.flow_out_demand:.1f} Mbps, "
            f"constraint_satisfied={constraint_ok})"
        )


class CityGrid:
    """
    Represents the 2-D simulation map on which users and IAB nodes are placed.

    CityGrid acts as the spatial container for the DRL environment.  It
    defines the valid deployment area (a rectangle of ``width`` × ``height``
    metres), holds the current set of active User objects, and provides
    factory utilities for populating that set with randomly distributed users.

    Coordinates within the grid are defined on the half-open interval
    ``[0, width)`` × ``[0, height)`` so that all generated positions lie
    strictly inside the boundary.

    Attributes
    ----------
    width : float
        Horizontal extent of the simulation area [m].
    height : float
        Vertical extent of the simulation area [m].
    users : List[User]
        Collection of User objects currently placed on the grid.  Populated
        by ``generate_users``; empty on construction.
    """

    def __init__(self, width: float, height: float) -> None:
        """
        Initialise an empty CityGrid of the given dimensions.

        Parameters
        ----------
        width : float
            Horizontal extent of the grid [m].  Must be strictly positive.
        height : float
            Vertical extent of the grid [m].  Must be strictly positive.

        Raises
        ------
        ValueError
            If ``width`` or ``height`` is not strictly positive.
        """
        if width <= 0.0:
            raise ValueError(f"width must be strictly positive, got {width}.")
        if height <= 0.0:
            raise ValueError(f"height must be strictly positive, got {height}.")

        self.width: float = float(width)
        self.height: float = float(height)
        self.users: List[User] = []

    def generate_users(
        self,
        num_users: int,
        data_demand_mbps: float = 100.0,
        seed: int | None = None,
    ) -> List[User]:
        """
        Populate the grid with randomly positioned User objects.

        Each user is placed at an independent, uniformly distributed (x, y)
        position drawn from the half-open interval
        ``[0, width)`` × ``[0, height)``, ensuring all coordinates lie
        strictly within the grid boundary.  The existing ``self.users`` list
        is replaced entirely on each call.

        NumPy is used for vectorised coordinate generation:

            x_coords = np.random.uniform(0.0, width,  num_users)
            y_coords = np.random.uniform(0.0, height, num_users)

        Parameters
        ----------
        num_users : int
            Number of User objects to generate.  Must be strictly positive.
        data_demand_mbps : float, optional
            Throughput demand assigned to every generated user [Mbps].
            Defaults to 100.0 Mbps.
        seed : int or None, optional
            Seed for the NumPy random number generator.  Pass an integer for
            reproducible layouts; ``None`` (default) uses the global RNG state.

        Returns
        -------
        List[User]
            The newly generated list of User objects, also stored in
            ``self.users``.

        Raises
        ------
        ValueError
            If ``num_users`` is not strictly positive.
        """
        if num_users <= 0:
            raise ValueError(
                f"num_users must be strictly positive, got {num_users}."
            )

        rng: np.random.Generator = np.random.default_rng(seed)

        x_coords: np.ndarray = rng.uniform(0.0, self.width, size=num_users)
        y_coords: np.ndarray = rng.uniform(0.0, self.height, size=num_users)

        # Clip defensively to guarantee strict boundary adherence despite any
        # floating-point rounding at the open upper bound.
        x_coords = np.clip(x_coords, 0.0, np.nextafter(self.width, 0.0))
        y_coords = np.clip(y_coords, 0.0, np.nextafter(self.height, 0.0))

        self.users = [
            User(x=float(x), y=float(y), data_demand_mbps=data_demand_mbps)
            for x, y in zip(x_coords, y_coords)
        ]
        return self.users

    def __repr__(self) -> str:
        return (
            f"CityGrid(width={self.width:.1f} m, height={self.height:.1f} m, "
            f"users={len(self.users)})"
        )
