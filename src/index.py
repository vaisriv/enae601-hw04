from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MU_EARTH = 398600.0  # km^3/s^2
R_EARTH = 6378.0  # km
TAU = 2.0 * np.pi
ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "outputs" / "text"
FIGURE_DIR = ROOT / "outputs" / "figures"

# Figure ink and categorical series colors (fixed order).
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "legend.frameon": False,
        "savefig.dpi": 200,
    }
)


def heading(problem: str) -> None:
    """Print the heading used to separate each problem's output."""
    print(f"\n------\n {problem}\n------")


def clean(value: float) -> float:
    """Zero out round-off so values like -1e-17 do not print as -0.000000."""
    return 0.0 if abs(value) < 1e-12 else float(value)


def write_output(name: str, lines: list[str]) -> None:
    """Print and save a solution's text output for inclusion in Typst."""
    output = "\n".join(lines) + "\n"
    print(output, end="")
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    (TEXT_DIR / f"s{name}.txt").write_text(output, encoding="utf-8")


def save_figure(fig: plt.Figure, name: str) -> None:
    """Save a solution's figure for inclusion in Typst."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"s{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path.relative_to(ROOT)}")


def vector_lines(
    name: str, vector: np.ndarray, unit: str = "", precision: int = 6
) -> list[str]:
    """Format each component of a vector as its own key/value line."""
    return [
        f"{name}_{axis} = {clean(value):.{precision}f} {unit}".rstrip()
        for axis, value in zip("xyz", vector)
    ]


def rk4(rate, state: np.ndarray, duration: float, steps: int) -> np.ndarray:
    """Integrate state' = rate(state) with fixed-step fourth-order Runge-Kutta."""
    dt = duration / steps
    for _ in range(steps):
        k1 = rate(state)
        k2 = rate(state + dt / 2.0 * k1)
        k3 = rate(state + dt / 2.0 * k2)
        k4 = rate(state + dt * k3)
        state = state + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


def two_body_rate(state: np.ndarray) -> np.ndarray:
    """Two-body equations of motion, r'' = -mu r / r^3."""
    r, v = state[:3], state[3:]
    return np.concatenate((v, -MU_EARTH * r / np.linalg.norm(r) ** 3))


def cw_rate(n: float):
    """Linearized Clohessy-Wiltshire equations (x radial, y along-track)."""

    def rate(state: np.ndarray) -> np.ndarray:
        x, _, z, vx, vy, vz = state
        return np.array(
            [vx, vy, vz, 3.0 * n * n * x + 2.0 * n * vy, -2.0 * n * vx, -n * n * z]
        )

    return rate


def solve_hyperbolic_kepler(e: float, mean_anomaly: float) -> float:
    """Solve e sinh(F) - F = M_h for the hyperbolic eccentric anomaly F."""
    anomaly = mean_anomaly
    for _ in range(50):
        step = (e * np.sinh(anomaly) - anomaly - mean_anomaly) / (
            e * np.cosh(anomaly) - 1.0
        )
        anomaly -= step
        if abs(step) < 1e-14:
            return anomaly
    raise RuntimeError("Hyperbolic Kepler iteration did not converge.")


def cw_stm(n: float, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Clohessy-Wiltshire state transition matrices (Curtis Eq. 7.53)."""
    s, c, nt = np.sin(n * t), np.cos(n * t), n * t
    phi_rr = np.array(
        [[4.0 - 3.0 * c, 0.0, 0.0], [6.0 * (s - nt), 1.0, 0.0], [0.0, 0.0, c]]
    )
    phi_rv = np.array(
        [
            [s / n, 2.0 * (1.0 - c) / n, 0.0],
            [2.0 * (c - 1.0) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, s / n],
        ]
    )
    phi_vr = np.array(
        [[3.0 * n * s, 0.0, 0.0], [6.0 * n * (c - 1.0), 0.0, 0.0], [0.0, 0.0, -n * s]]
    )
    phi_vv = np.array(
        [[c, 2.0 * s, 0.0], [-2.0 * s, 4.0 * c - 3.0, 0.0], [0.0, 0.0, c]]
    )
    return phi_rr, phi_rv, phi_vr, phi_vv


def cw_propagate(
    n: float, t: float, dr0: np.ndarray, dv0: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Relative position and velocity at time t from the CW STM."""
    phi_rr, phi_rv, phi_vr, phi_vv = cw_stm(n, t)
    return phi_rr @ dr0 + phi_rv @ dv0, phi_vr @ dr0 + phi_vv @ dv0


def check_cw(n: float, t: float, dr0: np.ndarray, dv0: np.ndarray) -> float:
    """Compare the STM solution against direct integration of the CW equations."""
    dr, dv = cw_propagate(n, t, dr0, dv0)
    numerical = rk4(cw_rate(n), np.concatenate((dr0, dv0)), t, 6000)
    error = np.linalg.norm(numerical - np.concatenate((dr, dv)))
    assert error < 1e-9, error
    return error


def problem_01() -> None:
    """Curtis 2.45: delta-theta Lagrange coefficients over a 10 min coast."""
    heading("p01")
    r0_vec = np.array([6320.0, 0.0, 7750.0])  # km
    v0_vec = np.array([0.0, 11.0, 0.0])  # km/s
    dt = 10.0 * 60.0  # s

    r0 = np.linalg.norm(r0_vec)
    v0 = np.linalg.norm(v0_vec)
    vr0 = r0_vec @ v0_vec / r0
    h_vec = np.cross(r0_vec, v0_vec)
    h = np.linalg.norm(h_vec)
    h_hat = h_vec / h
    e_vec = np.cross(v0_vec, h_vec) / MU_EARTH - r0_vec / r0
    e = np.linalg.norm(e_vec)
    energy = v0**2 / 2.0 - MU_EARTH / r0
    if e <= 1.0 or energy <= 0.0:
        raise ValueError("Expected a hyperbolic trajectory.")

    # Signed initial true anomaly; v_r0 = 0 places the spacecraft at perigee.
    theta0 = np.arctan2(h_hat @ np.cross(e_vec, r0_vec), e_vec @ r0_vec)

    # Hyperbolic Kepler's equation turns the 10 min coast into delta theta.
    ratio = np.sqrt((e + 1.0) / (e - 1.0))
    anomaly0 = 2.0 * np.arctanh(np.tan(theta0 / 2.0) / ratio)
    mean_rate = MU_EARTH**2 / h**3 * (e * e - 1.0) ** 1.5  # rad/s
    mean_anomaly = e * np.sinh(anomaly0) - anomaly0 + mean_rate * dt
    anomaly = solve_hyperbolic_kepler(e, mean_anomaly)
    theta = 2.0 * np.arctan(ratio * np.tanh(anomaly / 2.0))
    dtheta = theta - theta0

    # Lagrange coefficients in terms of delta theta (Curtis Eqs. 2.152, 2.158).
    c, s = np.cos(dtheta), np.sin(dtheta)
    r = (h**2 / MU_EARTH) / (
        1.0 + (h**2 / (MU_EARTH * r0) - 1.0) * c - h * vr0 / MU_EARTH * s
    )
    f = 1.0 - MU_EARTH * r / h**2 * (1.0 - c)
    g = r * r0 / h * s
    f_dot = (
        MU_EARTH
        / h
        * (1.0 - c)
        / s
        * (MU_EARTH / h**2 * (1.0 - c) - 1.0 / r0 - 1.0 / r)
    )
    g_dot = 1.0 - MU_EARTH * r0 / h**2 * (1.0 - c)
    r_vec = f * r0_vec + g * v0_vec
    v_vec = f_dot * r0_vec + g_dot * v0_vec

    # Independent checks: invariants, the Lagrange identity, the swept angle, and
    # direct numerical integration of the two-body equations.
    np.testing.assert_allclose(np.linalg.norm(r_vec), r, rtol=1e-12)
    np.testing.assert_allclose(f * g_dot - f_dot * g, 1.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(r_vec, v_vec), h_vec, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(v_vec @ v_vec / 2.0 - MU_EARTH / r, energy, rtol=1e-12)
    swept = np.arctan2(h_hat @ np.cross(r0_vec, r_vec), r0_vec @ r_vec)
    np.testing.assert_allclose(swept, dtheta, atol=1e-12)
    numerical = rk4(two_body_rate, np.concatenate((r0_vec, v0_vec)), dt, 6000)
    rk4_error = np.linalg.norm(numerical[:3] - r_vec)
    # The angle to the integrated position checks delta theta without f and g.
    r_numerical = numerical[:3]
    cos_numerical = r0_vec @ r_numerical / (r0 * np.linalg.norm(r_numerical))
    swept_numerical = np.arctan2(
        h_hat @ np.cross(r0_vec, r_numerical), r0_vec @ r_numerical
    )
    np.testing.assert_allclose(swept_numerical, dtheta, atol=1e-9)
    assert rk4_error < 1e-6, rk4_error

    write_output(
        "01a",
        [
            f"r0 = {r0:.6f} km",
            f"v0 = {v0:.6f} km/s",
            f"vr0 = {clean(vr0):.6f} km/s",
            *vector_lines("h", h_vec, "km^2/s"),
            f"h = {h:.6f} km^2/s",
            *vector_lines("e", e_vec, precision=10),
            f"e = {e:.10f}",
            f"energy = {energy:.6f} km^2/s^2",
            f"theta0 = {np.degrees(clean(theta0)):.6f} deg",
            f"Mh rate = {mean_rate:.10f} rad/s",
            f"M_h = {mean_anomaly:.10f}",
            f"F = {anomaly:.10f}",
            f"theta = {np.degrees(theta):.6f} deg",
            f"dtheta = {np.degrees(dtheta):.6f} deg",
            f"r = {r:.6f} km",
            f"f = {f:.10f}",
            f"g = {g:.6f} s",
            f"f_dot = {f_dot:.10f} 1/s",
            f"g_dot = {g_dot:.10f}",
            *vector_lines("r", r_vec, "km"),
            *vector_lines("v", v_vec, "km/s"),
            f"lagrange identity = {f * g_dot - f_dot * g:.12f}",
            f"rk4 position difference = {rk4_error:.3e} km",
        ],
    )
    write_output(
        "01b",
        [
            f"theta0 = {np.degrees(clean(theta0)):.6f} deg",
            f"theta = {np.degrees(theta):.6f} deg",
            f"dtheta = {np.degrees(dtheta):.6f} deg",
            *vector_lines("r_rk4", r_numerical, "km"),
            f"cos dtheta rk4 = {cos_numerical:.10f}",
            f"swept rk4 = {np.degrees(swept_numerical):.6f} deg",
        ],
    )

    # Trajectory in the perifocal frame (p toward perigee, q along the motion).
    p_hat = e_vec / e
    q_hat = np.cross(h_hat, p_hat)
    semilatus = h**2 / MU_EARTH

    def perifocal(angles: np.ndarray) -> np.ndarray:
        radius = semilatus / (1.0 + e * np.cos(angles))
        return np.array([radius * np.cos(angles), radius * np.sin(angles)]) / 1e3

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.add_patch(
        plt.Circle((0.0, 0.0), R_EARTH / 1e3, facecolor=GRID, edgecolor=BASELINE)
    )
    ax.text(0.0, -3.2, "Earth", ha="center", va="center", color=INK_SECONDARY)
    branch = perifocal(np.radians(np.linspace(-70.0, 70.0, 400)))
    ax.plot(*branch, color=MUTED, lw=1.0, ls="--", label="hyperbolic trajectory")
    arc = perifocal(np.linspace(theta0, theta, 200))
    ax.plot(*arc, color=SERIES[0], lw=2.5, label="10 min coast")
    for vector, text, offset in (
        (r0_vec, r"$\mathbf{r}_0$ ($t = 0$, perigee)", (6, -12)),
        (r_vec, r"$\mathbf{r}$ ($t = 10$ min)", (6, 4)),
    ):
        tip = np.array([vector @ p_hat, vector @ q_hat]) / 1e3
        ax.annotate(
            "",
            xy=tip,
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "-|>", "color": INK_SECONDARY, "lw": 1.2},
        )
        ax.annotate(text, tip, xytext=offset, textcoords="offset points", color=INK)
    wedge = np.linspace(theta0, theta, 100)
    ax.plot(8.0 * np.cos(wedge), 8.0 * np.sin(wedge), color=SERIES[1], lw=1.5)
    middle = (theta0 + theta) / 2.0
    ax.text(
        10.9 * np.cos(middle),
        10.9 * np.sin(middle),
        rf"$\Delta\theta = {np.degrees(dtheta):.2f}^\circ$",
        color=INK,
        ha="left",
        va="center",
    )
    ax.set_aspect("equal")
    ax.set_xlim(-8.0, 16.0)
    ax.set_ylim(-14.0, 14.0)
    ax.set_xlabel(r"$\hat{\mathbf{p}}$ (toward perigee) [$10^3$ km]")
    ax.set_ylabel(r"$\hat{\mathbf{q}}$ [$10^3$ km]")
    ax.legend(loc="lower left")
    save_figure(fig, "01a")


def problem_02() -> None:
    """Curtis 7.7: CW propagation of a satellite near a space station."""
    heading("p02")
    period = 90.0 * 60.0  # s
    t = 15.0 * 60.0  # s
    n = TAU / period
    dr0 = np.array([1.0, 0.0, 0.0])  # km
    dv0 = np.array([0.0, 10.0, 0.0]) / 1e3  # km/s

    phi_rr, phi_rv, _, _ = cw_stm(n, t)
    dr, dv = cw_propagate(n, t, dr0, dv0)
    distance = np.linalg.norm(dr)
    error = check_cw(n, t, dr0, dv0)

    write_output(
        "02",
        [
            f"n = {n:.10f} rad/s",
            f"nt = {n * t:.10f} rad",
            f"Phi_rr_11 = {phi_rr[0, 0]:.6f}",
            f"Phi_rr_21 = {phi_rr[1, 0]:.6f}",
            f"Phi_rv_12 = {phi_rv[0, 1]:.6f} s",
            f"Phi_rv_22 = {phi_rv[1, 1]:.6f} s",
            *vector_lines("dr", dr, "km"),
            f"distance = {distance:.6f} km",
            *vector_lines("dv", dv * 1e3, "m/s"),
            f"rk4 state difference = {error:.3e}",
        ],
    )

    times = np.linspace(0.0, t, 301)
    path = np.array([cw_propagate(n, tau, dr0, dv0)[0] for tau in times])
    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    ax.plot(path[:, 1], path[:, 0], color=SERIES[0], lw=2.0)
    ax.plot([0.0, dr[1]], [0.0, dr[0]], color=INK_SECONDARY, lw=1.0, ls="--")
    ax.text(
        dr[1] / 2.0 + 0.3,
        dr[0] / 2.0,
        f"{distance:.2f} km",
        color=INK,
        ha="left",
        va="center",
    )
    ax.plot(0.0, 0.0, marker="s", color=INK, ms=7, ls="none")
    ax.annotate(
        "space station",
        (0.0, 0.0),
        xytext=(-8, -4),
        textcoords="offset points",
        ha="right",
        va="top",
        color=INK,
    )
    ax.plot(dr0[1], dr0[0], "o", mfc="white", mec=SERIES[0], mew=2.0, ms=8)
    ax.annotate(
        r"$t = 0$",
        (dr0[1], dr0[0]),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=INK,
    )
    ax.plot(dr[1], dr[0], "o", color=SERIES[0], ms=8)
    ax.annotate(
        r"$t = 15$ min",
        (dr[1], dr[0]),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=INK,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-4.0, 6.0)
    ax.set_ylim(-1.0, 12.5)
    ax.set_xlabel(r"$\delta y$, along-track [km]")
    ax.set_ylabel(r"$\delta x$, radial [km]")
    save_figure(fig, "02")


def problem_03() -> None:
    """Curtis 7.8: retrofire by a leading spacecraft in the same circular orbit."""
    heading("p03")
    period = 2.0 * 3600.0  # s
    t = 30.0 * 60.0  # s
    n = TAU / period
    dr0 = np.array([0.0, 6.0, 0.0])  # km
    # Co-orbiting, so B is at rest in A's CW frame until the in-track retrofire.
    dv0 = np.array([0.0, 0.0, 0.0]) + np.array([0.0, -3.0, 0.0]) / 1e3  # km/s

    phi_rr, phi_rv, phi_vr, phi_vv = cw_stm(n, t)
    dr, dv = cw_propagate(n, t, dr0, dv0)
    distance = np.linalg.norm(dr)
    speed = np.linalg.norm(dv)
    error = check_cw(n, t, dr0, dv0)
    # Before the burn, B is an equilibrium of the CW equations.
    np.testing.assert_allclose(
        cw_propagate(n, t, dr0, np.zeros(3))[0], dr0, rtol=0.0, atol=1e-12
    )

    write_output(
        "03",
        [
            f"n = {n:.10f} rad/s",
            f"nt = {n * t:.10f} rad",
            f"Phi_rr_22 = {phi_rr[1, 1]:.6f}",
            f"Phi_rv_12 = {phi_rv[0, 1]:.6f} s",
            f"Phi_rv_22 = {phi_rv[1, 1]:.6f} s",
            f"Phi_vr_12 = {clean(phi_vr[0, 1]):.6f} 1/s",
            f"Phi_vr_22 = {clean(phi_vr[1, 1]):.6f} 1/s",
            f"Phi_vv_12 = {phi_vv[0, 1]:.6f}",
            f"Phi_vv_22 = {clean(phi_vv[1, 1]):.6f}",
            *vector_lines("dr", dr, "km"),
            f"distance = {distance:.6f} km",
            *vector_lines("dv", dv * 1e3, "m/s"),
            f"speed = {speed * 1e3:.6f} m/s",
            f"rk4 state difference = {error:.3e}",
        ],
    )

    times = np.linspace(0.0, t, 301)
    path = np.array([cw_propagate(n, tau, dr0, dv0)[0] for tau in times])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot(path[:, 1], path[:, 0], color=SERIES[0], lw=2.0)
    ax.plot([0.0, dr[1]], [0.0, dr[0]], color=INK_SECONDARY, lw=1.0, ls="--")
    ax.annotate(
        f"{distance:.2f} km",
        (dr[1] / 2.0, dr[0] / 2.0),
        xytext=(-8, -8),
        textcoords="offset points",
        color=INK,
        ha="right",
        va="top",
    )
    ax.plot(0.0, 0.0, marker="s", color=INK, ms=7, ls="none")
    ax.annotate(
        r"$A$",
        (0.0, 0.0),
        xytext=(-8, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        color=INK,
    )
    ax.plot(dr0[1], dr0[0], "o", mfc="white", mec=SERIES[0], mew=2.0, ms=8)
    ax.annotate(
        r"$B$, $t = 0$ (retrofire)",
        (dr0[1], dr0[0]),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=INK,
    )
    ax.plot(dr[1], dr[0], "o", color=SERIES[0], ms=8)
    ax.annotate(
        r"$B$, $t = 30$ min",
        (dr[1], dr[0]),
        xytext=(-10, -6),
        textcoords="offset points",
        ha="right",
        va="top",
        color=INK,
    )
    # Relative velocity at 30 min, drawn at 0.25 km per m/s.
    arrow = 0.25 * dv * 1e3
    ax.annotate(
        "",
        xy=(dr[1] + arrow[1], dr[0] + arrow[0]),
        xytext=(dr[1], dr[0]),
        arrowprops={"arrowstyle": "-|>", "color": SERIES[1], "lw": 1.8},
    )
    ax.annotate(
        rf"$\delta\mathbf{{v}}$, {speed * 1e3:.2f} m/s",
        (dr[1] + arrow[1], dr[0] + arrow[0]),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        color=INK,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 13.0)
    ax.set_ylim(-9.0, 1.5)
    ax.set_xlabel(r"$\delta y$, along-track [km]")
    ax.set_ylabel(r"$\delta x$, radial [km]")
    save_figure(fig, "03")


def problem_04() -> None:
    """Curtis 7.10: relative speed half an orbit after an along-track launch."""
    heading("p04")
    # Nondimensional: n = 1 and v = 1, so positions are in v/n and time is nt.
    n, v = 1.0, 1.0
    dr0 = np.zeros(3)
    dv0 = np.array([0.0, v, 0.0])

    dr, dv = cw_propagate(n, np.pi / n, dr0, dv0)
    ratio = np.linalg.norm(dv) / v
    np.testing.assert_allclose(dv, [0.0, -7.0 * v, 0.0], atol=1e-12)
    check_cw(n, np.pi / n, dr0, dv0)

    angles = np.linspace(0.0, np.pi, 301)
    states = [cw_propagate(n, angle / n, dr0, dv0) for angle in angles]
    path = np.array([state[0] for state in states]) * n / v
    speeds = np.array([np.linalg.norm(state[1]) for state in states]) / v
    closed_form = np.sqrt(13.0 - 24.0 * np.cos(angles) + 12.0 * np.cos(angles) ** 2)
    np.testing.assert_allclose(speeds, closed_form, rtol=1e-12)
    assert np.argmax(speeds) == len(angles) - 1

    write_output(
        "04",
        [
            f"nt = {np.pi:.10f} rad",
            *vector_lines("dr", dr * n / v),
            *vector_lines("dv", dv / v),
            f"speed ratio = {ratio:.6f}",
        ],
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 3.4), width_ratios=(1.4, 1.0))
    left.plot(path[:, 1], path[:, 0], color=SERIES[0], lw=2.0)
    left.plot(0.0, 0.0, "o", mfc="white", mec=SERIES[0], mew=2.0, ms=8)
    left.annotate(
        r"$nt = 0$, $\delta\mathbf{v}_0 = v\,\hat{\mathbf{j}}$",
        (0.0, 0.0),
        xytext=(-6, -8),
        textcoords="offset points",
        ha="right",
        va="top",
        color=INK,
    )
    end = dr * n / v
    left.plot(end[1], end[0], "o", color=SERIES[0], ms=8)
    left.annotate(
        r"$nt = \pi$",
        (end[1], end[0]),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=INK,
    )
    # Relative velocity at nt = pi, drawn at 0.5 (v/n) per v.
    tip = (end[1] + 0.5 * dv[1] / v, end[0] + 0.5 * dv[0] / v)
    left.annotate(
        "",
        xy=tip,
        xytext=(end[1], end[0]),
        arrowprops={"arrowstyle": "-|>", "color": SERIES[1], "lw": 1.8},
    )
    left.annotate(
        r"$\delta\mathbf{v} = -7v\,\hat{\mathbf{j}}$",
        tip,
        xytext=(0, -8),
        textcoords="offset points",
        ha="left",
        va="top",
        color=INK,
    )
    left.set_aspect("equal")
    left.set_xlim(-13.5, 1.5)
    left.set_ylim(-2.0, 5.5)
    left.set_xlabel(r"$n\,\delta y / v$, along-track")
    left.set_ylabel(r"$n\,\delta x / v$, radial")

    right.plot(angles, speeds, color=SERIES[0], lw=2.0)
    right.plot(np.pi, ratio, "o", color=SERIES[0], ms=8)
    right.annotate(
        rf"$|\delta\mathbf{{v}}| = {ratio:.0f}v$",
        (np.pi, ratio),
        xytext=(-6, 6),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=INK,
    )
    right.set_xticks([0.0, np.pi / 2.0, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
    right.set_ylim(0.0, 8.0)
    right.set_xlabel(r"$nt$ [rad]")
    right.set_ylabel(r"$|\delta\mathbf{v}| / v$")
    fig.tight_layout()
    save_figure(fig, "04")


def main() -> None:
    #########
    # p01 #
    #########
    problem_01()

    #########
    # p02 #
    #########
    problem_02()

    #########
    # p03 #
    #########
    problem_03()

    #########
    # p04 #
    #########
    problem_04()


if __name__ == "__main__":
    main()
