import contextlib
import io
import runpy
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


def sci(value: float, digits: int = 1) -> str:
    """Scientific notation without exponent padding, e.g. 2.5e-6 (for unify)."""
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


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


def matrix_lines(
    name: str, matrix: np.ndarray, unit: str = "", precision: int = 6
) -> list[str]:
    """Format each entry of a matrix as its own key/value line (name_ij)."""
    return [
        f"{name}_{i + 1}{j + 1} = {clean(value):.{precision}f} {unit}".rstrip()
        for i, row in enumerate(matrix)
        for j, value in enumerate(row)
    ]


def run_snippet(name: str) -> dict:
    """Run a standalone Tellurion snippet, saving its printed output for Typst."""
    buffer = io.StringIO()
    # Snippets set their own numpy print options; restore ours afterwards.
    with np.printoptions(), contextlib.redirect_stdout(buffer):
        namespace = runpy.run_path(
            str(ROOT / "src" / f"{name}.py"), run_name="__main__"
        )
    output = buffer.getvalue()
    print(output, end="")
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    (TEXT_DIR / f"{name}.txt").write_text(output, encoding="utf-8")
    return namespace


def solve_kepler_delta_e(mean: float, ecos: float, esin: float) -> float:
    """Solve n dt = dE - (e cos E1) sin dE + (e sin E1)(1 - cos dE) for dE."""
    delta = mean
    for _ in range(50):
        step = (delta - ecos * np.sin(delta) + esin * (1.0 - np.cos(delta)) - mean) / (
            1.0 - ecos * np.cos(delta) + esin * np.sin(delta)
        )
        delta -= step
        if abs(step) < 1e-14:
            return delta
    raise RuntimeError("Elliptic Kepler iteration did not converge.")


def lagrange_propagate(
    r1_vec: np.ndarray, v1_vec: np.ndarray, dt: float, mu: float = MU_EARTH
) -> dict:
    """Two-body propagation with the Lagrange coefficients in Delta E."""
    r1 = np.linalg.norm(r1_vec)
    alpha = 2.0 / r1 - v1_vec @ v1_vec / mu
    if alpha <= 0.0:
        raise ValueError("The Delta E Lagrange coefficients require an ellipse.")
    a = 1.0 / alpha
    sigma1 = r1_vec @ v1_vec / np.sqrt(mu)
    # e cos(E1) and e sin(E1) from the initial state.
    ecos, esin = 1.0 - r1 / a, sigma1 / np.sqrt(a)
    mean = np.sqrt(mu / a**3) * dt
    delta = solve_kepler_delta_e(mean, ecos, esin)
    c, s = np.cos(delta), np.sin(delta)
    r = a + (r1 - a) * c + sigma1 * np.sqrt(a) * s
    f = 1.0 - a / r1 * (1.0 - c)
    g = dt - np.sqrt(a**3 / mu) * (delta - s)
    f_dot = -np.sqrt(mu * a) * s / (r * r1)
    g_dot = 1.0 - a / r * (1.0 - c)
    return {
        "a": a,
        "e": np.hypot(ecos, esin),
        "sigma1": sigma1,
        "ecos": ecos,
        "esin": esin,
        "M": mean,
        "dE": delta,
        "r": r,
        "f": f,
        "g": g,
        "f_dot": f_dot,
        "g_dot": g_dot,
        "r2": f * r1_vec + g * v1_vec,
        "v2": f_dot * r1_vec + g_dot * v1_vec,
    }


def battin_stm(
    r1_vec: np.ndarray, v1_vec: np.ndarray, dt: float, prop: dict, mu: float = MU_EARTH
) -> tuple[np.ndarray, dict]:
    """Battin's two-body STM from the universal functions U1-U5.

    Arora, Russell, Strange, and Ottesen (2015), Section IV.A, Eqs. 38-50.
    """
    r2_vec, v2_vec = prop["r2"], prop["v2"]
    f, g, f_dot, g_dot = (prop[key] for key in ("f", "g", "f_dot", "g_dot"))
    r1, r2 = np.linalg.norm(r1_vec), np.linalg.norm(r2_vec)
    root_mu = np.sqrt(mu)
    alpha = 2.0 / r1 - v1_vec @ v1_vec / mu
    sigma1 = r1_vec @ v1_vec / root_mu
    sigma2 = r2_vec @ v2_vec / root_mu
    chi = alpha * root_mu * dt + sigma2 - sigma1
    u1 = -r2 * r1 * f_dot / root_mu
    u2 = r1 * (1.0 - f)
    u3 = root_mu * (dt - g)
    u4 = u1 * u3 - 0.5 * (u2**2 - alpha * u3**2)
    u5 = (chi**3 / 6.0 - u3) / alpha
    c_bar = (3.0 * u5 - chi * u4 - root_mu * dt * u2) / root_mu

    eye, outer = np.eye(3), np.outer
    dv = v2_vec - v1_vec
    block_a = (
        r2 / mu * outer(dv, dv)
        + (r1 * (1.0 - f) * outer(r2_vec, r1_vec) + c_bar * outer(v2_vec, r1_vec))
        / r1**3
        + f * eye
    )
    block_b = (
        r1 / mu * (1.0 - f) * (outer(r2_vec - r1_vec, v1_vec) - outer(dv, r1_vec))
        + c_bar / mu * outer(v2_vec, v1_vec)
        + g * eye
    )
    rotation = outer(r2_vec, v2_vec) - outer(v2_vec, r2_vec)
    block_c = (
        -outer(dv, r1_vec) / r1**2
        - outer(r2_vec, dv) / r2**2
        + f_dot
        * (
            eye
            - outer(r2_vec, r2_vec) / r2**2
            + outer(rotation @ r2_vec, dv) / (mu * r2)
        )
        - mu * c_bar / (r1**3 * r2**3) * outer(r2_vec, r1_vec)
    )
    block_d = (
        r1 / mu * outer(dv, dv)
        + (r1 * (1.0 - f) * outer(r2_vec, r1_vec) - c_bar * outer(r2_vec, v1_vec))
        / r2**3
        + g_dot * eye
    )
    terms = {
        "alpha": alpha,
        "sigma1": sigma1,
        "sigma2": sigma2,
        "chi": chi,
        "U1": u1,
        "U2": u2,
        "U3": u3,
        "U4": u4,
        "U5": u5,
        "C_bar": c_bar,
    }
    return np.block([[block_a, block_b], [block_c, block_d]]), terms


def finite_difference_stm(
    r1_vec: np.ndarray, v1_vec: np.ndarray, dt: float, mu: float = MU_EARTH
) -> np.ndarray:
    """Central-difference Jacobian of the Lagrange propagation (checks Phi)."""
    state = np.concatenate((r1_vec, v1_vec))
    steps = np.array([1e-3] * 3 + [1e-6] * 3)  # 1 m and 1 mm/s
    phi = np.empty((6, 6))
    for k, step in enumerate(steps):
        offset = np.zeros(6)
        offset[k] = step
        ends = []
        for sign in (1.0, -1.0):
            shifted = state + sign * offset
            prop = lagrange_propagate(shifted[:3], shifted[3:], dt, mu)
            ends.append(np.concatenate((prop["r2"], prop["v2"])))
        phi[:, k] = (ends[0] - ends[1]) / (2.0 * step)
    return phi


def block_errors(difference: np.ndarray, reference: np.ndarray) -> list[float]:
    """Largest entry of each 3x3 block of a difference, relative to the block."""
    errors = []
    for rows in (slice(0, 3), slice(3, 6)):
        for cols in (slice(0, 3), slice(3, 6)):
            errors.append(
                np.max(np.abs(difference[rows, cols]))
                / np.max(np.abs(reference[rows, cols]))
            )
    return errors


def lambert_prussing(
    r1_vec: np.ndarray,
    r2_vec: np.ndarray,
    dt: float,
    shortway: bool = True,
    mu: float = MU_EARTH,
) -> dict:
    """Zero-revolution Lambert solution by iterating on the semimajor axis.

    Lagrange's time equation in the alpha, beta form (Prussing and Conway):
    s = (r1 + r2 + c)/2, sin^2(alpha/2) = s/2a, sin^2(beta/2) = (s - c)/2a.
    """
    r1, r2 = np.linalg.norm(r1_vec), np.linalg.norm(r2_vec)
    chord = np.linalg.norm(r2_vec - r1_vec)
    s = (r1 + r2 + chord) / 2.0
    sign = 1.0 if shortway else -1.0
    # Parabolic (Euler) flight time; shorter flights are hyperbolic.
    t_p = ((r1 + r2 + chord) ** 1.5 - sign * (r1 + r2 - chord) ** 1.5) / (
        6.0 * np.sqrt(mu)
    )
    if dt <= t_p:
        raise ValueError("Flight time is below the parabolic time (hyperbola).")
    # Minimum-energy ellipse: a_min = s/2, alpha = pi.
    a_min = s / 2.0
    beta_c = sign * 2.0 * np.arcsin(np.sqrt((s - chord) / s))
    t_c = (np.pi - (beta_c - np.sin(beta_c))) / np.sqrt(mu / a_min**3)

    def angles(a: float) -> tuple[float, float]:
        alpha = 2.0 * np.arcsin(min(1.0, np.sqrt(s / (2.0 * a))))
        beta = sign * 2.0 * np.arcsin(np.sqrt((s - chord) / (2.0 * a)))
        # Flights longer than t_c take the branch through the vacant focus.
        return (TAU - alpha if dt > t_c else alpha), beta

    def residual(a: float) -> float:
        alpha, beta = angles(a)
        return (alpha - beta - (np.sin(alpha) - np.sin(beta))) / np.sqrt(mu / a**3) - dt

    # Bracket the root in a, then bisect.
    low, high = a_min, 2.0 * a_min
    while np.sign(residual(high)) == np.sign(residual(low)):
        high *= 2.0
    for iterations in range(1, 201):
        middle = 0.5 * (low + high)
        if np.sign(residual(middle)) == np.sign(residual(low)):
            low = middle
        else:
            high = middle
        if high - low <= 1e-14 * high:
            break
    a = 0.5 * (low + high)

    alpha, beta = angles(a)
    delta = alpha - beta
    f = 1.0 - a / r1 * (1.0 - np.cos(delta))
    g = dt - np.sqrt(a**3 / mu) * (delta - np.sin(delta))
    g_dot = 1.0 - a / r2 * (1.0 - np.cos(delta))
    swept = np.arccos(r1_vec @ r2_vec / (r1 * r2))
    return {
        "c": chord,
        "s": s,
        "a_min": a_min,
        "t_p": t_p,
        "beta_c": beta_c,
        "t_c": t_c,
        "a": a,
        "alpha": alpha,
        "beta": beta,
        "dE": delta,
        "dtheta": swept if shortway else TAU - swept,
        "f": f,
        "g": g,
        "g_dot": g_dot,
        "v1": (r2_vec - f * r1_vec) / g,
        "v2": (g_dot * r2_vec - r1_vec) / g,
        "iterations": iterations,
        "residual": residual(a),
    }


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


def problem_05() -> None:
    """Propagation with Lagrange coefficients, Battin's STM, and Tellurion."""
    heading("p05")
    r1_vec = np.array([5740.13268349, 3314.06715, 0.0])  # km
    v1_vec = np.array([-2.75082684, 4.76457184, 5.5016536])  # km/s
    dt = 30.0 * 60.0  # s

    # (a) Lagrange coefficients in terms of Delta E.
    prop = lagrange_propagate(r1_vec, v1_vec, dt)
    r2_vec, v2_vec = prop["r2"], prop["v2"]
    energy = v1_vec @ v1_vec / 2.0 - MU_EARTH / np.linalg.norm(r1_vec)
    np.testing.assert_allclose(np.linalg.norm(r2_vec), prop["r"], rtol=1e-12)
    np.testing.assert_allclose(
        prop["f"] * prop["g_dot"] - prop["f_dot"] * prop["g"], 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.cross(r2_vec, v2_vec), np.cross(r1_vec, v1_vec), rtol=1e-12
    )
    np.testing.assert_allclose(
        v2_vec @ v2_vec / 2.0 - MU_EARTH / prop["r"], energy, rtol=1e-12
    )
    numerical = rk4(two_body_rate, np.concatenate((r1_vec, v1_vec)), dt, 6000)
    rk4_error = np.linalg.norm(numerical[:3] - r2_vec)
    assert rk4_error < 1e-6, rk4_error
    period = TAU * np.sqrt(prop["a"] ** 3 / MU_EARTH)

    write_output(
        "05a",
        [
            f"r1 = {np.linalg.norm(r1_vec):.6f} km",
            f"v1 = {np.linalg.norm(v1_vec):.6f} km/s",
            f"energy = {energy:.6f} km^2/s^2",
            f"a = {prop['a']:.6f} km",
            f"e = {prop['e']:.10f}",
            f"period = {period:.6f} s",
            f"n = {np.sqrt(MU_EARTH / prop['a'] ** 3):.10f} rad/s",
            f"sigma1 = {prop['sigma1']:.10f} km^0.5",
            f"ecos = {prop['ecos']:.10f}",
            f"esin = {prop['esin']:.10f}",
            f"M = {prop['M']:.10f} rad",
            f"dE = {prop['dE']:.10f} rad",
            f"dE_deg = {np.degrees(prop['dE']):.6f} deg",
            f"r2 = {prop['r']:.6f} km",
            f"f = {prop['f']:.10f}",
            f"g = {prop['g']:.6f} s",
            f"f_dot = {prop['f_dot']:.10f} 1/s",
            f"g_dot = {prop['g_dot']:.10f}",
            *vector_lines("r2", r2_vec, "km"),
            *vector_lines("v2", v2_vec, "km/s", precision=9),
            f"dt = {dt:.1f} s",
            f"lagrange identity = {prop['f'] * prop['g_dot'] - prop['f_dot'] * prop['g']:.12f}",
            f"rk4 position difference = {sci(rk4_error, 3)} km",
        ],
    )

    # (b) Battin's two-body STM in universal functions.
    phi, terms = battin_stm(r1_vec, v1_vec, dt, prop)
    fd_errors = block_errors(finite_difference_stm(r1_vec, v1_vec, dt) - phi, phi)
    assert max(fd_errors) < 1e-6, fd_errors
    # A two-body STM is symplectic: Phi^T J Phi = J, so det(Phi) = 1.
    zero, eye = np.zeros((3, 3)), np.eye(3)
    j_matrix = np.block([[zero, eye], [-eye, zero]])
    symplectic_error = np.max(np.abs(phi.T @ j_matrix @ phi - j_matrix))
    assert symplectic_error < 1e-9, symplectic_error
    determinant = np.linalg.det(phi)
    np.testing.assert_allclose(determinant, 1.0, atol=1e-9)
    np.testing.assert_allclose(
        terms["chi"], np.sqrt(prop["a"]) * prop["dE"], rtol=1e-12
    )

    write_output(
        "05b",
        [
            f"alpha = {terms['alpha']:.12f} 1/km",
            f"sigma1 = {terms['sigma1']:.10f} km^0.5",
            f"sigma2 = {terms['sigma2']:.10f} km^0.5",
            f"chi = {terms['chi']:.10f} km^0.5",
            f"U1 = {terms['U1']:.6f} km^0.5",
            f"U2 = {terms['U2']:.6f} km",
            f"U3 = {terms['U3']:.6f} km^1.5",
            f"U4 = {terms['U4']:.6f} km^2",
            f"U5 = {terms['U5']:.6f} km^2.5",
            f"C_bar = {terms['C_bar']:.6f} km s",
            *matrix_lines("A", phi[:3, :3], precision=7),
            *matrix_lines("B", phi[:3, 3:], "s", precision=4),
            *matrix_lines("C", phi[3:, :3] * 1e3, "1e-3/s", precision=7),
            *matrix_lines("D", phi[3:, 3:], precision=7),
            f"finite difference error = {sci(max(fd_errors))}",
            f"symplectic error = {sci(symplectic_error)}",
            f"det = {determinant:.12f}",
        ],
    )

    # (c) Tellurion's two-body propagation and STM, then the source of the
    # differences: Orekit's gravitational parameter.
    tell = run_snippet("s05c")
    mu_tell = tell["mu"]
    dr = np.linalg.norm(tell["r2"] - r2_vec)
    dv = np.linalg.norm(tell["v2"] - v2_vec)
    dphi = block_errors(tell["stm"] - phi, phi)
    # Sensitivity of the final state to mu predicts the position difference.
    step = 1e-3  # km^3/s^2
    plus = lagrange_propagate(r1_vec, v1_vec, dt, MU_EARTH + step)["r2"]
    minus = lagrange_propagate(r1_vec, v1_vec, dt, MU_EARTH - step)["r2"]
    predicted = (plus - minus) / (2.0 * step) * (mu_tell - MU_EARTH)
    prop_tell = lagrange_propagate(r1_vec, v1_vec, dt, mu_tell)
    phi_tell, _ = battin_stm(r1_vec, v1_vec, dt, prop_tell, mu_tell)
    dr_same = np.linalg.norm(tell["r2"] - prop_tell["r2"])
    dv_same = np.linalg.norm(tell["v2"] - prop_tell["v2"])
    dphi_same = block_errors(tell["stm"] - phi_tell, phi_tell)
    np.testing.assert_allclose(tell["r2"] - r2_vec, predicted, atol=1e-6)
    assert dr_same < 1e-6 and dv_same < 1e-9, (dr_same, dv_same)
    assert max(dphi_same) < 1e-8, dphi_same

    write_output(
        "05c-cmp",
        [
            f"mu tellurion = {mu_tell:.4f} km^3/s^2",
            f"mu difference = {mu_tell - MU_EARTH:.4f} km^3/s^2",
            f"mu ratio = {sci((mu_tell - MU_EARTH) / MU_EARTH, 2)}",
            *vector_lines("dr", (tell["r2"] - r2_vec) * 1e3, "m", precision=4),
            f"dr = {dr * 1e3:.4f} m",
            f"dv = {dv * 1e6:.4f} mm/s",
            *vector_lines("predicted", predicted * 1e3, "m", precision=4),
            f"dphi A = {sci(dphi[0])}",
            f"dphi B = {sci(dphi[1])}",
            f"dphi C = {sci(dphi[2])}",
            f"dphi D = {sci(dphi[3])}",
            f"dphi max = {sci(max(dphi))}",
            f"dr same mu = {sci(dr_same * 1e3)} m",
            f"dv same mu = {sci(dv_same * 1e6)} mm/s",
            f"dphi same mu = {sci(max(dphi_same))}",
        ],
    )


def problem_06() -> None:
    """Zero-revolution Lambert problem, both directions."""
    heading("p06")
    r1_vec = np.array([-6991.0, -5118.0, 7078.0])  # km
    r2_vec = np.array([11326.0, -6317.0, -6677.0])  # km
    dt = 5200.0  # s
    r1, r2 = np.linalg.norm(r1_vec), np.linalg.norm(r2_vec)
    branches = {"short": True, "long": False}

    # (a) Semimajor-axis iteration, then the Delta E Lagrange coefficients.
    solutions = {
        branch: lambert_prussing(r1_vec, r2_vec, dt, shortway)
        for branch, shortway in branches.items()
    }
    lines = [
        f"r1 = {r1:.6f} km",
        f"r2 = {r2:.6f} km",
        f"c = {solutions['short']['c']:.6f} km",
        f"s = {solutions['short']['s']:.6f} km",
        f"a_min = {solutions['short']['a_min']:.6f} km",
        f"cos dtheta = {r1_vec @ r2_vec / (r1 * r2):.10f}",
        f"h_z sign = {np.sign(np.cross(r1_vec, r2_vec)[2]):+.0f}",
    ]
    for branch, sol in solutions.items():
        # The solved initial state must reach r2 after dt (Lagrange and RK4).
        prop = lagrange_propagate(r1_vec, sol["v1"], dt)
        np.testing.assert_allclose(prop["r2"], r2_vec, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(prop["v2"], sol["v2"], rtol=0.0, atol=1e-9)
        numerical = rk4(two_body_rate, np.concatenate((r1_vec, sol["v1"])), dt, 20000)
        assert np.linalg.norm(numerical[:3] - r2_vec) < 1e-6
        np.testing.assert_allclose(prop["a"], sol["a"], rtol=1e-12)
        lines += [
            f"t_p_{branch} = {sol['t_p']:.6f} s",
            f"beta_c_{branch} = {np.degrees(sol['beta_c']):.6f} deg",
            f"t_c_{branch} = {sol['t_c']:.6f} s",
            f"dtheta_{branch} = {np.degrees(sol['dtheta']):.6f} deg",
            f"a_{branch} = {sol['a']:.6f} km",
            f"alpha_{branch} = {np.degrees(sol['alpha']):.6f} deg",
            f"beta_{branch} = {np.degrees(sol['beta']):.6f} deg",
            f"dE_{branch} = {np.degrees(sol['dE']):.6f} deg",
            f"f_{branch} = {sol['f']:.10f}",
            f"g_{branch} = {sol['g']:.6f} s",
            f"g_dot_{branch} = {sol['g_dot']:.10f}",
            *vector_lines(f"v1_{branch}", sol["v1"], "km/s", precision=9),
            *vector_lines(f"v2_{branch}", sol["v2"], "km/s", precision=9),
            f"iterations_{branch} = {sol['iterations']}",
            f"residual_{branch} = {sci(abs(sol['residual']))} s",
        ]
    write_output("06a", lines)

    # (b) Closest approach on each arc: perigee if the arc passes through it,
    # otherwise the nearer endpoint.
    lines = []
    arcs = {}
    times = np.linspace(0.0, dt, 2001)
    for branch, sol in solutions.items():
        h_vec = np.cross(r1_vec, sol["v1"])
        h_hat = h_vec / np.linalg.norm(h_vec)
        e_vec = np.cross(sol["v1"], h_vec) / MU_EARTH - r1_vec / r1
        e = np.linalg.norm(e_vec)
        theta1 = np.arctan2(h_hat @ np.cross(e_vec, r1_vec), e_vec @ r1_vec) % TAU
        theta2 = theta1 + sol["dtheta"]
        perigee = sol["a"] * (1.0 - e)
        through_perigee = theta2 >= TAU
        r_min = perigee if through_perigee else min(r1, r2)
        # Time from r1 to perigee, from the mean anomaly at r1.
        e1 = 2.0 * np.arctan(np.sqrt((1.0 - e) / (1.0 + e)) * np.tan(theta1 / 2.0))
        mean1 = (e1 - e * np.sin(e1)) % TAU
        t_perigee = (TAU - mean1) / np.sqrt(MU_EARTH / sol["a"] ** 3)
        # Sampled check on the closest approach along the arc.
        arc = np.array([lagrange_propagate(r1_vec, sol["v1"], t)["r2"] for t in times])
        arcs[branch] = arc
        sampled = np.min(np.linalg.norm(arc, axis=1))
        assert r_min <= sampled + 1e-6 and sampled - r_min < 1.0, (r_min, sampled)
        assert through_perigee == (t_perigee < dt)
        lines += [
            f"e_{branch} = {e:.10f}",
            f"r_p_{branch} = {perigee:.6f} km",
            f"theta1_{branch} = {np.degrees(theta1):.6f} deg",
            f"theta2_{branch} = {np.degrees(theta2):.6f} deg",
            f"through perigee_{branch} = {through_perigee}",
            f"t_perigee_{branch} = {t_perigee:.6f} s",
            f"r_min_{branch} = {r_min:.6f} km",
            f"altitude_{branch} = {r_min - R_EARTH:.6f} km",
            f"sampled r_min_{branch} = {sampled:.6f} km",
        ]
    write_output("06b", lines)

    # Both arcs in the transfer plane (x along r1, z along r1 x r2).
    x_hat = r1_vec / r1
    z_hat = np.cross(r1_vec, r2_vec)
    z_hat /= np.linalg.norm(z_hat)
    y_hat = np.cross(z_hat, x_hat)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.add_patch(
        plt.Circle((0.0, 0.0), R_EARTH / 1e3, facecolor=GRID, edgecolor=BASELINE)
    )
    ax.text(0.0, 0.0, "Earth", ha="center", va="center", color=INK_SECONDARY)
    for (branch, arc), color in zip(arcs.items(), SERIES):
        x, y = arc @ x_hat / 1e3, arc @ y_hat / 1e3
        ax.plot(x, y, color=color, lw=2.0, label=f"{branch} way")
        middle = len(x) // 2
        ax.annotate(
            "",
            xy=(x[middle + 1], y[middle + 1]),
            xytext=(x[middle - 1], y[middle - 1]),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8},
        )
        closest = np.argmin(np.hypot(x, y))
        ax.plot(x[closest], y[closest], "o", color=color, ms=6)
        ax.annotate(
            rf"$r_{{\min}} = {np.hypot(x[closest], y[closest]) * 1e3:.0f}$ km",
            (x[closest], y[closest]),
            xytext=(8, -4) if branch == "long" else (8, -10),
            textcoords="offset points",
            ha="left",
            va="top",
            color=INK,
        )
    for vector, text, offset in (
        (r1_vec, r"$\mathbf{r}_1$ ($t = 0$)", (8, 0)),
        (r2_vec, r"$\mathbf{r}_2$ ($t = 5200$ s)", (8, 0)),
    ):
        point = (vector @ x_hat / 1e3, vector @ y_hat / 1e3)
        ax.plot(*point, "s", color=INK, ms=6)
        ax.annotate(
            text,
            point,
            xytext=offset,
            textcoords="offset points",
            ha="left",
            va="center",
            color=INK,
        )
    ax.set_aspect("equal")
    ax.set_xlabel(r"along $\hat{\mathbf{r}}_1$ [$10^3$ km]")
    ax.set_ylabel(r"in-plane normal [$10^3$ km]")
    ax.legend(loc="lower left")
    save_figure(fig, "06b")

    # (c) Tellurion's Lambert solver (Orekit), compared with the class method.
    tell = run_snippet("s06c")
    mu_tell = tell["mu"]
    lines = [f"mu tellurion = {mu_tell:.4f} km^3/s^2"]
    dv_all, dv_same_all = [], []
    for branch, shortway in branches.items():
        v1_tell, v2_tell = tell["velocities"][branch]
        same = lambert_prussing(r1_vec, r2_vec, dt, shortway, mu_tell)
        dv1 = np.linalg.norm(v1_tell - solutions[branch]["v1"])
        dv2 = np.linalg.norm(v2_tell - solutions[branch]["v2"])
        dv1_same = np.linalg.norm(v1_tell - same["v1"])
        dv2_same = np.linalg.norm(v2_tell - same["v2"])
        assert max(dv1_same, dv2_same) < 1e-9, (dv1_same, dv2_same)
        dv_all += [dv1, dv2]
        dv_same_all += [dv1_same, dv2_same]
        lines += [
            *vector_lines(
                f"dv1_{branch}", (v1_tell - solutions[branch]["v1"]) * 1e6, "mm/s", 4
            ),
            f"dv1_{branch} = {dv1 * 1e6:.4f} mm/s",
            f"dv2_{branch} = {dv2 * 1e6:.4f} mm/s",
            f"dv1 same mu_{branch} = {sci(dv1_same * 1e6)} mm/s",
            f"dv2 same mu_{branch} = {sci(dv2_same * 1e6)} mm/s",
            f"da mu_{branch} = {same['a'] - solutions[branch]['a']:.6f} km",
        ]
    write_output(
        "06c-cmp",
        lines
        + [
            f"dv max = {max(dv_all) * 1e6:.4f} mm/s",
            f"dv same mu max = {sci(max(dv_same_all) * 1e6)} mm/s",
        ],
    )

    # (d) Tellurion's analytic two-body propagation of each Lambert state.
    tell = run_snippet("s06d")
    lines = []
    for branch in branches:
        miss = tell["misses"][branch]
        assert miss < 1e-3, miss
        # Mixing gravitational parameters: our mu = 398600 solution flown with
        # Orekit's mu misses r2 by the size of the mu sensitivity.
        mixed = lagrange_propagate(r1_vec, solutions[branch]["v1"], dt, mu_tell)
        lines += [
            f"miss_{branch} = {sci(miss)} m",
            f"mixed miss_{branch} = {np.linalg.norm(mixed['r2'] - r2_vec) * 1e3:.4f} m",
        ]
    write_output("06d-cmp", lines)


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

    #########
    # p05 #
    #########
    problem_05()

    #########
    # p06 #
    #########
    problem_06()


if __name__ == "__main__":
    main()
