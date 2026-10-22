import numpy as np
import astropy.units as u
import tellurion as tell

# Fixed epoch for reproducible output; only the 5200 s separation matters.
now = tell.abstime("2026-10-21T00:00:00")
r1 = tell.pvtcart(np.array([-6991, -5118, 7078]), now)
print(r1.cartesian)

r2 = tell.pvtcart(np.array([11326, -6317, -6677]), now + 5200 * u.s)
print(r2.cartesian)

# Propagate each Lambert initial state to the final time with the analytic
# two-body propagator, then compare with the given final position.
tof = 5200 * u.s
np.set_printoptions(precision=10, suppress=False, linewidth=100)
misses = {}
for shortway in (True, False):
    branch = "short" if shortway else "long"
    initial = tell.lambert(r1, r2, shortway=shortway)[0]
    gen = tell.prepare(initial, tof, propagator="keplerian")
    final = tell.propagate(gen, tof, include_init=False, output="pvt")
    miss = (final.position_vector - r2.position_vector).to(u.m)
    misses[branch] = np.linalg.norm(miss.value)
    print(f"{branch} way:")
    print(f"  t2 = {final.time.isot}")
    print(f"  r(t2) = {final.position_vector.to_value(u.km)} km")
    print(f"  r(t2) - r2 = {miss.value} m")
    print(f"  |r(t2) - r2| = {misses[branch]:.3e} m")
