import numpy as np
import astropy.units as u
import tellurion as tell

# Fixed epoch for reproducible output; only the 5200 s separation matters.
now = tell.abstime("2026-10-21T00:00:00")
r1 = tell.pvtcart(np.array([-6991, 5118, 7078]), now)
print(r1.cartesian)

r2 = tell.pvtcart(np.array([11326, -6317, -6677]), now + 5200 * u.s)
print(r2.cartesian)

# Zero-revolution Lambert solutions in both directions.
mu = tell.force.deffe["earthmu"].to_value(u.km**3 / u.s**2)
velocities = {}
for shortway in (True, False):
    branch = "short" if shortway else "long"
    solution = tell.lambert(r1, r2, shortway=shortway)
    v1 = solution[0].velocity_vector.to_value(u.km / u.s)
    v2 = solution[1].velocity_vector.to_value(u.km / u.s)
    velocities[branch] = (v1, v2)

np.set_printoptions(precision=10, suppress=False, linewidth=100)
print(f"mu = {mu} km^3/s^2")
for branch, (v1, v2) in velocities.items():
    print(f"{branch} way:")
    print(f"  v1 = {v1} km/s")
    print(f"  v2 = {v2} km/s")
