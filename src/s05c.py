import numpy as np
import astropy.units as u
import tellurion as tell

r1 = np.array([5740.13268349, 3314.06715, 0.0])
v1 = np.array([-2.75082684, 4.76457184, 5.5016536])

# Two-body (Keplerian) propagation with the state transition matrix. The epoch
# is fixed for reproducible output; it does not affect two-body motion.
epoch = tell.abstime("2026-10-21T00:00:00")
state1 = tell.pvtcart(np.hstack((r1, v1)), epoch)
events = {"altitude": 125.0 * u.km, "eclipse": [], "visibility": [], "stm": True}
gen = tell.prepare(state1, 30 * u.min, events, propagator="keplerian")

state2 = tell.propagate(gen, 30 * u.min, include_init=False, output="pvt")
r2 = state2.position_vector.to_value(u.km)
v2 = state2.velocity_vector.to_value(u.km / u.s)
stm = gen["final"]["stm"]
mu = tell.force.deffe["earthmu"].to_value(u.km**3 / u.s**2)

np.set_printoptions(precision=10, suppress=False, linewidth=100)
print(f"mu = {mu} km^3/s^2")
print(f"t1 = {epoch.isot}")
print(f"t2 = {state2.time.isot}")
print(f"r2 = {r2} km")
print(f"v2 = {v2} km/s")
for name, block in (
    ("A = dr2/dr1 [-]", stm[:3, :3]),
    ("B = dr2/dv1 [s]", stm[:3, 3:]),
    ("C = dv2/dr1 [1/s]", stm[3:, :3]),
    ("D = dv2/dv1 [-]", stm[3:, 3:]),
):
    print(f"{name} =\n{block}")
