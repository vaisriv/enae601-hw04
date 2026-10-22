import numpy as np
import astropy.units as u
import tellurion as tell

now = tell.abstime(0.0)
r1 = tell.pvtcart(np.array([-6991, -5118, 7078]), now)
print(r1.cartesian)

r2 = tell.pvtcart(np.array([11326, -6317, -6677]), now + 5200*u.s)
print(r2.cartesian)
