from __future__ import annotations

import numpy as np

from spinecho_sim.state import CoherentSpin, Spin, expectation_values

if __name__ == "__main__":
    # order of state components is m_s=[j, j-1, ..., -j+1, -j]
    test = Spin.from_iter([CoherentSpin(theta=np.pi, phi=0)])
    print(test.momentum_states)
    print(expectation_values(test))
