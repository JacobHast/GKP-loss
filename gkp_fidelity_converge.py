from gkp_loss import *
import qutip


cutoff = 200
rdB_axis = np.linspace(0.01, 20, 10)


a0 = [1, 0, 0, 1]
a1 = [1, 0, 0, 0.9]

F = np.zeros(len(rdB_axis))
for i, rdB in enumerate(rdB_axis):
    print(i)
    state0 = GKP_dm_qutip(cutoff, rdB, a0)
    state1 = GKP_dm_qutip(cutoff, rdB, a1)

    F[i] = qutip.fidelity(state0, state1)

plt.plot(rdB_axis, F**2)
plt.hlines(0.5 * (1 + 0.9), 0, 20)
