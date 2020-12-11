import numpy as np
import matplotlib.pyplot as plt

from some_random_function import Himmelblau, Himmelblau_grad

np.random.seed(0)  # Just to make sure it is reproducible


# Change these badboys to whatever you want
function_to_hockey = Himmelblau
gradient_to_hockey = Himmelblau_grad

# Problem dimension
dimensions = 2

# Initial model
m_initial = np.random.randn(dimensions, 1) - 5

# Perturbation
m_perturbation = np.ones_like(m_initial) + np.random.rand(*m_initial.shape)

# Do we want to see some contours?
make_contours = True

# 2D Plot domain
x0, x1, y0, y1 = -5, 5, -5, 5

# plt.scatter(
#     m_initial[0, 0],
#     m_initial[1, 0],
#     label="Initial model",
#     marker="v",
#     s=250,
#     zorder=10,
# )
# plt.plot(
#     [m_perturbation[0, 0] * -10, m_initial[0, 0], m_perturbation[0, 0] * 10],
#     [m_perturbation[1, 0] * -10, m_initial[1, 0], m_perturbation[1, 0] * 10],
#     "--k",
#     label="perturbation direction",
#     zorder=1,
# )
# plt.scatter(
#     [m_perturbation[0, 0] * (i - 10) for i in range(20)],
#     [m_perturbation[1, 0] * (i - 10) for i in range(20)],
#     marker="o",
#     label="perturbation steps of 1",
#     zorder=5,
# )
# plt.legend()
# plt.xlabel("Dimension 0")
# plt.ylabel("Dimension 1")
# plt.xlim([x0, x1])
# plt.ylim([y0, y1])

# if make_contours:
#     m0v, m1v = np.meshgrid(np.arange(x0, x1, 0.1), np.arange(y0, y1, 0.1))
#     grid = np.vstack([m0v[None, :, :], m1v[None, :, :]])
#     some_values = function_to_hockey(grid)
#     plt.contour(some_values, extent=(x0, x1, y0, y1), levels=20)

# plt.show()

misfit_initial = function_to_hockey(m_initial)
gradient_initial = gradient_to_hockey(m_initial)

epsilons = np.logspace(-25, 5, 31)
actual_misfits = np.empty_like(epsilons)
predicted_misfits = np.empty_like(epsilons)
relative_errors = np.empty_like(epsilons)

for i_epsilon, epsilon in enumerate(epsilons):

    # Perturb model
    m_perturbed = m_initial.copy() + epsilon * m_perturbation.copy()

    # Predict misfit
    predicted_misfits[i_epsilon] = (
        epsilon * (gradient_initial.T @ m_perturbation) + misfit_initial
    )

    # Calculate actual misfit
    actual_misfits[i_epsilon] = function_to_hockey(m_perturbed)


relative_misfit_errors = (
    np.abs(predicted_misfits - actual_misfits) / actual_misfits
)


plt.semilogy(relative_misfit_errors)
plt.ylabel("relative error in misfit prediction")
plt.xlabel("epsilon")
plt.show()
