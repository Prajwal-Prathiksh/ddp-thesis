r"""
Calculation of Backward & Forward-in-time Finite-Time Lyapunov Exponent (FTLE)
#####################
References
-----------
    .. [Sun2016] P. N. Sun, A. Colagrossi, S. Marrone, and A. M. Zhang,
    “Detection of Lagrangian Coherent Structures in the SPH framework,” Comput.
    Methods Appl. Mech. Eng., vol. 305, pp. 849-868, 2016,
    doi: 10.1016/j.cma.2016.03.027.
"""
###########################################################################
#
###########################################################################
def calculate_flow_ftle(time_evolution, t_range=[], dim, bounded_flow=True):
    if not bounded_flow:
        raise Exception("FTLE calculation not modelled for inflow/outflow")

def calculate_deformation_gradient():
    