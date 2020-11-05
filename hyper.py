# -*- coding: utf-8 -*-
"""
A simple python script to calculate CCSD hyperpolarizability in length using coupled cluster linear response theory.

References: 
- Equations and algorithms from [Koch:1991:3333] and [Crawford:xxxx]
"""

__authors__ = "Monika Kodrycka"
__credits__ = [
    "Ashutosh Kumar", "Daniel G. A. Smith", "Lori A. Burns", "T. D. Crawford"
]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2018-02-20"

import os.path
import sys
#dirname = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(os.path.join(dirname, '../../../Coupled-Cluster/RHF'))
sys.path.append('/home/kordi/Hopper/mine/Coupled-Cluster-Quadratic-Response-Theory')
import numpy as np
np.set_printoptions(precision=15, linewidth=200, suppress=True)
# Import all the coupled cluster utilities
from helper_ccenergy import *
from helper_cchbar import *
from helper_cclambda import *
from helper_ccpert import *

import psi4
from psi4 import constants as pc

psi4.set_memory(int(2e9), False)
psi4.core.set_output_file('output.dat', False)

# can only handle C1 symmetry
mol = psi4.geometry("""
O
H 1 1.8084679
H 1 1.8084679 2 104.5
units bohr
symmetry c1  
no_reorient 
no_com
""")

# setting up SCF options
psi4.set_options({
    'basis': 'STO-3G',
    'scf_type': 'PK',
    'd_convergence': 1e-12,
    'e_convergence': 1e-12,
    'r_convergence' : 1e-12,
})
rhf_e, rhf_wfn = psi4.energy('SCF', return_wfn=True)

print('RHF Final Energy                          % 16.10f\n' % rhf_e)

# Calculate Ground State CCSD energy
ccsd = HelperCCEnergy(mol, rhf_e, rhf_wfn, memory=2)
ccsd.compute_energy(e_conv=1e-12, r_conv=1e-12)

CCSDcorr_E = ccsd.ccsd_corr_e
CCSD_E = ccsd.ccsd_e

print('\nFinal CCSD correlation energy:          % 16.15f' % CCSDcorr_E)
print('Total CCSD energy:                      % 16.15f' % CCSD_E)

# Now that we have T1 and T2 amplitudes, we can construct
# the pieces of the similarity transformed hamiltonian (Hbar).
cchbar = HelperCCHbar(ccsd)

# Calculate Lambda amplitudes using Hbar
cclambda = HelperCCLambda(ccsd, cchbar)
cclambda.compute_lambda(r_conv=1e-12)

# frequency of calculation
omega1 = 0.0656
omega2 = 0.14238
omega_sum = -(omega1+omega2)


cart = ['X', 'Y', 'Z']
Mu = {}
ccpert = {}
ccpert_om1 = {}
ccpert_om2 = {}
ccpert_om_sum = {}

ccpert_om1_2nd = {}
ccpert_om2_2nd = {}
ccpert_om_sum_2nd = {}

polar_AB = {}
hyper_AB = {}

# Obtain AO Dipole Matrices From Mints
dipole_array = ccsd.mints.ao_dipole()

for i in range(0, 3):
    string = "MU_" + cart[i]

    # Transform dipole integrals from AO to MO basis
    Mu[string] = np.einsum('uj,vi,uv', ccsd.npC, ccsd.npC,
                           np.asarray(dipole_array[i]))

    # Initializing the perturbation classs corresponding to dipole perturabtion at the given omega
    # First set
    ccpert_om_sum[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  omega_sum)

    ccpert_om1[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  omega1)

    ccpert_om2[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  omega2)

    #Second set
    ccpert_om_sum_2nd[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  -(-omega1-omega2))

    ccpert_om1_2nd[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  -omega1)

    ccpert_om2_2nd[string] = HelperCCPert(string, Mu[string], ccsd, cchbar, cclambda,
                                  -omega2)

    # Solve X and Y amplitudes corresponding to dipole perturabtion at the given omega
    print('\nsolving right hand perturbed amplitudes for %s\n' % string)
    ccpert_om_sum[string].solve('right', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om_sum[string].solve('left', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om1[string].solve('right', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om1[string].solve('left', r_conv=1e-12)

    print('\nsolving right hand perturbed amplitudes for %s\n' % string)
    ccpert_om2[string].solve('right', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om2[string].solve('left', r_conv=1e-12)

    # Solve a second set of equations for Solve X and Y amplitudes corresponding to dipole perturabtion at the given omega 
    print('\nsolving right hand perturbed amplitudes for %s\n' % string)
    ccpert_om_sum_2nd[string].solve('right', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om_sum_2nd[string].solve('left', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om1_2nd[string].solve('right', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om1_2nd[string].solve('left', r_conv=1e-12)

    print('\nsolving right hand perturbed amplitudes for %s\n' % string)
    ccpert_om2_2nd[string].solve('right', r_conv=1e-12)

    print('\nsolving left hand perturbed amplitudes for %s\n' % string)
    ccpert_om2_2nd[string].solve('left', r_conv=1e-12)


# Refer to eq. 107 of [Koch:1991:3333] for the general form of quadratic response functions.
# For electric dipole polarizabilities, A = mu[x/y/z], B = mu[x/y/z], C = mu[x/y/z]
# Ex. Beta_xyz = <<mu_x;mu_y;nu_z>>, where mu_x = x and mu_y = y, nu_z = z

print("\nComputing <<Mu;Mu;Mu> tensor @ %.4f nm and %.4f nm" %(omega1,omega2))

hyper_AB_1st = np.zeros((3,3,3))
hyper_AB_2nd = np.zeros((3,3,3))
hyper_AB = np.zeros((3,3,3))
for a in range(0, 3):
    str_a = "MU_" + cart[a]
    for b in range(0, 3):
        str_b = "MU_" + cart[b]
        for c in range(0, 3):
            str_c = "MU_" + cart[c]
            hyper_AB_1st[a,b,c] =  HelperCCQuadraticResp(ccsd, cchbar, cclambda, ccpert_om_sum[str_a],
                                 ccpert_om1[str_b],ccpert_om2[str_c]).quadraticresp()
            hyper_AB_2nd[a,b,c] =  HelperCCQuadraticResp(ccsd, cchbar, cclambda, ccpert_om_sum_2nd[str_a],
                                  ccpert_om1_2nd[str_b],ccpert_om2_2nd[str_c]).quadraticresp()
            hyper_AB[a,b,c] = (hyper_AB_1st[a,b,c] + hyper_AB_2nd[a,b,c] )/2


print("\n\nTest, hyperpolarizability:")
print("\Beta_zxx = %10.12lf" %(hyper_AB[2,0,0]))
print("\Beta_xzx = %10.12lf" %(hyper_AB[0,2,0]))
print("\Beta_xxz = %10.12lf" %(hyper_AB[0,0,2]))
print("\Beta_zyy = %10.12lf" %(hyper_AB[2,1,1]))
print("\Beta_yzy = %10.12lf" %(hyper_AB[1,2,1]))
print("\Beta_yyz = %10.12lf" %(hyper_AB[1,1,2]))
print("\Beta_zzz = %10.12lf" %(hyper_AB[2,2,2]))

"""
# Comaprison with PSI4 (if you have near to latest version of psi4)
psi4.set_options({'d_convergence': 1e-10})
psi4.set_options({'e_convergence': 1e-10})
psi4.set_options({'r_convergence': 1e-10})
psi4.set_options({'omega': [694, 'nm']})
psi4.properties('ccsd', properties=['polarizability'])
psi4.compare_values(Isotropic_polar, psi4.variable("CCSD DIPOLE POLARIZABILITY @ 694NM"),  \
6, "CCSD Isotropic Dipole Polarizability @ 694 nm (Length Gauge)") #TEST
"""

#psi4.compare_values(
#    Isotropic_polar, 3.359649777784, 6,
#    "CCSD Isotropic Dipole Polarizability @ 589 nm (Length Gauge)")  #TEST
