# -*- coding: utf-8 -*-
"""
A simple python script to compute RHF-CCSD linear response function 
for calculating properties like dipole polarizabilities, optical
rotations etc. 

References: 
- Equations and algoriths from [Koch:1991:3333], [Gwaltney:1996:189], 
[Helgaker:2000], and [Crawford:xxxx]

1. A Whirlwind Introduction to Coupled Cluster Response Theory, T.D. Crawford, Private Notes,
   (pdf in the current directory).
2. H. Koch and P. Jørgensen, J. Chem. Phys. Volume 93, pp. 3333-3344 (1991).
3. S. R. Gwaltney, M. Nooijen and R.J. Bartlett, Chemical Physics Letters, 248, pp. 189-198 (1996).
4. Chapter 13, "Molecular Electronic-Structure Theory", Trygve Helgaker, 
   Poul Jørgensen and Jeppe Olsen, John Wiley & Sons Ltd.

"""

__authors__ = "Monika Kodrycka, Ashutosh Kumar"
__credits__ = ["Monika Kodrycka", "Ashutosh Kumar", "Daniel G. A. Smith", "Lori A. Burns", "T. D. Crawford"]

__copyright__ = "(c) 2014-2018, The Psi4NumPy Developers"
__license__ = "BSD-3-Clause"
__date__ = "2017-05-17"

import time
import numpy as np
import psi4
import sys
sys.path.append("../../../Coupled-Cluster/RHF")
from utils import ndot
from utils import helper_diis

class HelperCCPert(object):
    def __init__(self, name, pert, ccsd, hbar, cclambda, omega1):
        """
        Initializes the HelperCCPert object.

        Parameters:
        -----------
        neme: string 
            Perturbation irrep.
        pert: NumpPy array
             Dipole integrals in the MO basis.
        ccsd: ccsd object
             An initialized ccsd object. 
        hbar: hbar object
             An initialized hbar object.  
        cclambda: cclambda object
	     An initialized cclambda object. 
        omega1: float 
	      Frequency of the perturbation.
        """

        time_init = time.time()

        # Grabbing all the info from the wavefunctions passed
        self.pert = pert
        self.name = name
        self.MO = ccsd.MO
        self.ndocc = ccsd.ndocc
        self.nmo = ccsd.nmo
        self.nocc = ccsd.ndocc
        self.nvirt = ccsd.nmo - ccsd.nocc 
        self.mints = ccsd.mints
        self.F = ccsd.F
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2
        self.ttau  =  hbar.ttau
        self.Loovv =  hbar.Loovv
        self.Looov =  hbar.Looov
        self.Lvovv =  hbar.Lvovv
        self.Hov   =  hbar.Hov
        self.Hvv   =  hbar.Hvv
        self.Hoo   =  hbar.Hoo
        self.Hoooo =  hbar.Hoooo
        self.Hvvvv =  hbar.Hvvvv
        self.Hvovv =  hbar.Hvovv
        self.Hooov =  hbar.Hooov
        self.Hovvo =  hbar.Hovvo
        self.Hovov =  hbar.Hovov
        self.Hvvvo =  hbar.Hvvvo
        self.Hovoo =  hbar.Hovoo
        self.l1 = cclambda.l1
        self.l2 = cclambda.l2
        self.omega1 = omega1

        self.slice_o = slice(0, self.nocc)
        self.slice_v = slice(self.nocc, self.nmo)
        self.slice_a = slice(0, self.nmo)
        self.slice_dict = {'o' : self.slice_o, 'v' : self.slice_v,
                           'a' : self.slice_a}

        # Build the denominators from diagonal elements of Hbar and omega
        self.Dia = self.Hoo.diagonal().reshape(-1, 1) - self.Hvv.diagonal()
        self.Dijab = self.Hoo.diagonal().reshape(-1, 1, 1, 1) + self.Hoo.diagonal().reshape(-1, 1, 1) \
                    - self.Hvv.diagonal().reshape(-1, 1) - self.Hvv.diagonal() 
        self.Dia += self.omega1      
        self.Dijab += self.omega1   
        
        # Guesses for X1 and X2 amplitudes (First order perturbed T amplitudes)
        self.x1 = self.build_Avo().swapaxes(0,1)/self.Dia
        self.pertbar_ijab = self.build_Avvoo().swapaxes(0,2).swapaxes(1,3)
        self.x2 = self.pertbar_ijab.copy()
        self.x2 += self.pertbar_ijab.swapaxes(0,1).swapaxes(2,3)
        self.x2 = self.x2/self.Dijab
       
        # Guesses for Y1 and Y2 amplitudes (First order perturbed Lambda amplitudes)
        self.y1 =  2.0 * self.x1.copy() 
        self.y2 =  4.0 * self.x2.copy()    
        self.y2 -= 2.0 * self.x2.swapaxes(2,3)

        # Conventions used :    
        # occ orbitals  : i, j, k, l, m, n
        # virt orbitals : a, b, c, d, e, f
        # all oribitals : p, q, r, s, t, u, v

    def get_MO(self, string):
        """
        Obtains integrals in the MO basis.

        Parameters:
        -----------
           string: string
             String of integral indexes.    
               
	Returns:
        --------  
           MO: Numpy array
             Integrals in the MO basis.
        """  
 
        if len(string) != 4:
            psi4.core.clean()
            raise Exception('get_MO: string %s must have 4 elements.' % string)
        return self.MO[self.slice_dict[string[0]], self.slice_dict[string[1]],
                       self.slice_dict[string[2]], self.slice_dict[string[3]]]

    def get_F(self, indexes):
        """
        Obtains the Fock Matix.

        Parameters:
        -----------
           indexes: string
             String of Fock indexes.    
               
        Returns:
        --------  
           F: NumPy array
             The Fock Matrix in the MO basis.
        """ 

        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_F: string %s must have 2 elements.' % string)
        return self.F[self.slice_dict[string[0]], self.slice_dict[string[1]]]


    def get_pert(self, string):
        """
        Obtains the Perturbation Matix.

        Parameters:
        -----------
           string: string
             String of perturbation indexes.    
               
        Returns:
        --------  
           per: Numpy array
             The perturbation matrix in the MO basis.
        """

        if len(string) != 2:
            psi4.core.clean()
            raise Exception('get_pert: string %s must have 2 elements.' % string)
        return self.pert[self.slice_dict[string[0]], self.slice_dict[string[1]]]

    # Build different pieces of the similarity transformed perturbation operator
    # using ground state T amplitudes i.e T(0).
    # A_bar = e^{-T(0)} A e^{T(0)} = A + [A,T(0)] + 1/2! [[A,T(0)],T(0)] 
    # since A is a one body operator, the expansion truncates at double commutators.

    def build_Aoo(self):
        """
        Obtains the Perturbation Matix with (o,o) indexes.

        Returns:
        --------  
           Avo: Numpy array
             The perturbation matrix with (o,o) indexes.
        """

        Aoo = self.get_pert('oo').copy()
        Aoo += ndot('ie,me->mi', self.t1, self.get_pert('ov'))

        return Aoo

    def build_Aov(self):
        """
        Obtains the Perturbation Matix with (o,v) indexes.
        
        Returns:
        --------  
           Avo: Numpy array
             The perturbation matrix with (o,v) indexes.
        """

        Aov = self.get_pert('ov').copy()

        return Aov

    def build_Avo(self):
        """
        Obtains the Perturbation Matix with (v,o) indexes.
        
        Returns:
        --------  
           Avo: Numpy array
             The perturbation matrix with (v,o) indexes.
        """

        Avo =  self.get_pert('vo').copy()
        Avo += ndot('ae,ie->ai', self.get_pert('vv'), self.t1)
        Avo -= ndot('ma,mi->ai', self.t1, self.get_pert('oo'))
        Avo += ndot('miea,me->ai', self.t2, self.get_pert('ov'), prefactor=2.0)
        Avo += ndot('imea,me->ai', self.t2, self.get_pert('ov'), prefactor=-1.0)
        tmp = ndot('ie,ma->imea', self.t1, self.t1)
        Avo -= ndot('imea,me->ai', tmp, self.get_pert('ov'))

        return Avo

    def build_Avv(self):
        """
        Obtains the Perturbation Matix with (v,v) indexes.
        
        Returns:
        --------  
           Avv: Numpy array
             The perturbation matrix with (v,v) indexes.
        """

        Avv =  self.get_pert('vv').copy()
        Avv -= ndot('ma,me->ae', self.t1, self.get_pert('ov'))

        return Avv

    def build_Aovoo(self):
        """
        Obtains the Perturbation tensor with (o,v,o,o) indexes.
        
        Returns:
        --------  
           Aovoo: Numpy array
             The perturbation tensor (o,v,o,o) indexes.
        """

        Aovoo = ndot('ijeb,me->mbij', self.t2, self.get_pert('ov'))

        return Aovoo

    def build_Avvvo(self):
        """
        Obtains the Perturbation tensor with (v,v,v,o) indexes.
        
        Returns:
        --------  
           Avvvo: Numpy array
             The perturbation tensor (v,v,v,o) indexes.
        """

        Avvvo = -1.0*ndot('miab,me->abei', self.t2, self.get_pert('ov'))

        return Avvvo

    def build_Avvoo(self):
        """
        Obtains the Perturbation tensor with (v,v,o,o) indexes.
        
        Returns:
        --------  
           Avvoo: Numpy array
             The perturbation tensor (v,v,o,o) indexes.
        """

        Avvoo = ndot('ijeb,ae->abij', self.t2, self.build_Avv())
        Avvoo -= ndot('mjab,mi->abij', self.t2, self.build_Aoo())

        return Avvoo

    # Intermediates to avoid construction of 3 body Hbar terms
    # in solving X amplitude equations.
    def build_Zvv(self):
        """
        Obtains the intermediate Z matrix with (v,v) indexes.
        It is used to avoid construction of 3 body Hbar terms.
        
        Returns:
        --------  
           Zvv: Numpy array
             The intermediate Z matrix with (v,v) indexes.
        """

        Zvv = ndot('amef,mf->ae', self.Hvovv, self.x1, prefactor=2.0)
        Zvv += ndot('amfe,mf->ae', self.Hvovv, self.x1, prefactor=-1.0)
        Zvv -= ndot('mnaf,mnef->ae', self.x2, self.Loovv)

        return Zvv

    def build_Zoo(self):
        """
        Obtains the intermediate Z matrix with (o,o) indexes.
        It is used to avoid construction of 3 body Hbar terms.
        
        Returns:
        --------  
           Zoo: Numpy array
             The intermediate Z matrix with (o,o) indexes.
        """ 

        Zoo = ndot('mnie,ne->mi', self.Hooov, self.x1, prefactor=2.0)
        Zoo -= ndot('nmie,ne->mi', self.Hooov, self.x1, prefactor=-1.0)
        Zoo -= ndot('mnef,inef->mi', self.Loovv, self.x2)

        return Zoo

    # Intermediates to avoid construction of 3 body Hbar terms
    # in solving Y amplitude equations (just like in lambda equations).
    def build_Goo(self, t2, y2):
        """
        Obtains the intermediate G matrix with (o,o) indexes.
        It is used to avoid construction of 3 body Hbar terms.
        
        Returns:
        --------  
           Goo: Numpy array
             The intermediate G matrix with (o,o) indexes.
        """

        Goo = ndot('mjab,ijab->mi', t2, y2)

        return Goo

    def build_Gvv(self, y2, t2):
        """
        Obtains the intermediate G matrix with (o,o) indexes.
        It is used to avoid construction of 3 body Hbar terms.
        
        Returns:
        --------  
           Gvv: Numpy array
             The intermediate G matrix with (v,v) indexes.
        """
        Gvv = -1.0*ndot('ijab,ijeb->ae', y2, t2)

        return Gvv

    def update_X(self, omega):

        """
        Updates X1 and X2 amplitudes.
  
        Parameters:
        ----------
            omega: float
	       Perturbation frequency.

        Returns: 
        -------
            rms: float
              Residual of amplitudes. 
                  
        Note:
        ------ 
         X1 and X2 amplitudes are the Fourier analogues of first order perturbed T1 and T2 amplitudes, 
         (eq. 65, [Crawford:xxxx]). For a given perturbation, these amplitudes are frequency dependent and 
         can be obtained by solving a linear system of equations, (Hbar(0) - omgea * I)X = Hbar(1)
         Refer to eq 70 of [Crawford:xxxx]. Writing t_mu^(1)(omega) as X_mu and Hbar^(1)(omega) as A_bar,
         X1 equations:
         omega * X_ia = <phi^a_i|A_bar|O> + <phi^a_i|Hbar^(0)|phi^c_k> * X_kc + <phi^a_i|Hbar^(0)|phi^cd_kl> * X_klcd
         X2 equations:
         omega * X_ijab = <phi^ab_ij|A_bar|O> + <phi^ab_ij|Hbar^(0)|phi^c_k> * X_kc + <phi^ab_ij|Hbar^(0)|phi^cd_kl> * X_klcd
         Note that the RHS terms have exactly the same structure as EOM-CCSD sigma equations.
         Spin Orbital expressions (Einstein summation):

         #X1 equations: 
         -omega * X_ia + A_bar_ai + X_ie * Hvv_ae - X_ma * Hoo_mi + X_me * Hovvo_maei + X_miea * Hov_me 
         + 0.5 * X_imef * Hvovv_amef - 0.5 * X_mnae * Hooov_mnie = 0

         X2 equations:
         -omega * X_ijab + A_bar_abij + P(ij) X_ie * Hvvvo_abej - P(ab) X_ma * Hovoo_mbij 
         + P(ab) X_mf * Hvovv_amef * t_ijeb - P(ij) X_ne * Hooov_mnie * t_mjab 
         + P(ab) X_ijeb * Hvv_ae  - P(ij) X_mjab * Hov_mi + 0.5 * X_mnab * Hoooo_mnij + 0.5 * X_ijef * Hvvvv_abef 
         + P(ij) P(ab) X_miea * Hovvo_mbej - 0.5 * P(ab) X_mnaf * Hoovv_mnef * t_ijeb
         - 0.5 * P(ij) X_inef * Hoovv_mnef * t_mjab    

         It should be noted that in order to avoid construction of 3-body Hbar terms appearing in X2 equations like,
         Hvvooov_bamjif = Hvovv_amef * t_ijeb, 
         Hvvooov_banjie = Hooov_mnie * t_mjab,
         Hvoooov_bmnjif = Hoovv_mnef * t_ijeb, 
         Hvvoovv_banjef = Hoovv_mnef * t_mjab,  
         we make use of Z intermediates: 
         Zvv_ae = - Hooov_amef * X_mf - 0.5 * X_mnaf * Hoovv_mnef,  
         Zoo_mi = - X_ne * Hooov_mnie - 0.5 * Hoovv_mnef * X_inef,  
         And then contract Z with T2 amplitudes.
          
        """ 
  
        # X1 equations 
        r_x1  = self.build_Avo().swapaxes(0,1).copy()
        r_x1 -= omega * self.x1.copy()
        r_x1 += ndot('ie,ae->ia', self.x1, self.Hvv)
        r_x1 -= ndot('mi,ma->ia', self.Hoo, self.x1)
        r_x1 += ndot('maei,me->ia', self.Hovvo, self.x1, prefactor=2.0)
        r_x1 += ndot('maie,me->ia', self.Hovov, self.x1, prefactor=-1.0)
        r_x1 += ndot('miea,me->ia', self.x2, self.Hov, prefactor=2.0)
        r_x1 += ndot('imea,me->ia', self.x2, self.Hov, prefactor=-1.0)
        r_x1 += ndot('imef,amef->ia', self.x2, self.Hvovv, prefactor=2.0)
        r_x1 += ndot('imef,amfe->ia', self.x2, self.Hvovv, prefactor=-1.0)
        r_x1 -= ndot('mnie,mnae->ia', self.Hooov, self.x2, prefactor=2.0)
        r_x1 -= ndot('nmie,mnae->ia', self.Hooov, self.x2, prefactor=-1.0)
        # X1 equations over!    

        # X2 equations 
        # Final r_x2_ijab = r_x2_ijab + r_x2_jiba
        r_x2 = self.build_Avvoo().swapaxes(0,2).swapaxes(1,3).copy()
        # a factor of 0.5 because of the comment just above
        # and due to the fact that X2_ijab = X2_jiba  
        r_x2 -= 0.5 * omega * self.x2
        r_x2 += ndot('ie,abej->ijab', self.x1, self.Hvvvo)
        r_x2 -= ndot('mbij,ma->ijab', self.Hovoo, self.x1)
        r_x2 += ndot('ijeb,ae->ijab', self.x2, self.Hvv)
        r_x2 -= ndot('mi,mjab->ijab', self.Hoo, self.x2)
        r_x2 += ndot('mnij,mnab->ijab', self.Hoooo, self.x2, prefactor=0.5)
        r_x2 += ndot('ijef,abef->ijab', self.x2, self.Hvvvv, prefactor=0.5)
        r_x2 += ndot('miea,mbej->ijab', self.x2, self.Hovvo, prefactor=2.0)
        r_x2 += ndot('miea,mbje->ijab', self.x2, self.Hovov, prefactor=-1.0)
        r_x2 -= ndot('imeb,maje->ijab', self.x2, self.Hovov)
        r_x2 -= ndot('imea,mbej->ijab', self.x2, self.Hovvo)
        r_x2 += ndot('mi,mjab->ijab', self.build_Zoo(), self.t2)
        r_x2 += ndot('ijeb,ae->ijab', self.t2, self.build_Zvv())
        # X2 equations over!    

        old_x2 = self.x2.copy()
        old_x1 = self.x1.copy()

        # update X1 and X2
        self.x1 += r_x1/self.Dia
        # Final r_x2_ijab = r_x2_ijab + r_x2_jiba
        tmp = r_x2/self.Dijab
        self.x2 += tmp + tmp.swapaxes(0,1).swapaxes(2,3)

        # Calcuate rms with the residual 
        rms = 0
        rms += np.einsum('ia,ia->', old_x1 - self.x1, old_x1 - self.x1)
        rms += np.einsum('ijab,ijab->', old_x2 - self.x2, old_x2 - self.x2)

        return np.sqrt(rms)

    def inhomogenous_y2(self):
        """
        Computes inhomogenous terms appearing in Y2 equations.
  
        Returns: 
        -------
            r_y2: NumPy array
              Perturbed Y2 amplitudes. 
        """

        # <O|L1(0)|A_bar|phi^ab_ij>
        r_y2  = ndot('ia,jb->ijab', self.l1, self.build_Aov(), prefactor=2.0)
        r_y2 -= ndot('ja,ib->ijab', self.l1, self.build_Aov()) 
        # <O|L2(0)|A_bar|phi^ab_ij>
        r_y2 += ndot('ijeb,ea->ijab', self.l2, self.build_Avv())
        r_y2 -= ndot('im,mjab->ijab', self.build_Aoo(), self.l2)
        # <O|L1(0)|[Hbar(0), X1]|phi^ab_ij>
        tmp   = ndot('me,ja->meja', self.x1, self.l1)
        r_y2 -= ndot('mieb,meja->ijab', self.Loovv, tmp)
        tmp   = ndot('me,mb->eb', self.x1, self.l1)
        r_y2 -= ndot('ijae,eb->ijab', self.Loovv, tmp)
        tmp   = ndot('me,ie->mi', self.x1, self.l1)
        r_y2 -= ndot('mi,jmba->ijab', tmp, self.Loovv)
        tmp   = ndot('me,jb->mejb', self.x1, self.l1, prefactor=2.0)
        r_y2 += ndot('imae,mejb->ijab', self.Loovv, tmp)
        # <O|L2(0)|[Hbar(0), X1]|phi^ab_ij>
        tmp   = ndot('me,ma->ea', self.x1, self.Hov)
        r_y2 -= ndot('ijeb,ea->ijab', self.l2, tmp)
        tmp   = ndot('me,ie->mi', self.x1, self.Hov)
        r_y2 -= ndot('mi,jmba->ijab', tmp, self.l2)
        tmp   = ndot('me,ijef->mijf', self.x1, self.l2)
        r_y2 -= ndot('mijf,fmba->ijab', tmp, self.Hvovv)
        tmp   = ndot('me,imbf->eibf', self.x1, self.l2)
        r_y2 -= ndot('eibf,fjea->ijab', tmp, self.Hvovv)
        tmp   = ndot('me,jmfa->ejfa', self.x1, self.l2)
        r_y2 -= ndot('fibe,ejfa->ijab', self.Hvovv, tmp)
        tmp   = ndot('me,fmae->fa', self.x1, self.Hvovv, prefactor=2.0)
        tmp  -= ndot('me,fmea->fa', self.x1, self.Hvovv)
        r_y2 += ndot('ijfb,fa->ijab', self.l2, tmp)
        tmp   = ndot('me,fiea->mfia', self.x1, self.Hvovv, prefactor=2.0)
        tmp  -= ndot('me,fiae->mfia', self.x1, self.Hvovv)
        r_y2 += ndot('mfia,jmbf->ijab', tmp, self.l2)
        tmp   = ndot('me,jmna->ejna', self.x1, self.Hooov)
        r_y2 += ndot('ineb,ejna->ijab', self.l2, tmp)
        tmp   = ndot('me,mjna->ejna', self.x1, self.Hooov)
        r_y2 += ndot('nieb,ejna->ijab', self.l2, tmp)
        tmp   = ndot('me,nmba->enba', self.x1, self.l2)
        r_y2 += ndot('jine,enba->ijab', self.Hooov, tmp)
        tmp   = ndot('me,mina->eina', self.x1, self.Hooov, prefactor=2.0)
        tmp  -= ndot('me,imna->eina', self.x1, self.Hooov)
        r_y2 -= ndot('eina,njeb->ijab', tmp, self.l2)
        tmp   = ndot('me,imne->in', self.x1, self.Hooov, prefactor=2.0)
        tmp  -= ndot('me,mine->in', self.x1, self.Hooov)
        r_y2 -= ndot('in,jnba->ijab', tmp, self.l2)
        # <O|L2(0)|[Hbar(0), X2]|phi^ab_ij>
        tmp   = ndot('ijef,mnef->ijmn', self.l2, self.x2, prefactor=0.5)        
        r_y2 += ndot('ijmn,mnab->ijab', tmp, self.get_MO('oovv'))     
        tmp   = ndot('ijfe,mnef->ijmn', self.get_MO('oovv'), self.x2, prefactor=0.5)        
        r_y2 += ndot('ijmn,mnba->ijab', tmp, self.l2)        
        tmp   = ndot('mifb,mnef->ibne', self.l2, self.x2)        
        r_y2 += ndot('ibne,jnae->ijab', tmp, self.get_MO('oovv'))        
        tmp   = ndot('imfb,mnef->ibne', self.l2, self.x2)        
        r_y2 += ndot('ibne,njae->ijab', tmp, self.get_MO('oovv'))        
        tmp   = ndot('mjfb,mnef->jbne', self.l2, self.x2)        
        r_y2 -= ndot('jbne,inae->ijab', tmp, self.Loovv)        
        r_y2 -=  ndot('in,jnba->ijab', self.build_Goo(self.Loovv, self.x2), self.l2) 
        r_y2 +=  ndot('ijfb,af->ijab', self.l2, self.build_Gvv(self.Loovv, self.x2))
        r_y2 +=  ndot('ijae,be->ijab', self.Loovv, self.build_Gvv(self.l2, self.x2))
        r_y2 -=  ndot('imab,jm->ijab', self.Loovv, self.build_Goo(self.l2, self.x2))
        tmp   = ndot('nifb,mnef->ibme', self.l2, self.x2)
        r_y2 -= ndot('ibme,mjea->ijab', tmp, self.Loovv)
        tmp   = ndot('njfb,mnef->jbme', self.l2, self.x2, prefactor=2.0)
        r_y2 += ndot('imae,jbme->ijab', self.Loovv, tmp)

        return r_y2


    def inhomogenous_y1(self):
        """
        Computes inhomogenous terms appearing in Y1 equations.
  
        Returns: 
        -------
            r_y1: NumPy array
              Perturbed Y1 amplitudes. 
        """

        # <O|A_bar|phi^a_i>
        r_y1 = 2.0 * self.build_Aov().copy()
        # <O|L1(0)|A_bar|phi^a_i>
        r_y1 -= ndot('im,ma->ia', self.build_Aoo(), self.l1)
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Avv())
        # <O|L2(0)|A_bar|phi^a_i>
        r_y1 += ndot('imfe,feam->ia', self.l2, self.build_Avvvo())
        r_y1 -= ndot('ienm,mnea->ia', self.build_Aovoo(), self.l2, prefactor=0.5)
        r_y1 -= ndot('iemn,mnae->ia', self.build_Aovoo(), self.l2, prefactor=0.5)
        # <O|[Hbar(0), X1]|phi^a_i>
        r_y1 +=  ndot('imae,me->ia', self.Loovv, self.x1, prefactor=2.0)
        # <O|L1(0)|[Hbar(0), X1]|phi^a_i>
        tmp  = ndot('ma,ie->miae', self.Hov, self.l1, prefactor=-1.0)
        tmp -= ndot('ma,ie->miae', self.l1, self.Hov)
        tmp -= ndot('mina,ne->miae', self.Hooov, self.l1, prefactor=2.0)
        tmp -= ndot('imna,ne->miae', self.Hooov, self.l1, prefactor=-1.0)
        tmp -= ndot('imne,na->miae', self.Hooov, self.l1, prefactor=2.0)
        tmp -= ndot('mine,na->miae', self.Hooov, self.l1, prefactor=-1.0)
        tmp += ndot('fmae,if->miae', self.Hvovv, self.l1, prefactor=2.0)
        tmp += ndot('fmea,if->miae', self.Hvovv, self.l1, prefactor=-1.0)
        tmp += ndot('fiea,mf->miae', self.Hvovv, self.l1, prefactor=2.0)
        tmp += ndot('fiae,mf->miae', self.Hvovv, self.l1, prefactor=-1.0)
        r_y1 += ndot('miae,me->ia', tmp, self.x1)    
        # <O|L1(0)|[Hbar(0), X2]|phi^a_i>
        tmp  = ndot('mnef,nf->me', self.x2, self.l1, prefactor=2.0)
        tmp  += ndot('mnfe,nf->me', self.x2, self.l1, prefactor=-1.0)
        r_y1 += ndot('imae,me->ia', self.Loovv, tmp)
        r_y1 -= ndot('ni,na->ia', self.build_Goo(self.x2, self.Loovv), self.l1)
        r_y1 += ndot('ie,ea->ia', self.l1, self.build_Gvv(self.x2, self.Loovv))
        # <O|L2(0)|[Hbar(0), X1]|phi^a_i>
        tmp   = ndot('nief,mfna->iema', self.l2, self.Hovov, prefactor=-1.0)
        tmp  -= ndot('ifne,nmaf->iema', self.Hovov, self.l2)
        tmp  -= ndot('inef,mfan->iema', self.l2, self.Hovvo)
        tmp  -= ndot('ifen,nmfa->iema', self.Hovvo, self.l2)
        tmp  += ndot('imfg,fgae->iema', self.l2, self.Hvvvv, prefactor=0.5)
        tmp  += ndot('imgf,fgea->iema', self.l2, self.Hvvvv, prefactor=0.5)
        tmp  += ndot('imno,onea->iema', self.Hoooo, self.l2, prefactor=0.5)
        tmp  += ndot('mino,noea->iema', self.Hoooo, self.l2, prefactor=0.5)
        r_y1 += ndot('iema,me->ia', tmp, self.x1) 
        tmp  =  ndot('nb,fb->nf', self.x1, self.build_Gvv(self.t2, self.l2))
        r_y1 += ndot('inaf,nf->ia', self.Loovv, tmp) 
        tmp  =  ndot('me,fa->mefa', self.x1, self.build_Gvv(self.t2, self.l2))
        r_y1 += ndot('mief,mefa->ia', self.Loovv, tmp)
        tmp  =  ndot('me,ni->meni', self.x1, self.build_Goo(self.t2, self.l2))
        r_y1 -= ndot('meni,mnea->ia', tmp, self.Loovv)
        tmp  =  ndot('jf,nj->fn', self.x1, self.build_Goo(self.t2, self.l2))
        r_y1 -= ndot('inaf,fn->ia', self.Loovv, tmp)
        # <O|L2(0)|[Hbar(0), X2]|phi^a_i>
        r_y1 -= ndot('mi,ma->ia', self.build_Goo(self.x2, self.l2), self.Hov)  
        r_y1 += ndot('ie,ea->ia', self.Hov, self.build_Gvv(self.x2, self.l2)) 
        tmp   =  ndot('imfg,mnef->igne',self.l2, self.x2)
        r_y1 -=  ndot('igne,gnea->ia', tmp, self.Hvovv)
        tmp   =  ndot('mifg,mnef->igne',self.l2, self.x2)
        r_y1 -=  ndot('igne,gnae->ia', tmp, self.Hvovv)
        tmp   =  ndot('mnga,mnef->gaef',self.l2, self.x2)
        r_y1 -=  ndot('gief,gaef->ia', self.Hvovv, tmp)
        tmp   =  ndot('gmae,mnef->ganf',self.Hvovv, self.x2, prefactor=2.0)
        tmp  +=  ndot('gmea,mnef->ganf',self.Hvovv, self.x2, prefactor=-1.0)
        r_y1 +=  ndot('nifg,ganf->ia', self.l2, tmp)
        r_y1 -=  ndot('giea,ge->ia', self.Hvovv, self.build_Gvv(self.l2, self.x2), prefactor=2.0) 
        r_y1 -=  ndot('giae,ge->ia', self.Hvovv, self.build_Gvv(self.l2, self.x2), prefactor=-1.0)
        tmp   = ndot('oief,mnef->oimn', self.l2, self.x2) 
        r_y1 += ndot('oimn,mnoa->ia', tmp, self.Hooov)
        tmp   = ndot('mofa,mnef->oane', self.l2, self.x2) 
        r_y1 += ndot('inoe,oane->ia', self.Hooov, tmp)
        tmp   = ndot('onea,mnef->oamf', self.l2, self.x2) 
        r_y1 += ndot('miof,oamf->ia', self.Hooov, tmp)
        r_y1 -=  ndot('mioa,mo->ia', self.Hooov, self.build_Goo(self.x2, self.l2), prefactor=2.0) 
        r_y1 -=  ndot('imoa,mo->ia', self.Hooov, self.build_Goo(self.x2, self.l2), prefactor=-1.0) 
        tmp   = ndot('imoe,mnef->ionf', self.Hooov, self.x2, prefactor=-2.0) 
        tmp  -= ndot('mioe,mnef->ionf', self.Hooov, self.x2, prefactor=-1.0) 
        r_y1 += ndot('ionf,nofa->ia', tmp, self.l2)
        
        return r_y1

    def update_Y(self, omega):
        """
        Updates Y1 and Y2 amplitudes.
  
        Parameters:
        ----------
            omega: float
               Perturbation frequency.

        Returns: 
        -------
            rms: float
              Residual of amplitudes. 
                  
        Note:
        ------ 
         Y1 and Y2 amplitudes are the Fourier analogues of first order perturbed L1 and L2 amplitudes, 
         While X amplitudes are referred to as right hand perturbed amplitudes, Y amplitudes are the
         left hand perturbed amplitudes. Just like X1 and X2, they can be obtained by solving a linear 
         sytem of equations. Refer to eq 73 of [Crawford:xxxx]. for Writing l_mu^(1)(omega) as Y_mu, 
         Y1 equations:
         omega * Y_ia + Y_kc * <phi^c_k|Hbar(0)|phi^a_i>  + Y_klcd * <phi^cd_kl|Hbar(0)|phi^a_i> 
         + <O|(1 + L(0))|Hbar_bar(1)(omega)|phi^a_i> = 0
         Y2 equations: 
         omega * Y_ijab + Y_kc * <phi^c_k|Hbar(0)|phi^ab_ij>  + Y_klcd * <phi^cd_kl|Hbar(0)|phi^ab_ij> 
         + <O|(1 + L(0))|Hbar_bar(1)(omega)|phi^ab_ij> = 0
         where Hbar_bar(1)(omega) = Hbar(1) + [Hbar(0), T(1)] = A_bar + [Hbar(0), X]
         Note that the homogenous terms of Y1 and Y2 equations except the omega term are exactly identical in 
         structure to the L1 and L2 equations and just like lambdas, the equations for these Y amplitudes have 
         been derived using the unitray group approach. Please refer to helper_cclambda file for a complete  
         decsription.
        """
 
        # Y1 equations
        # Inhomogenous terms
        r_y1 = self.im_y1.copy()
        # Homogenous terms now!
        r_y1 += omega * self.y1
        r_y1 += ndot('ie,ea->ia', self.y1, self.Hvv)
        r_y1 -= ndot('im,ma->ia', self.Hoo, self.y1)
        r_y1 += ndot('ieam,me->ia', self.Hovvo, self.y1, prefactor=2.0)
        r_y1 += ndot('iema,me->ia', self.Hovov, self.y1, prefactor=-1.0)
        r_y1 += ndot('imef,efam->ia', self.y2, self.Hvvvo)
        r_y1 -= ndot('iemn,mnae->ia', self.Hovoo, self.y2)
        r_y1 -= ndot('eifa,ef->ia', self.Hvovv, self.build_Gvv(self.y2, self.t2), prefactor=2.0)
        r_y1 -= ndot('eiaf,ef->ia', self.Hvovv, self.build_Gvv(self.y2, self.t2), prefactor=-1.0)
        r_y1 -= ndot('mina,mn->ia', self.Hooov, self.build_Goo(self.t2, self.y2), prefactor=2.0)
        r_y1 -= ndot('imna,mn->ia', self.Hooov, self.build_Goo(self.t2, self.y2), prefactor=-1.0)
        # Y1 equations over!

        # Y2 equations
        # Final r_y2_ijab = r_y2_ijab + r_y2_jiba
        # Inhomogenous terms
        r_y2 = self.im_y2.copy()
        # Homogenous terms now!
        # a factor of 0.5 because of the relation/comment just above
        # and due to the fact that Y2_ijab = Y2_jiba  
        r_y2 += 0.5 * omega * self.y2.copy()
        r_y2 += ndot('ia,jb->ijab', self.y1, self.Hov, prefactor=2.0)
        r_y2 -= ndot('ja,ib->ijab', self.y1, self.Hov)
        r_y2 += ndot('ijeb,ea->ijab', self.y2, self.Hvv)
        r_y2 -= ndot('im,mjab->ijab', self.Hoo, self.y2)
        r_y2 += ndot('ijmn,mnab->ijab', self.Hoooo, self.y2, prefactor=0.5)
        r_y2 += ndot('ijef,efab->ijab', self.y2, self.Hvvvv, prefactor=0.5)
        r_y2 += ndot('ie,ejab->ijab', self.y1, self.Hvovv, prefactor=2.0)
        r_y2 += ndot('ie,ejba->ijab', self.y1, self.Hvovv, prefactor=-1.0)
        r_y2 -= ndot('mb,jima->ijab', self.y1, self.Hooov, prefactor=2.0)
        r_y2 -= ndot('mb,ijma->ijab', self.y1, self.Hooov, prefactor=-1.0)
        r_y2 += ndot('ieam,mjeb->ijab', self.Hovvo, self.y2, prefactor=2.0)
        r_y2 += ndot('iema,mjeb->ijab', self.Hovov, self.y2, prefactor=-1.0)
        r_y2 -= ndot('mibe,jema->ijab', self.y2, self.Hovov)
        r_y2 -= ndot('mieb,jeam->ijab', self.y2, self.Hovvo)
        r_y2 += ndot('ijeb,ae->ijab', self.Loovv, self.build_Gvv(self.y2, self.t2))
        r_y2 -= ndot('mi,mjab->ijab', self.build_Goo(self.t2, self.y2), self.Loovv)
        # Y2 equations over!

        old_y1 = self.y1.copy()
        old_y2 = self.y2.copy()

        # update Y1 and Y2
        self.y1 += r_y1/self.Dia
        # Final r_y2_ijab = r_y2_ijab + r_y2_jiba
        tmp = r_y2/self.Dijab    
        self.y2 += tmp + tmp.swapaxes(0,1).swapaxes(2,3) 

        # Calcuate rms from the residual 
        rms = np.einsum('ia,ia->', r_y1/self.Dia, r_y1/self.Dia)
        rms += np.einsum('ijab,ijab->', old_y2 - self.y2, old_y2 - self.y2)
        return np.sqrt(rms)

    def pseudoresponse(self, hand):
        """
        Obtains psudoresponse value.
  
        Parameters:
        ----------
            hand: string
               Specifies the type of aplitudes to be computed.
	       Right: X
 	       Left: Y

        Returns: 
        -------
            polar: float
              Psudoresponse value. 
        """

        polar1 = 0
        polar2 = 0
        if hand == 'right':
            z1 = self.x1 ; z2 = self.x2
        else:
            z1 = self.y1 ; z2 = self.y2

        # To match ihe pseudoresponse values with PSI4
        polar1 += ndot('ia,ai->', z1, self.build_Avo(), prefactor=2.0)
        tmp = self.pertbar_ijab + self.pertbar_ijab.swapaxes(0,1).swapaxes(2,3) 
        polar2 += ndot('ijab,ijab->', z2, tmp, prefactor=2.0)
        polar2 += ndot('ijba,ijab->', z2, tmp, prefactor=-1.0)

        polar = -2.0 * (polar1 + polar2)

        return polar

    def solve(self, hand, r_conv=1.e-7, maxiter=100, max_diis=8, start_diis=1):
        """
        Engine which solves perturbed amplitudes.
  
        Parameters:
        ----------
            hand: string
               Specifies the type of aplitudes to be computed.
               Right: X
               Left: Y
            r_conv: float, optional
               Convergance threshold.
            maxiter: float, optional  
               Maximum number of CC iterations.
            max_diis: float, optional
               Maximum numer of DISS steps.    
            start_diis: float, optional
               The diss step to begin with.    

        Returns:
        --------
            pseudoresponse: float
               The converged pseudoresponse value.        

        """

        ### Start of the solve routine 
        ccpert_tstart = time.time()
        
        # calculate the pseudoresponse from guess amplitudes
        pseudoresponse_old = self.pseudoresponse(hand)
        print("CCPERT_%s Iteration %3d: pseudoresponse = %.15f   dE = % .5E " % \
             (self.name, 0, pseudoresponse_old, -pseudoresponse_old))

        # Set up DIIS before iterations begin
        if hand == 'right':
            diis_object = helper_diis(self.x1, self.x2, max_diis)
        else:
            diis_object = helper_diis(self.y1, self.y2, max_diis)
            # calculate the inhomogenous terms of the left hand amplitudes equation before iterations begin
            self.im_y1 = self.inhomogenous_y1()
            self.im_y2 = self.inhomogenous_y2()

        # Iterate!
        for CCPERT_iter in range(1, maxiter + 1):

            # Residual build and update
            if hand == 'right':
                rms = self.update_X(self.omega1)
            else:
                rms = self.update_Y(self.omega1)

            # pseudoresponse with updated amplitudes
            pseudoresponse = self.pseudoresponse(hand)

            # Print CCPERT iteration information
            print('CCPERT_%s Iteration %3d: pseudoresponse = %.15f   dE = % .5E   DIIS = %d' % \
                 (self.name, CCPERT_iter, pseudoresponse, (pseudoresponse - pseudoresponse_old), diis_object.diis_size))

            # Check convergence
            if (rms < r_conv):
                print('\nCCPERT_%s has converged in %.3f seconds!' % (self.name, time.time() - ccpert_tstart))
                return pseudoresponse

            # Update old pseudoresponse
            pseudoresponse_old = pseudoresponse

            #  Add the new error vector
            if hand == 'right':
                diis_object.add_error_vector(self.x1, self.x2)
            else:
                diis_object.add_error_vector(self.y1, self.y2)


            if CCPERT_iter >= start_diis:
                if hand == 'right':    
                    self.x1, self.x2 = diis_object.extrapolate(self.x1, self.x2)
                else:    
                    self.y1, self.y2 = diis_object.extrapolate(self.y1, self.y2)

# End HelperCCPert class

class HelperCCLinresp(object):

    def __init__(self, cclambda, ccpert_A, ccpert_B):
        """
        Initializes the HelperCCLinresp object.
        
        Parameters:
        -----------
        cclambda: cclambda object
             An initialized cclambda object.  
        ccpert_A: ccpert object
             An initialized ccpert object containing all info about perturbation A.
        ccpert_B: ccpert object
             An initialized ccpert object containing all info about perturbation B.
        ccpert_C: ccpert object  
             An initialized ccpert object containing all info about perturbation C. 
        """

        # start of the cclinresp class 
        time_init = time.time()
        # Grab all the info from ccpert obejct, a and b here are the two 
        # perturbations Ex. for dipole polarizabilities, A = mu, B = mu (dipole operator) 
        self.ccpert_A = ccpert_A
        self.ccpert_B = ccpert_B
        self.pert_A = ccpert_A.pert
        self.pert_B = ccpert_B.pert
        self.l1 = cclambda.l1
        self.l2 = cclambda.l2
        # Grab X and Y amplitudes corresponding to perturbation A
        self.x1_A = ccpert_A.x1
        self.x2_A = ccpert_A.x2
        self.y1_A = ccpert_A.y1
        self.y2_A = ccpert_A.y2
        # Grab X and Y amplitudes corresponding to perturbation B
        self.x1_B = ccpert_B.x1
        self.x2_B = ccpert_B.x2
        self.y1_B = ccpert_B.y1
        self.y2_B = ccpert_B.y2
	
    def linresp(self):
        """
        Computes the Linear Response Function value.
 
        Returns
        -------
        polar: float
             The linear response function value.

        Note:
        -----
        Refer to equation 78 of [Crawford:xxxx]. 
        Writing H(1)(omega) = B, T(1)(omega) = X, L(1)(omega) = Y
        <<A;B>> =  <0|Y(B) * A_bar|0> + <0|(1+L(0))[A_bar, X(B)]|0> 
                        polar1                    polar2
        """

        self.polar1 = 0
        self.polar2 = 0
        # <0|Y1(B) * A_bar|0>
        self.polar1 += ndot("ai,ia->", self.ccpert_A.build_Avo(), self.y1_B)
        # <0|Y2(B) * A_bar|0>
        self.polar1 += ndot("abij,ijab->", self.ccpert_A.build_Avvoo(), self.y2_B, prefactor=0.5)
        self.polar1 += ndot("baji,ijab->", self.ccpert_A.build_Avvoo(), self.y2_B, prefactor=0.5)
        # <0|[A_bar, X(B)]|0>
        self.polar2 += ndot("ia,ia->", self.ccpert_A.build_Aov(), self.x1_B, prefactor=2.0)
        # <0|L1(0)[A_bar, X1(B)]|0>
        tmp = ndot('ia,ic->ac', self.l1, self.x1_B)
        self.polar2 += ndot('ac,ac->', tmp, self.ccpert_A.build_Avv())
        tmp = ndot('ia,ka->ik', self.l1, self.x1_B)
        self.polar2 -= ndot('ik,ki->', tmp, self.ccpert_A.build_Aoo())
        # <0|L1(0)[A_bar, X2(B)]|0>
        tmp = ndot('ia,jb->ijab', self.l1, self.ccpert_A.build_Aov())
        self.polar2 += ndot('ijab,ijab->', tmp, self.x2_B, prefactor=2.0)
        self.polar2 += ndot('ijab,ijba->', tmp, self.x2_B, prefactor=-1.0)
        # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = ndot('ijbc,bcaj->ia', self.l2, self.ccpert_A.build_Avvvo())
        self.polar2 += ndot('ia,ia->', tmp, self.x1_B)
        tmp = ndot('ijab,kbij->ak', self.l2, self.ccpert_A.build_Aovoo())
        self.polar2 -= ndot('ak,ka->', tmp, self.x1_B, prefactor=0.5)
        tmp = ndot('ijab,kaji->bk', self.l2, self.ccpert_A.build_Aovoo())
        self.polar2 -= ndot('bk,kb->', tmp, self.x1_B, prefactor=0.5)
        # <0|L2(0)[A_bar, X1(B)]|0>
        tmp = ndot('ijab,kjab->ik', self.l2, self.x2_B)
        self.polar2 -= ndot('ik,ki->', tmp, self.ccpert_A.build_Aoo(), prefactor=0.5)
        tmp = ndot('ijab,kiba->jk', self.l2, self.x2_B,)
        self.polar2 -= ndot('jk,kj->', tmp, self.ccpert_A.build_Aoo(), prefactor=0.5)
        tmp = ndot('ijab,ijac->bc', self.l2, self.x2_B,)
        self.polar2 += ndot('bc,bc->', tmp, self.ccpert_A.build_Avv(), prefactor=0.5)
        tmp = ndot('ijab,ijcb->ac', self.l2, self.x2_B,)
        self.polar2 += ndot('ac,ac->', tmp, self.ccpert_A.build_Avv(), prefactor=0.5)

        self.polar = -1.0*(self.polar1 + self.polar2)

        return self.polar


class HelperCCQuadraticResp(object):

    def __init__(self, ccsd,  cchbar, cclambda, ccpert_A, ccpert_B, ccpert_C):
        """
        Initializes the HelperCCQuadraticResp object.
        
        Parameters:
        -----------
        ccsd: ccsd object
             An initialized ccsd object.        
        cchbar: cchbar object
             An initialized cchbar object.  
        cclambda: cclambda object
             An initialized cclambda object.  
        ccpert_A: ccpert object
             An initialized ccpert object containing all info about perturbation A.
        ccpert_B: ccpert object
             An initialized ccpert object containing all info about perturbation B.
        ccpert_C: ccpert object  
             An initialized ccpert object containing all info about perturbation C. 
        """

        time_init = time.time()
        # Grab all the info from ccpert obejct, A, B and C are the perturbations
        # Ex. for dipole polarizabilities, A = mu, B = mu, C =mu (dipole operator) 
        self.ccpert_A = ccpert_A
        self.ccpert_B = ccpert_B
        self.ccpert_C = ccpert_C
        self.pert_A = ccpert_A.pert
        self.pert_B = ccpert_B.pert
        self.pert_C = ccpert_C.pert
        self.t1 = ccsd.t1
        self.t2 = ccsd.t2
        self.l1 = cclambda.l1
        self.l2 = cclambda.l2 
        # Grab X and Y amplitudes corresponding to perturbation A
        self.x1_A = ccpert_A.x1
        self.x2_A = ccpert_A.x2
        self.y1_A = ccpert_A.y1
        self.y2_A = ccpert_A.y2 
        # Grab X and Y amplitudes corresponding to perturbation B
        self.x1_B = ccpert_B.x1
        self.x2_B = ccpert_B.x2
        self.y1_B = ccpert_B.y1
        self.y2_B = ccpert_B.y2
        # Grab X and Y amplitudes corresponding to perturbation C
        self.x1_C = ccpert_C.x1
        self.x2_C = ccpert_C.x2
        self.y1_C = ccpert_C.y1
        self.y2_C = ccpert_C.y2

        self.Loovv = cchbar.Loovv
        self.Goovv = ccsd.get_MO("oovv")
        
        self.Aoo = self.ccpert_A.build_Aoo()
        self.Avv = self.ccpert_A.build_Avv()
        self.Aov = self.ccpert_A.build_Aov()
        self.Avvvo = self.ccpert_A.build_Avvvo()
        self.Aovoo = self.ccpert_A.build_Aovoo()

        self.Boo = self.ccpert_B.build_Aoo()
        self.Bvv = self.ccpert_B.build_Avv()
        self.Bov = self.ccpert_B.build_Aov()
        self.Bvvvo = self.ccpert_B.build_Avvvo()
        self.Bovoo = self.ccpert_B.build_Aovoo()

        self.Coo = self.ccpert_C.build_Aoo()
        self.Cvv = self.ccpert_C.build_Avv()
        self.Cov = self.ccpert_C.build_Aov()
        self.Cvvvo = self.ccpert_C.build_Avvvo()
        self.Covoo = self.ccpert_C.build_Aovoo()

        self.Hov = cchbar.Hov
        self.Hooov = cchbar.Hooov
        self.Hvovv = cchbar.Hvovv
        self.Hovov = cchbar.Hovov
        self.Hovvo = cchbar.Hovvo
        self.Hvvvv = cchbar.Hvvvv
        self.Hoooo = cchbar.Hoooo


    def quadraticresp(self):
        """
        Computes Quadratic Response Function value.
        Refer to eq. 107 of [Koch:1991:3333] for the general form of quadratic response functions.
        
        Returns
        -------
        hyper: float
             The quadratic response function value.
        """        

        # <0|L1(B)[A_bar, X1(C)]|0>
        tmp = np.einsum('ia,ic->ac',self.y1_B,self.x1_C)
        self.LAX = np.einsum('ac,ac->',tmp,self.Avv)
        tmp = ndot('ia,ka->ik',self.y1_B,self.x1_C)
        self.LAX -= np.einsum('ik,ki->',tmp,self.Aoo)
        # <0|L1(B)[A_bar, X2(C)]|0>
        tmp = ndot('ia,jb->ijab',self.y1_B,self.Aov)
        self.LAX += 2.*np.einsum('ijab,ijab->',tmp,self.x2_C)
        self.LAX -= np.einsum('ijab,ijba->',tmp,self.x2_C)
        # <0|L2(B)[A_bar, X1(C)]|0>
        tmp = np.einsum('ijbc,bcaj->ia',self.y2_B,self.Avvvo)
        self.LAX += np.einsum('ia,ia->',tmp,self.x1_C)
        tmp = ndot('ijab,kbij->ak',self.y2_B,self.Aovoo)
        self.LAX -= np.einsum('ak,ka->',tmp,self.x1_C)
        # <0|L2(B)[A_bar, X2(C)]|0>
        tmp = ndot('ijab,kjab->ik',self.y2_B,self.x2_C)
        self.LAX -= np.einsum('ik,ki->',tmp,self.Aoo)
        tmp = ndot('ijab,ijac->bc',self.y2_B,self.x2_C,)
        self.LAX += np.einsum('bc,bc->',tmp,self.Avv)

        self.hyper = self.LAX

        # <0|L1(C)[A_bar, X1(B)]|0>
        tmp = np.einsum('ia,ic->ac', self.y1_C,self.x1_B)
        self.LAX2 = np.einsum('ac,ac->',tmp,self.Avv)
        tmp = np.einsum('ia,ka->ik',self.y1_C,self.x1_B)
        self.LAX2 -= np.einsum('ik,ki->',tmp,self.Aoo)
        # <0|L1(C)[A_bar, X2(B)]|0>
        tmp = np.einsum('ia,jb->ijab',self.y1_C,self.Aov)
        self.LAX2 += 2.*np.einsum('ijab,ijab->',tmp,self.x2_B)
        self.LAX2 -= np.einsum('ijab,ijba->',tmp,self.x2_B)
        # <0|L2(C)[A_bar, X1(B)]|0>
        tmp = ndot('ijbc,bcaj->ia',self.y2_C,self.Avvvo)
        self.LAX2 += np.einsum('ia,ia->',tmp,self.x1_B)
        tmp = ndot('ijab,kbij->ak', self.y2_C,self.Aovoo)
        self.LAX2 -= np.einsum('ak,ka->', tmp, self.x1_B)
        # <0|L2(C)[A_bar, X2(B)]|0>
        tmp = np.einsum('ijab,kjab->ik',self.y2_C,self.x2_B)
        self.LAX2 -= np.einsum('ik,ki->',tmp,self.Aoo)
        tmp = np.einsum('ijab,ijac->bc',self.y2_C,self.x2_B) 
        self.LAX2 += np.einsum('bc,bc->',tmp,self.Avv)
     
        self.hyper += self.LAX2

        # <0|L1(A)[B_bar,X1(C)]|0>
        tmp = ndot('ia,ic->ac', self.y1_A, self.x1_C)
        self.LAX3 = np.einsum('ac,ac->',tmp,self.Bvv)
        tmp = ndot('ia,ka->ik',self.y1_A,self.x1_C)
        self.LAX3 -= np.einsum('ik,ki->',tmp,self.Boo)
        # <0|L1(A)[B_bar, X2(C)]|0>
        tmp = np.einsum('ia,jb->ijab',self.y1_A,self.Bov)
        self.LAX3 += 2.*np.einsum('ijab,ijab->',tmp,self.x2_C)
        self.LAX3 -= np.einsum('ijab,ijba->',tmp,self.x2_C)
        # <0|L2(A)[B_bar, X1(C)]|0>
        tmp = np.einsum('ijbc,bcaj->ia',self.y2_A,self.Bvvvo)
        self.LAX3 += np.einsum('ia,ia->',tmp,self.x1_C)
        tmp = np.einsum('ijab,kbij->ak',self.y2_A,self.Bovoo)
        self.LAX3 -= np.einsum('ak,ka->',tmp,self.x1_C)
        # <0|L2(A)[B_bar, X2(C)]|0>
        tmp = np.einsum('ijab,kjab->ik',self.y2_A,self.x2_C)
        self.LAX3 -= np.einsum('ik,ki->',tmp,self.Boo)
        tmp = np.einsum('ijab,ijac->bc',self.y2_A,self.x2_C)
        self.LAX3 += np.einsum('bc,bc->',tmp,self.Bvv)

        self.hyper += self.LAX3

        # <0|L1(C)|[B_bar,X1(A)]|0>
        tmp = np.einsum('ia,ic->ac',self.y1_C,self.x1_A)
        self.LAX4 = np.einsum('ac,ac->',tmp,self.Bvv)
        tmp = np.einsum('ia,ka->ik',self.y1_C,self.x1_A)
        self.LAX4 -= np.einsum('ik,ki->',tmp,self.Boo)
        # <0|L1(C)[B_bar, X2(A)]|0>
        tmp = np.einsum('ia,jb->ijab',self.y1_C,self.Bov)
        self.LAX4 += 2.*np.einsum('ijab,ijab->',tmp,self.x2_A)
        self.LAX4 -= np.einsum('ijab,ijba->',tmp,self.x2_A)
        # <0|L2(C)[B_bar, X1(A)]|0>
        tmp = np.einsum('ijbc,bcaj->ia',self.y2_C,self.Bvvvo)
        self.LAX4 += np.einsum('ia,ia->',tmp,self.x1_A)
        tmp = np.einsum('ijab,kbij->ak',self.y2_C,self.Bovoo)
        self.LAX4 -= np.einsum('ak,ka->',tmp,self.x1_A)
        # <0|L2(C)[B_bar, X2(A)]|0>
        tmp = np.einsum('ijab,kjab->ik',self.y2_C,self.x2_A)
        self.LAX4 -= np.einsum('ik,ki->',tmp,self.Boo)
        tmp = np.einsum('ijab,kiba->jk',self.y2_C,self.x2_A)
        tmp = np.einsum('ijab,ijac->bc',self.y2_C,self.x2_A)
        self.LAX4 += np.einsum('bc,bc->',tmp,self.Bvv)

        self.hyper += self.LAX4

        # <0|L1(A)[C_bar,X1(B)]|0>
        tmp = np.einsum('ia,ic->ac',self.y1_A,self.x1_B)
        self.LAX5 = np.einsum('ac,ac->',tmp,self.Cvv)
        tmp = np.einsum('ia,ka->ik',self.y1_A,self.x1_B)
        self.LAX5 -= np.einsum('ik,ki->',tmp,self.Coo)
        # <0|L1(A)[C_bar, X2(B)]|0>
        tmp = ndot('ia,jb->ijab',self.y1_A,self.Cov)
        self.LAX5 += 2.*np.einsum('ijab,ijab->',tmp,self.x2_B)
        self.LAX5 -= np.einsum('ijab,ijba->',tmp,self.x2_B)
        # <0|L2(A)[C_bar, X1(B)]|0>
        tmp = np.einsum('ijbc,bcaj->ia',self.y2_A,self.Cvvvo)
        self.LAX5 += np.einsum('ia,ia->',tmp,self.x1_B)
        tmp = ndot('ijab,kbij->ak', self.y2_A,self.Covoo)
        self.LAX5 -= np.einsum('ak,ka->',tmp,self.x1_B)
        # <0|L2(A)[C_bar, X2(B)]|0>
        tmp = np.einsum('ijab,kjab->ik',self.y2_A,self.x2_B)
        self.LAX5 -= np.einsum('ik,ki->',tmp,self.Coo)
        tmp = np.einsum('ijab,ijac->bc',self.y2_A,self.x2_B)
        self.LAX5 += np.einsum('bc,bc->',tmp,self.Cvv)

        self.hyper += self.LAX5

        # <0|L1(B)|[C_bar,X1(A)]|0>
        tmp = np.einsum('ia,ic->ac',self.y1_B,self.x1_A)
        self.LAX6 = np.einsum('ac,ac->',tmp,self.Cvv)
        tmp = np.einsum('ia,ka->ik',self.y1_B,self.x1_A)
        self.LAX6 -= np.einsum('ik,ki->',tmp,self.Coo)
        # <0|L1(B)[C_bar, X2(A)]|0>
        tmp = np.einsum('ia,jb->ijab',self.y1_B,self.Cov)
        self.LAX6 += 2.*np.einsum('ijab,ijab->',tmp,self.x2_A)
        self.LAX6 -= np.einsum('ijab,ijba->',tmp,self.x2_A)
        # <0|L2(B)[C_bar, X1(A)]|0>
        tmp = np.einsum('ijbc,bcaj->ia',self.y2_B,self.Cvvvo)
        self.LAX6 += np.einsum('ia,ia->',tmp,self.x1_A)
        tmp = np.einsum('ijab,kbij->ak',self.y2_B,self.Covoo)
        self.LAX6 -= np.einsum('ak,ka->',tmp,self.x1_A)
        # <0|L2(B)[C_bar, X2(A)]|0>
        tmp = np.einsum('ijab,kjab->ik',self.y2_B,self.x2_A)
        self.LAX6 -= np.einsum('ik,ki->',tmp,self.Coo)
        tmp = ndot('ijab,ijac->bc',self.y2_B,self.x2_A)
        self.LAX6 += np.einsum('bc,bc->',tmp,self.Cvv)
 
        self.hyper += self.LAX6
 
        # <0|L1(0)[[A_bar,X1(B)],X1(C)]|0>
        tmp = np.einsum('ia,ja->ij',self.x1_B,self.Aov)
        tmp2 = np.einsum('ib,jb->ij',self.l1,self.x1_C)
        self.Fz1 = np.einsum('ij,ij->',tmp2,tmp)
        tmp = np.einsum('jb,ib->ij',self.x1_C,self.Aov)
        tmp2 = np.einsum('ia,ja->ij',self.x1_B,self.l1)
        self.Fz1 -= np.einsum('ij,ij->',tmp2,tmp)
        # <0|L2(0)[[A_bar,X1(B)],X2(C)]|0>  
        tmp = np.einsum('ia,ja->ij',self.x1_B,self.Aov)
        tmp2 = np.einsum('jkbc,ikbc->ij',self.x2_C,self.l2)
        self.Fz1 -= np.einsum('ij,ij->',tmp2,tmp)

        tmp = np.einsum('ia,jkac->jkic',self.x1_B,self.l2)
        tmp = np.einsum('jkbc,jkic->ib',self.x2_C,tmp)
        self.Fz1 -= np.einsum('ib,ib->',tmp,self.Aov)

        # <0|L2(0)[[A_bar,X2(B)],X1(C)]|0>   
        tmp = np.einsum('ia,ja->ij',self.x1_C,self.Aov)
        tmp2 = np.einsum('jkbc,ikbc->ij',self.x2_B,self.l2)
        self.Fz1 = np.einsum('ij,ij->',tmp2,tmp)

        tmp = np.einsum('ia,jkac->jkic',self.x1_C,self.l2)
        tmp = np.einsum('jkbc,jkic->ib',self.x2_B,tmp)
        self.Fz1 -= np.einsum('ib,ib->',tmp,self.Aov)     

        # <0|L1(0)[B_bar,X1(A)],X1(C)]|0>
        tmp = np.einsum('ia,ja->ij',self.x1_A,self.Bov)
        tmp2 = np.einsum('ib,jb->ij',self.l1,self.x1_C)
        self.Fz2 = np.einsum('ij,ij->',tmp2,tmp)

        tmp = np.einsum('jb,ib->ij',self.x1_C,self.Bov)
        tmp2 = np.einsum('ia,ja->ij',self.x1_A,self.l1)
        self.Fz2 -= np.einsum('ij,ij->',tmp2,tmp)      

        # <0|L2(0)[[B_bar,X1(A)],X2(C)]|0>  
        tmp = np.einsum('ia,ja->ij',self.x1_A,self.Bov)
        tmp2 = np.einsum('jkbc,ikbc->ij',self.x2_C,self.l2)
        self.Fz2 -= np.einsum('ij,ij->',tmp2,tmp)

        tmp = np.einsum('ia,jkac->jkic',self.x1_A,self.l2)
        tmp = np.einsum('jkbc,jkic->ib',self.x2_C,tmp)
        self.Fz2 -= np.einsum('ib,ib->',tmp,self.Bov)

        # <0|L2(0)[[B_bar,X2(A)],X1(C)]|0>  
        tmp = np.einsum('ia,ja->ij',self.x1_C,self.Bov)
        tmp2 = np.einsum('jkbc,ikbc->ij',self.x2_A,self.l2)
        self.Fz2 -= np.einsum('ij,ij->',tmp2,tmp)
        
        tmp = np.einsum('ia,jkac->jkic',self.x1_C,self.l2)
        tmp = np.einsum('jkbc,jkic->ib',self.x2_A,tmp)
        self.Fz2 -= np.einsum('ib,ib->',tmp,self.Bov)

        # <0|L1(0)[C_bar,X1(A)],X1(B)]|0>  
        tmp = np.einsum('ia,ja->ij',self.x1_A,self.Cov)
        tmp2 = np.einsum('ib,jb->ij',self.l1,self.x1_B)
        self.Fz3 = np.einsum('ij,ij->',tmp2,tmp)
        
        tmp = np.einsum('jb,ib->ij',self.x1_B,self.Cov)
        tmp2 = np.einsum('ia,ja->ij',self.x1_A,self.l1)
        self.Fz3 -= np.einsum('ij,ij->',tmp2,tmp)         

        # <0|L2(0)[[C_bar,X1(A)],X2(B)]|0>  
        tmp = np.einsum('ia,ja->ij',self.x1_A,self.Cov)
        tmp2 = np.einsum('jkbc,ikbc->ij',self.x2_B,self.l2)
        self.Fz3 -= np.einsum('ij,ij->',tmp2,tmp)

        tmp = np.einsum('ia,jkac->jkic',self.x1_A,self.l2)
        tmp = np.einsum('jkbc,jkic->ib',self.x2_B,tmp)
        self.Fz3 -= np.einsum('ib,ib->',tmp,self.Cov)

        # <0|L2(0)[[C_bar,X2(A)],X1(B)]|0>
        tmp = np.einsum('ia,ja->ij',self.x1_B,self.Cov)
        tmp2 = np.einsum('jkbc,ikbc->ij',self.x2_A,self.l2)
        self.Fz3 -= np.einsum('ij,ij->',tmp2,tmp)

        tmp = np.einsum('ia,jkac->jkic',self.x1_B,self.l2)
        tmp = np.einsum('jkbc,jkic->ib',self.x2_A,tmp)
        self.Fz3 -= np.einsum('ib,ib->',tmp,self.Cov)

        self.hyper += self.Fz1+self.Fz2+self.Fz3

        # <L1(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
        tmp = np.einsum('ia,ijac->jc',self.x1_A,self.Loovv)
        tmp = np.einsum('kc,jc->jk',self.x1_C,tmp)
        tmp2 = np.einsum('jb,kb->jk',self.x1_B,self.l1)
        self.G = np.einsum('jk,jk->',tmp2,tmp)

        tmp = np.einsum('ia,ikab->kb',self.x1_A,self.Loovv)
        tmp = np.einsum('jb,kb->jk',self.x1_B,tmp)
        tmp2 = np.einsum('jc,kc->jk',self.l1,self.x1_C)
        self.G -= np.einsum('jk,jk->',tmp2,tmp)

        tmp = np.einsum('jb,jkba->ka',self.x1_B,self.Loovv)
        tmp = np.einsum('ia,ka->ki',self.x1_A,tmp)
        tmp2 = np.einsum('kc,ic->ki',self.x1_C,self.l1)
        self.G -= np.einsum('ki,ki->',tmp2,tmp)

        tmp = np.einsum('jb,jibc->ic',self.x1_B,self.Loovv)
        tmp = np.einsum('kc,ic->ki',self.x1_C,tmp)
        tmp2 = np.einsum('ka,ia->ki',self.l1,self.x1_A)
        self.G -= np.einsum('ki,ki->',tmp2,tmp)

        tmp = np.einsum('kc,kicb->ib',self.x1_C,self.Loovv)
        tmp = np.einsum('jb,ib->ji',self.x1_B,tmp)
        tmp2 = np.einsum('ja,ia->ji',self.l1,self.x1_A)
        self.G -= np.einsum('ji,ji->',tmp2,tmp)

        tmp = np.einsum('kc,kjca->ja',self.x1_C,self.Loovv)
        tmp = np.einsum('ia,ja->ji',self.x1_A,tmp)
        tmp2 = np.einsum('jb,ib->ji',self.x1_B,self.l1)
        self.G -= np.einsum('ji,ji->',tmp2,tmp)

        # <L2(0)|[[[H_bar,X1(A)],X1(B)],X1(C)]|0>
        tmp = np.einsum('jb,klib->klij',self.x1_A,self.Hooov)
        tmp2  = np.einsum('ld,ijcd->ijcl',self.x1_C,self.l2)
        tmp2  = np.einsum('kc,ijcl->ijkl',self.x1_B,tmp2)
        self.G += np.einsum('ijkl,klij->',tmp2,tmp)

        tmp = np.einsum('jb,lkib->lkij',self.x1_A,self.Hooov)
        tmp2 = np.einsum('ld,ijdc->ijlc',self.x1_C,self.l2)
        tmp2 = np.einsum('kc,ijlc->ijlk',self.x1_B,tmp2)
        self.G += np.einsum('ijlk,lkij->',tmp2,tmp)

        tmp = np.einsum('kc,jlic->jlik',self.x1_B,self.Hooov)
        tmp2  = np.einsum('jb,ikbd->ikjd',self.x1_A,self.l2)
        tmp2  = np.einsum('ld,ikjd->ikjl',self.x1_C,tmp2)
        self.G += np.einsum('ikjl,jlik->',tmp2,tmp)

        tmp = np.einsum('kc,ljic->ljik',self.x1_B,self.Hooov)
        tmp2  = np.einsum('jb,ikdb->ikdj',self.x1_A,self.l2)
        tmp2  = np.einsum('ld,ikdj->iklj',self.x1_C,tmp2)
        self.G += np.einsum('iklj,ljik->',tmp2,tmp)

        tmp = np.einsum('ld,jkid->jkil',self.x1_C,self.Hooov)
        tmp2  = np.einsum('jb,ilbc->iljc',self.x1_A,self.l2)
        tmp2  = np.einsum('kc,iljc->iljk',self.x1_B,tmp2)
        self.G += np.einsum('iljk,jkil->',tmp2,tmp)

        tmp = np.einsum('ld,kjid->kjil',self.x1_C,self.Hooov)
        tmp2  = np.einsum('jb,ilcb->ilcj',self.x1_A,self.l2)
        tmp2  = np.einsum('kc,ilcj->ilkj',self.x1_B,tmp2)
        self.G += np.einsum('ilkj,kjil->',tmp2,tmp)

        tmp = np.einsum('jb,albc->aljc',self.x1_A,self.Hvovv)
        tmp = np.einsum('kc,aljc->aljk',self.x1_B,tmp)
        tmp2  = np.einsum('ld,jkad->jkal',self.x1_C,self.l2)
        self.G -= np.einsum('jkal,aljk->',tmp2,tmp)

        tmp = np.einsum('jb,alcb->alcj',self.x1_A,self.Hvovv)
        tmp = np.einsum('kc,alcj->alkj',self.x1_B,tmp)
        tmp2  = np.einsum('ld,jkda->jkla',self.x1_C,self.l2)
        self.G -= np.einsum('jkla,alkj->',tmp2,tmp)

        tmp = np.einsum('jb,akbd->akjd',self.x1_A,self.Hvovv)
        tmp = np.einsum('ld,akjd->akjl',self.x1_C,tmp)
        tmp2  = np.einsum('kc,jlac->jlak',self.x1_B,self.l2)
        self.G -= np.einsum('jlak,akjl->',tmp2,tmp)

        tmp = np.einsum('jb,akdb->akdj',self.x1_A,self.Hvovv)
        tmp = np.einsum('ld,akdj->aklj',self.x1_C,tmp)
        tmp2  = np.einsum('kc,jlca->jlka',self.x1_B,self.l2)
        self.G -= np.einsum('jlka,aklj->',tmp2,tmp)

        tmp = np.einsum('kc,ajcd->ajkd',self.x1_B,self.Hvovv)
        tmp = np.einsum('ld,ajkd->ajkl',self.x1_C,tmp)
        tmp2  = np.einsum('jb,klab->klaj',self.x1_A,self.l2)
        self.G -= np.einsum('klaj,ajkl->',tmp2,tmp)

        tmp = np.einsum('kc,ajdc->ajdk',self.x1_B,self.Hvovv)
        tmp = np.einsum('ld,ajdk->ajlk',self.x1_C,tmp)
        tmp2  = np.einsum('jb,klba->klja',self.x1_A,self.l2)
        self.G -= np.einsum('klja,ajlk->',tmp2,tmp)

        # <L2(0)|[[[H_bar,X2(A)],X1(B)],X1(C)]|0>
        tmp = np.einsum('kc,jlbc->jlbk',self.x1_B,self.l2)
        tmp2 = np.einsum('ld,ikad->ikal',self.x1_C,self.Loovv)
        tmp2 = np.einsum('ijab,ikal->jlbk',self.x2_A,tmp2)
        self.G -= np.einsum('jlbk,jlbk->',tmp,tmp2)

        tmp = np.einsum('ld,jkbd->jkbl',self.x1_C,self.l2)
        tmp2 = np.einsum('kc,ilac->ilak',self.x1_B,self.Loovv)
        tmp2 = np.einsum('ijab,ilak->jkbl',self.x2_A,tmp2)
        self.G -= np.einsum('jkbl,jkbl->',tmp,tmp2)

        tmp = np.einsum('ijab,jibd->ad',self.x2_A,self.l2)
        tmp = np.einsum('ld,ad->la',self.x1_C,tmp)
        tmp2 = np.einsum('klca,kc->la',self.Loovv,self.x1_B)
        self.G -= np.einsum('la,la->',tmp,tmp2)

        tmp = np.einsum('ijab,jlba->il',self.x2_A,self.l2)
        tmp2 = np.einsum('kc,kicd->id',self.x1_B,self.Loovv)
        tmp2 = np.einsum('ld,id->il',self.x1_C,tmp2)
        self.G -= np.einsum('il,il->',tmp,tmp2)

        tmp = np.einsum('ijab,jkba->ik',self.x2_A,self.l2)
        tmp2 = np.einsum('ld,lidc->ic',self.x1_C,self.Loovv)
        tmp2 = np.einsum('kc,ic->ik',self.x1_B,tmp2)
        self.G -= np.einsum('ik,ik->',tmp,tmp2)

        tmp = np.einsum('ijab,jibc->ac',self.x2_A,self.l2)
        tmp = np.einsum('ac,kc->ka',tmp,self.x1_B)
        tmp2 = np.einsum('ld,lkda->ka',self.x1_C,self.Loovv)
        self.G -= np.einsum('ka,ka->',tmp,tmp2)

        tmp = np.einsum('ijab,klab->ijkl',self.x2_A,self.Goovv)
        tmp2 = np.einsum('kc,ijcd->ijkd',self.x1_B,self.l2)
        tmp2 = np.einsum('ld,ijkd->ijkl',self.x1_C,tmp2)
        self.G += np.einsum('ijkl,ijkl->',tmp,tmp2)

        tmp = np.einsum('kc,jlac->jlak',self.x1_B,self.Goovv)
        tmp = np.einsum('ijab,jlak->ilbk',self.x2_A,tmp)
        tmp2 = np.einsum('ikbd,ld->ilbk',self.l2,self.x1_C)
        self.G += np.einsum('ilbk,ilbk->',tmp,tmp2)

        tmp = np.einsum('kc,ljac->ljak',self.x1_B,self.Goovv)
        tmp = np.einsum('ijab,ljak->ilbk',self.x2_A,tmp)
        tmp2 = np.einsum('ikdb,ld->ilbk',self.l2,self.x1_C)
        self.G += np.einsum('ilbk,ilbk->',tmp,tmp2)

        tmp = np.einsum('ld,jkad->jkal',self.x1_C,self.Goovv)
        tmp = np.einsum('ijab,jkal->ikbl',self.x2_A,tmp)
        tmp2 = np.einsum('kc,ilbc->ilbk',self.x1_B,self.l2)
        self.G += np.einsum('ikbl,ilbk->',tmp,tmp2)

        tmp = np.einsum('ld,kjad->kjal',self.x1_C,self.Goovv)
        tmp = np.einsum('ijab,kjal->iklb',self.x2_A,tmp)
        tmp2 = np.einsum('kc,ilcb->ilkb',self.x1_B,self.l2)
        self.G += np.einsum('iklb,ilkb->',tmp,tmp2)

        tmp = np.einsum('kc,ijcd->ijkd',self.x1_B,self.Goovv)
        tmp = np.einsum('ld,ijkd->ijkl',self.x1_C,tmp)
        tmp2 = np.einsum('ijab,klab->ijkl',self.x2_A,self.l2)
        self.G += np.einsum('ijkl,ijkl->',tmp,tmp2)

        # <L2(0)|[[[H_bar,X1(A)],X2(B)],X1(C)]|0>
        tmp = np.einsum('kc,jlbc->jlbk',self.x1_A,self.l2)
        tmp2 = np.einsum('ld,ikad->ikal',self.x1_C,self.Loovv)
        tmp2 = np.einsum('ijab,ikal->jlbk',self.x2_B,tmp2)
        self.G -= np.einsum('jlbk,jlbk->',tmp,tmp2)

        tmp = np.einsum('ld,jkbd->jkbl',self.x1_C,self.l2)
        tmp2 = np.einsum('kc,ilac->ilak',self.x1_A,self.Loovv)
        tmp2 = np.einsum('ijab,ilak->jkbl',self.x2_B,tmp2)
        self.G -= np.einsum('jkbl,jkbl->',tmp,tmp2)

        tmp = np.einsum('ijab,jibd->ad',self.x2_B,self.l2)
        tmp = np.einsum('ld,ad->la',self.x1_C,tmp)
        tmp2 = np.einsum('klca,kc->la',self.Loovv,self.x1_A)
        self.G -= np.einsum('la,la->',tmp,tmp2)

        tmp = np.einsum('ijab,jlba->il',self.x2_B,self.l2)
        tmp2 = np.einsum('kc,kicd->id',self.x1_A,self.Loovv)
        tmp2 = np.einsum('ld,id->il',self.x1_C,tmp2)
        self.G -= np.einsum('il,il->',tmp,tmp2)

        tmp = np.einsum('ijab,jkba->ik',self.x2_B,self.l2)
        tmp2 = np.einsum('ld,lidc->ic',self.x1_C,self.Loovv)
        tmp2 = np.einsum('kc,ic->ik',self.x1_A,tmp2)
        self.G -= np.einsum('ik,ik->',tmp,tmp2)

        tmp = np.einsum('ijab,jibc->ac',self.x2_B,self.l2)
        tmp = np.einsum('ac,kc->ka',tmp,self.x1_A)
        tmp2 = np.einsum('ld,lkda->ka',self.x1_C,self.Loovv)
        self.G -= np.einsum('ka,ka->',tmp,tmp2)

        tmp = np.einsum('ijab,klab->ijkl',self.x2_B,self.Goovv)
        tmp2 = np.einsum('kc,ijcd->ijkd',self.x1_A,self.l2)
        tmp2 = np.einsum('ld,ijkd->ijkl',self.x1_C,tmp2)
        self.G += np.einsum('ijkl,ijkl->',tmp,tmp2)

        tmp = np.einsum('kc,jlac->jlak',self.x1_A,self.Goovv)
        tmp = np.einsum('ijab,jlak->ilbk',self.x2_B,tmp)
        tmp2 = np.einsum('ikbd,ld->ilbk',self.l2,self.x1_C)
        self.G += np.einsum('ilbk,ilbk->',tmp,tmp2)

        tmp  = np.einsum('kc,ljac->ljak',self.x1_A,self.Goovv)
        tmp  = np.einsum('ijab,ljak->ilbk',self.x2_B,tmp)
        tmp2 = np.einsum('ikdb,ld->ilbk',self.l2,self.x1_C)
        self.G += np.einsum('ilbk,ilbk->',tmp,tmp2)

        tmp = np.einsum('ld,jkad->jkal',self.x1_C,self.Goovv)
        tmp = np.einsum('ijab,jkal->ikbl',self.x2_B,tmp)
        tmp2 = np.einsum('kc,ilbc->ilbk',self.x1_A,self.l2)
        self.G += np.einsum('ikbl,ilbk->',tmp,tmp2)

        tmp = np.einsum('ld,kjad->kjal',self.x1_C,self.Goovv)
        tmp = np.einsum('ijab,kjal->iklb',self.x2_B,tmp)
        tmp2 = np.einsum('kc,ilcb->ilkb',self.x1_A,self.l2)
        self.G += np.einsum('iklb,ilkb->',tmp,tmp2)

        tmp = np.einsum('kc,ijcd->ijkd',self.x1_A,self.Goovv)
        tmp = np.einsum('ld,ijkd->ijkl',self.x1_C,tmp)
        tmp2 = np.einsum('ijab,klab->ijkl',self.x2_B,self.l2)
        self.G += np.einsum('ijkl,ijkl->',tmp,tmp2)
 
        # <L2(0)|[[[H_bar,X1(A)],X1(B)],X2(C)]|0>
        tmp = np.einsum('kc,jlbc->jlbk',self.x1_A,self.l2)
        tmp2 = np.einsum('ld,ikad->ikal',self.x1_B,self.Loovv)
        tmp2 = np.einsum('ijab,ikal->jlbk',self.x2_C,tmp2)
        self.G -= np.einsum('jlbk,jlbk->',tmp,tmp2)

        tmp = np.einsum('ld,jkbd->jkbl',self.x1_B,self.l2)
        tmp2 = np.einsum('kc,ilac->ilak',self.x1_A,self.Loovv)
        tmp2 = np.einsum('ijab,ilak->jkbl',self.x2_C,tmp2)
        self.G -= np.einsum('jkbl,jkbl->',tmp,tmp2)

        tmp = np.einsum('ijab,jibd->ad',self.x2_C,self.l2)
        tmp = np.einsum('ld,ad->la',self.x1_B,tmp)
        tmp2 = np.einsum('klca,kc->la',self.Loovv,self.x1_A)
        self.G -= np.einsum('la,la->',tmp,tmp2)

        tmp = np.einsum('ijab,jlba->il',self.x2_C,self.l2)
        tmp2 = np.einsum('kc,kicd->id',self.x1_A,self.Loovv)
        tmp2 = np.einsum('ld,id->il',self.x1_B,tmp2)
        self.G -= np.einsum('il,il->',tmp,tmp2)

        tmp = np.einsum('ijab,jkba->ik',self.x2_C,self.l2)
        tmp2 = np.einsum('ld,lidc->ic',self.x1_B,self.Loovv)
        tmp2 = np.einsum('kc,ic->ik',self.x1_A,tmp2)
        self.G -= np.einsum('ik,ik->',tmp,tmp2)

        tmp = np.einsum('ijab,jibc->ac',self.x2_C,self.l2)
        tmp = np.einsum('ac,kc->ka',tmp,self.x1_A)
        tmp2 = np.einsum('ld,lkda->ka',self.x1_B,self.Loovv)
        self.G -= np.einsum('ka,ka->',tmp,tmp2)

        tmp = np.einsum('ijab,klab->ijkl',self.x2_C,self.Goovv)
        tmp2 = np.einsum('kc,ijcd->ijkd',self.x1_A,self.l2)
        tmp2 = np.einsum('ld,ijkd->ijkl',self.x1_B,tmp2)
        self.G += np.einsum('ijkl,ijkl->',tmp,tmp2)

        tmp = np.einsum('kc,jlac->jlak',self.x1_A,self.Goovv)
        tmp = np.einsum('ijab,jlak->ilbk',self.x2_C,tmp)
        tmp2 = np.einsum('ikbd,ld->ilbk',self.l2,self.x1_B)
        self.G += np.einsum('ilbk,ilbk->',tmp,tmp2)

        tmp  = np.einsum('kc,ljac->ljak',self.x1_A,self.Goovv)
        tmp  = np.einsum('ijab,ljak->ilbk',self.x2_C,tmp)
        tmp2 = np.einsum('ikdb,ld->ilbk',self.l2,self.x1_B)
        self.G += np.einsum('ilbk,ilbk->',tmp,tmp2)

        tmp = np.einsum('ld,jkad->jkal',self.x1_B,self.Goovv)
        tmp = np.einsum('ijab,jkal->ikbl',self.x2_C,tmp)
        tmp2 = np.einsum('kc,ilbc->ilbk',self.x1_A,self.l2)
        self.G += np.einsum('ikbl,ilbk->',tmp,tmp2)

        tmp = np.einsum('ld,kjad->kjal',self.x1_B,self.Goovv)
        tmp = np.einsum('ijab,kjal->iklb',self.x2_C,tmp)
        tmp2 = np.einsum('kc,ilcb->ilkb',self.x1_A,self.l2)
        self.G += np.einsum('iklb,ilkb->',tmp,tmp2)

        tmp = np.einsum('kc,ijcd->ijkd',self.x1_A,self.Goovv)
        tmp = np.einsum('ld,ijkd->ijkl',self.x1_B,tmp)
        tmp2 = np.einsum('ijab,klab->ijkl',self.x2_C,self.l2)
        self.G += np.einsum('ijkl,ijkl->',tmp,tmp2)

        self.hyper += self.G

        # <O|L1(A)[[Hbar(0),X1(B),X1(C)]]|0>
        tmp  = -1.0*np.einsum('jc,kb->jkcb',self.Hov,self.y1_A)
        tmp -= np.einsum('jc,kb->jkcb',self.y1_A,self.Hov)
        tmp -= 2.0*np.einsum('kjib,ic->jkcb',self.Hooov,self.y1_A)
        tmp += np.einsum('jkib,ic->jkcb',self.Hooov,self.y1_A)         
        tmp -= 2.0*np.einsum('jkic,ib->jkcb',self.Hooov,self.y1_A) 
        tmp = np.einsum('kjic,ib->jkcb',self.Hooov,self.y1_A)         
        tmp += 2.0*np.einsum('ajcb,ka->jkcb',self.Hvovv,self.y1_A)
        tmp -= np.einsum('ajbc,ka->jkcb',self.Hvovv,self.y1_A)
        tmp += 2.0*np.einsum('akbc,ja->jkcb',self.Hvovv,self.y1_A)
        tmp -= np.einsum('akcb,ja->jkcb',self.Hvovv,self.y1_A)

        tmp2 = np.einsum('miae,me->ia',tmp,self.x1_B)         
        self.Bcon1 = ndot('ia,ia->',tmp2,self.x1_C)

        # <O|L2(A)|[[Hbar(0),X1(B)],X1(C)]|0>
        tmp   = -1.0*np.einsum('janc,nkba->jckb',self.Hovov,self.y2_A) 
        tmp  -= np.einsum('kanb,njca->jckb',self.Hovov,self.y2_A)
        tmp  -= np.einsum('jacn,nkab->jckb',self.Hovvo,self.y2_A)
        tmp  -= np.einsum('kabn,njac->jckb',self.Hovvo,self.y2_A)
        tmp  += 0.5*np.einsum('fabc,jkfa->jckb',self.Hvvvv,self.y2_A)
        tmp  += 0.5*np.einsum('facb,kjfa->jckb',self.Hvvvv,self.y2_A)
        tmp  += 0.5*np.einsum('kjin,nibc->jckb',self.Hoooo,self.y2_A)        
        tmp  += 0.5*np.einsum('jkin,nicb->jckb',self.Hoooo,self.y2_A)
        tmp2 = np.einsum('iema,me->ia',tmp,self.x1_B)       
        self.Bcon1 += ndot('ia,ia->', tmp2, self.x1_C)

        tmp = np.einsum('ijab,ijdb->ad',self.t2,self.y2_A)
        tmp = np.einsum('ld,ad->la',self.x1_C,tmp)
        tmp = np.einsum('la,klca->kc',tmp,self.Loovv)
        self.Bcon1 -= ndot('kc,kc->',tmp,self.x1_B)        

        tmp = np.einsum('ijab,jlba->il',self.t2,self.y2_A)
        tmp2 = np.einsum('kc,kicd->id',self.x1_B,self.Loovv)
        tmp2 = np.einsum('id,ld->il',tmp2,self.x1_C)
        self.Bcon1 -= ndot('il,il->',tmp2,tmp)     

        tmp = np.einsum('ijab,jkba->ik',self.t2,self.y2_A)
        tmp2 = np.einsum('ld,lidc->ic',self.x1_C,self.Loovv)
        tmp2 = np.einsum('ic,kc->ik',tmp2,self.x1_B)
        self.Bcon1 -= ndot('ik,ik->',tmp2,tmp)  

        tmp = np.einsum('ijab,ijcb->ac',self.t2,self.y2_A)
        tmp = np.einsum('kc,ac->ka',self.x1_B,tmp)
        tmp2 = np.einsum('ld,lkda->ka',self.x1_C,self.Loovv)
        self.Bcon1 -= ndot('ka,ka->',tmp2,tmp)

        # <O|L2(A)[[Hbar(0),X2(B)],X2(C)]|0>
        tmp = np.einsum("klcd,ijcd->ijkl",self.x2_C,self.y2_A)   
        tmp = np.einsum("ijkl,ijab->klab",tmp,self.x2_B)         
        self.Bcon1 += 0.5*np.einsum('klab,klab->',tmp,self.Goovv)        

        tmp = np.einsum("ijab,ikbd->jkad",self.x2_B,self.y2_A)   
        tmp = np.einsum("jkad,klcd->jlac",tmp,self.x2_C)         
        self.Bcon1 += np.einsum('jlac,jlac->',tmp,self.Goovv)     

        tmp = np.einsum("klcd,ikdb->licb",self.x2_C,self.y2_A)   
        tmp = np.einsum("licb,ijab->ljca",tmp,self.x2_B)         
        self.Bcon1 += np.einsum('ljca,ljac->',tmp,self.Goovv) 

        tmp = np.einsum("ijab,klab->ijkl",self.x2_B,self.y2_A)   
        tmp = np.einsum("ijkl,klcd->ijcd",tmp,self.x2_C)         
        self.Bcon1 += 0.5*np.einsum('ijcd,ijcd->',tmp,self.Goovv)    

        tmp = np.einsum("ijab,ijac->bc",self.x2_B,self.Loovv)  
        tmp = np.einsum("bc,klcd->klbd",tmp,self.x2_C)
        self.Bcon1 -= np.einsum("klbd,klbd->",tmp,self.y2_A)
        tmp = np.einsum("ijab,ikab->jk",self.x2_B,self.Loovv)
        tmp = np.einsum("jk,klcd->jlcd",tmp,self.x2_C)
        self.Bcon1 -= np.einsum("jlcd,jlcd->",tmp,self.y2_A)
        tmp = np.einsum("ikbc,klcd->ilbd",self.Loovv,self.x2_C)
        tmp = np.einsum("ilbd,ijab->jlad",tmp,self.x2_B)       
        self.Bcon1 -= np.einsum("jlad,jlad->",tmp,self.y2_A)
        tmp = np.einsum("ijab,jlbc->ilac",self.x2_B,self.y2_A)
        tmp = np.einsum("ilac,klcd->ikad",tmp,self.x2_C)
        self.Bcon1 -= np.einsum("ikad,ikad->",tmp,self.Loovv)
        tmp = np.einsum("klca,klcd->ad",self.Loovv,self.x2_C)
        tmp = np.einsum("ad,ijdb->ijab",tmp,self.y2_A)
        self.Bcon1 -= np.einsum("ijab,ijab->",tmp,self.x2_B)
        tmp = np.einsum("kicd,klcd->il",self.Loovv,self.x2_C)
        tmp = np.einsum("ijab,il->ljab",self.x2_B,tmp)
        self.Bcon1 -= np.einsum("ljab,ljab->",tmp,self.y2_A)

        tmp = np.einsum("klcd,ikac->lida",self.x2_C,self.y2_A)
        tmp = np.einsum("lida,jlbd->ijab",tmp,self.Loovv)
        self.Bcon1 += 2.*np.einsum("ijab,ijab->",tmp,self.x2_B) 

        # <O|L1(A)[[Hbar(0),X1(B)],X2(C)]]|0> 
        tmp  = 2.*np.einsum("jkbc,kc->jb",self.x2_C,self.y1_A)         
        tmp -= np.einsum("jkcb,kc->jb",self.x2_C,self.y1_A)         
        tmp = np.einsum('ijab,jb->ia',self.Loovv,tmp)
        self.Bcon1 += np.einsum("ia,ia->",tmp,self.x1_B)

        tmp = np.einsum("jkbc,jkba->ca",self.x2_C,self.Loovv)
        tmp = np.einsum("ia,ca->ic",self.x1_B,tmp)
        self.Bcon1 -= np.einsum("ic,ic->",tmp,self.y1_A)

        tmp = np.einsum("jkbc,jibc->ki",self.x2_C,self.Loovv)
        tmp = np.einsum("ki,ia->ka",tmp,self.x1_B)
        self.Bcon1 -= np.einsum("ka,ka->",tmp,self.y1_A) 

        # <O|L2(A)[[Hbar(0),X1(B)],X2(C)]]|0>
        tmp = np.einsum("klcd,lkdb->cb",self.x2_C,self.y2_A)
        tmp = np.einsum("jb,cb->jc",self.x1_B,tmp)
        self.Bcon1 -= np.einsum("jc,jc->",tmp,self.Hov)
        
        tmp = np.einsum("klcd,ljdc->kj",self.x2_C,self.y2_A)
        tmp = np.einsum("kj,jb->kb",tmp,self.x1_B)
        self.Bcon1 -= np.einsum("kb,kb->",tmp,self.Hov)

        tmp = np.einsum('lkda,klcd->ac',self.y2_A,self.x2_C)
        tmp2 = np.einsum('jb,ajcb->ac',self.x1_B,self.Hvovv)
        self.Bcon1 += 2.*np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('lkda,klcd->ac',self.y2_A,self.x2_C)
        tmp2 = np.einsum('jb,ajbc->ac',self.x1_B,self.Hvovv)
        self.Bcon1 -= np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('jb,ljda->lbda',self.x1_B,self.y2_A)
        tmp2 = 2.*np.einsum('klcd,akbc->ldab',self.x2_C,self.Hvovv)
        tmp2 -= np.einsum('klcd,akcb->ldab',self.x2_C,self.Hvovv)
        self.Bcon1 += np.einsum('lbda,ldab->',tmp,tmp2)

        tmp = np.einsum('ia,fkba->fkbi',self.x1_B,self.Hvovv)
        tmp = np.einsum('fkbi,jifc->kjbc',tmp,self.y2_A)
        self.Bcon1 -= np.einsum('jkbc,kjbc->',self.x2_C,tmp)

        tmp = np.einsum('ia,fjac->fjic',self.x1_B,self.Hvovv)
        tmp = np.einsum('fjic,ikfb->jkbc',tmp,self.y2_A)
        self.Bcon1 -= np.einsum('jkbc,jkbc->',self.x2_C,tmp)

        tmp = np.einsum('ia,jkfa->jkfi',self.x1_B,self.y2_A)
        tmp2 = np.einsum('jkbc,fibc->jkfi',self.x2_C,self.Hvovv)
        self.Bcon1 -= np.einsum('jkfi,jkfi->',tmp2,tmp)
 
        tmp = np.einsum('jb,kjib->ki',self.x1_B,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_C,self.y2_A)
        self.Bcon1 -= 2.*np.einsum('ki,ki->',tmp,tmp2)       

        tmp = np.einsum('jb,jkib->ki',self.x1_B,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_C,self.y2_A)
        self.Bcon1 += np.einsum('ki,ki->',tmp,tmp2)         
        
        tmp  = 2.*np.einsum('jkic,klcd->jild',self.Hooov,self.x2_C)
        tmp -= np.einsum('kjic,klcd->jild',self.Hooov,self.x2_C)
        tmp  = np.einsum('jild,jb->bild',tmp,self.x1_B)
        self.Bcon1 -= np.einsum('bild,ilbd->',tmp,self.y2_A)  
       
        tmp  = np.einsum('ia,jkna->jkni',self.x1_B,self.Hooov) 
        tmp2  = np.einsum('jkbc,nibc->jkni',self.x2_C,self.y2_A)
        self.Bcon1 += np.einsum('jkni,jkni->',tmp2,tmp)

        tmp  = np.einsum('ia,nkab->nkib',self.x1_B,self.y2_A)          
        tmp  = np.einsum('jkbc,nkib->jnic',self.x2_C,tmp)
        self.Bcon1 += np.einsum('jnic,ijnc->',tmp,self.Hooov)

        tmp  = np.einsum('ia,nkba->nkbi',self.x1_B,self.y2_A)         
        tmp  = np.einsum('jkbc,nkbi->jnci',self.x2_C,tmp)
        self.Bcon1 += np.einsum('jnci,jinc->',tmp,self.Hooov)
        
        # <O|L1(A)[[Hbar(0),X2(B)],X1(C)]]|0>
        tmp  = 2.*np.einsum("jkbc,kc->jb",self.x2_B,self.y1_A)
        tmp -= np.einsum("jkcb,kc->jb",self.x2_B,self.y1_A)
        tmp = np.einsum('ijab,jb->ia',self.Loovv,tmp)
        self.Bcon1 += np.einsum("ia,ia->",tmp,self.x1_C)

        tmp = np.einsum("jkbc,jkba->ca",self.x2_B,self.Loovv)
        tmp = np.einsum("ia,ca->ic",self.x1_C,tmp)
        self.Bcon1 -= np.einsum("ic,ic->",tmp,self.y1_A)

        tmp = np.einsum("jkbc,jibc->ki",self.x2_B,self.Loovv)
        tmp = np.einsum("ki,ia->ka",tmp,self.x1_C)
        self.Bcon1 -= np.einsum("ka,ka->",tmp,self.y1_A)

        # <O|L2(A)[[Hbar(0),X2(B)],X1(C)]]|0>
        tmp = np.einsum("klcd,lkdb->cb",self.x2_B,self.y2_A)
        tmp = np.einsum("jb,cb->jc",self.x1_C,tmp)
        self.Bcon1 -= np.einsum("jc,jc->",tmp,self.Hov)

        tmp = np.einsum("klcd,ljdc->kj",self.x2_B,self.y2_A)
        tmp = np.einsum("kj,jb->kb",tmp,self.x1_C)
        self.Bcon1 -= np.einsum("kb,kb->",tmp,self.Hov)

        tmp = np.einsum('lkda,klcd->ac',self.y2_A,self.x2_B)
        tmp2 = np.einsum('jb,ajcb->ac',self.x1_C,self.Hvovv)
        self.Bcon1 += 2.*np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('lkda,klcd->ac',self.y2_A,self.x2_B)
        tmp2 = np.einsum('jb,ajbc->ac',self.x1_C,self.Hvovv)
        self.Bcon1 -= np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('jb,ljda->lbda',self.x1_C,self.y2_A)
        tmp2 = 2.*np.einsum('klcd,akbc->ldab',self.x2_B,self.Hvovv)
        tmp2 -= np.einsum('klcd,akcb->ldab',self.x2_B,self.Hvovv)
        self.Bcon1 += np.einsum('lbda,ldab->',tmp,tmp2)

        tmp = np.einsum('ia,fkba->fkbi',self.x1_C,self.Hvovv)
        tmp = np.einsum('fkbi,jifc->kjbc',tmp,self.y2_A)
        self.Bcon1 -= np.einsum('jkbc,kjbc->',self.x2_B,tmp)

        tmp = np.einsum('ia,fjac->fjic',self.x1_C,self.Hvovv)
        tmp = np.einsum('fjic,ikfb->jkbc',tmp,self.y2_A)
        self.Bcon1 -= np.einsum('jkbc,jkbc->',self.x2_B,tmp)

        tmp = np.einsum('ia,jkfa->jkfi',self.x1_C,self.y2_A)
        tmp2 = np.einsum('jkbc,fibc->jkfi',self.x2_B,self.Hvovv)
        self.Bcon1 -= np.einsum('jkfi,jkfi->',tmp2,tmp)

        tmp = np.einsum('jb,kjib->ki',self.x1_C,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_B,self.y2_A)
        self.Bcon1 -= 2.*np.einsum('ki,ki->',tmp,tmp2)       

        tmp = np.einsum('jb,jkib->ki',self.x1_C,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_B,self.y2_A)
        self.Bcon1 += np.einsum('ki,ki->',tmp,tmp2)          

        tmp  = 2.*np.einsum('jkic,klcd->jild',self.Hooov,self.x2_B)
        tmp -= np.einsum('kjic,klcd->jild',self.Hooov,self.x2_B)
        tmp  = np.einsum('jild,jb->bild',tmp,self.x1_C)
        self.Bcon1 -= np.einsum('bild,ilbd->',tmp,self.y2_A)  

        tmp  = np.einsum('ia,jkna->jkni',self.x1_C,self.Hooov)
        tmp2  = np.einsum('jkbc,nibc->jkni',self.x2_B,self.y2_A)
        self.Bcon1 += np.einsum('jkni,jkni->',tmp2,tmp)

        tmp  = np.einsum('ia,nkab->nkib',self.x1_C,self.y2_A)
        tmp  = np.einsum('jkbc,nkib->jnic',self.x2_B,tmp)
        self.Bcon1 += np.einsum('jnic,ijnc->',tmp,self.Hooov)

        tmp  = np.einsum('ia,nkba->nkbi',self.x1_C,self.y2_A)
        tmp  = np.einsum('jkbc,nkbi->jnci',self.x2_B,tmp)
        self.Bcon1 += np.einsum('jnci,jinc->',tmp,self.Hooov)

        # <O|L1(B)[[Hbar(0),X1(A),X1(C)]]|0>
        tmp  = -1.0*np.einsum('jc,kb->jkcb',self.Hov,self.y1_B)
        tmp -= np.einsum('jc,kb->jkcb',self.y1_B,self.Hov)
        tmp -= 2.0*np.einsum('kjib,ic->jkcb',self.Hooov,self.y1_B)
        tmp += np.einsum('jkib,ic->jkcb',self.Hooov,self.y1_B)
        tmp -= 2.0*np.einsum('jkic,ib->jkcb',self.Hooov,self.y1_B)
        tmp += np.einsum('kjic,ib->jkcb',self.Hooov,self.y1_B)
        tmp += 2.0*np.einsum('ajcb,ka->jkcb',self.Hvovv,self.y1_B)
        tmp -= np.einsum('ajbc,ka->jkcb',self.Hvovv,self.y1_B)
        tmp += 2.0*np.einsum('akbc,ja->jkcb',self.Hvovv,self.y1_B)
        tmp -= np.einsum('akcb,ja->jkcb',self.Hvovv,self.y1_B)

        tmp2 = np.einsum('miae,me->ia',tmp,self.x1_A)         
        self.Bcon2 = ndot('ia,ia->',tmp2,self.x1_C)

        # <O|L2(B)|[[Hbar(0),X1(A)],X1(C)]|0>
        tmp   = -1.0*np.einsum('janc,nkba->jckb',self.Hovov,self.y2_B) 
        tmp  -= np.einsum('kanb,njca->jckb',self.Hovov,self.y2_B)
        tmp  -= np.einsum('jacn,nkab->jckb',self.Hovvo,self.y2_B)
        tmp  -= np.einsum('kabn,njac->jckb',self.Hovvo,self.y2_B)
        tmp  += 0.5*np.einsum('fabc,jkfa->jckb',self.Hvvvv,self.y2_B)
        tmp  += 0.5*np.einsum('facb,kjfa->jckb',self.Hvvvv,self.y2_B)
        tmp  += 0.5*np.einsum('kjin,nibc->jckb',self.Hoooo,self.y2_B)
        tmp  += 0.5*np.einsum('jkin,nicb->jckb',self.Hoooo,self.y2_B)
        tmp2 = np.einsum('iema,me->ia',tmp,self.x1_A)       
        self.Bcon2 += ndot('ia,ia->', tmp2, self.x1_C)

        tmp = np.einsum('ijab,ijdb->ad',self.t2,self.y2_B)
        tmp = np.einsum('ld,ad->la',self.x1_C,tmp)
        tmp = np.einsum('la,klca->kc',tmp,self.Loovv)
        self.Bcon2 -= ndot('kc,kc->',tmp,self.x1_A)

        tmp = np.einsum('ijab,jlba->il',self.t2,self.y2_B)
        tmp2 = np.einsum('kc,kicd->id',self.x1_A,self.Loovv)
        tmp2 = np.einsum('id,ld->il',tmp2,self.x1_C)
        self.Bcon2 -= ndot('il,il->',tmp2,tmp)

        tmp = np.einsum('ijab,jkba->ik',self.t2,self.y2_B)
        tmp2 = np.einsum('ld,lidc->ic',self.x1_C,self.Loovv)
        tmp2 = np.einsum('ic,kc->ik',tmp2,self.x1_A)
        self.Bcon2 -= ndot('ik,ik->',tmp2,tmp)

        tmp = np.einsum('ijab,ijcb->ac',self.t2,self.y2_B)
        tmp = np.einsum('kc,ac->ka',self.x1_A,tmp)
        tmp2 = np.einsum('ld,lkda->ka',self.x1_C,self.Loovv)
        self.Bcon2 -= ndot('ka,ka->',tmp2,tmp)

        # <O|L2(B)[[Hbar(0),X2(A)],X2(C)]|0>
        tmp = np.einsum("klcd,ijcd->ijkl",self.x2_C,self.y2_B)   
        tmp = np.einsum("ijkl,ijab->klab",tmp,self.x2_A)         
        self.Bcon2 += 0.5*np.einsum('klab,klab->',tmp,self.Goovv)

        tmp = np.einsum("ijab,ikbd->jkad",self.x2_A,self.y2_B)   
        tmp = np.einsum("jkad,klcd->jlac",tmp,self.x2_C)         
        self.Bcon2 += np.einsum('jlac,jlac->',tmp,self.Goovv)

        tmp = np.einsum("klcd,ikdb->licb",self.x2_C,self.y2_B)   
        tmp = np.einsum("licb,ijab->ljca",tmp,self.x2_A)         
        self.Bcon2 += np.einsum('ljca,ljac->',tmp,self.Goovv)

        tmp = np.einsum("ijab,klab->ijkl",self.x2_A,self.y2_B)   
        tmp = np.einsum("ijkl,klcd->ijcd",tmp,self.x2_C)         
        self.Bcon2 += 0.5*np.einsum('ijcd,ijcd->',tmp,self.Goovv)
         
        tmp = np.einsum("ijab,ijac->bc",self.x2_A,self.Loovv)
        tmp = np.einsum("bc,klcd->klbd",tmp,self.x2_C)
        self.Bcon2 -= np.einsum("klbd,klbd->",tmp,self.y2_B)
        tmp = np.einsum("ijab,ikab->jk",self.x2_A,self.Loovv)
        tmp = np.einsum("jk,klcd->jlcd",tmp,self.x2_C)
        self.Bcon2 -= np.einsum("jlcd,jlcd->",tmp,self.y2_B)
        tmp = np.einsum("ikbc,klcd->ilbd",self.Loovv,self.x2_C)
        tmp = np.einsum("ilbd,ijab->jlad",tmp,self.x2_A)       
        self.Bcon2 -= np.einsum("jlad,jlad->",tmp,self.y2_B) 
        tmp = np.einsum("ijab,jlbc->ilac",self.x2_A,self.y2_B)
        tmp = np.einsum("ilac,klcd->ikad",tmp,self.x2_C)
        self.Bcon2 -= np.einsum("ikad,ikad->",tmp,self.Loovv)
        tmp = np.einsum("klca,klcd->ad",self.Loovv,self.x2_C)
        tmp = np.einsum("ad,ijdb->ijab",tmp,self.y2_B)
        self.Bcon2 -= np.einsum("ijab,ijab->",tmp,self.x2_A)
        tmp = np.einsum("kicd,klcd->il",self.Loovv,self.x2_C)
        tmp = np.einsum("ijab,il->ljab",self.x2_A,tmp)
        self.Bcon2 -= np.einsum("ljab,ljab->",tmp,self.y2_B)

        tmp = np.einsum("klcd,ikac->lida",self.x2_C,self.y2_B)
        tmp = np.einsum("lida,jlbd->ijab",tmp,self.Loovv)
        self.Bcon2 += 2.*np.einsum("ijab,ijab->",tmp,self.x2_A)

        # <O|L1(B)[[Hbar(0),X1(A)],X2(C)]]|0> 
        tmp = 2.*np.einsum("jkbc,kc->jb",self.x2_C,self.y1_B)
        tmp -= np.einsum("jkcb,kc->jb",self.x2_C,self.y1_B)
        tmp = np.einsum('ijab,jb->ia',self.Loovv,tmp)
        self.Bcon2 += np.einsum("ia,ia->",tmp,self.x1_A)

        tmp = np.einsum("jkbc,jkba->ca",self.x2_C,self.Loovv)
        tmp = np.einsum("ia,ca->ic",self.x1_A,tmp)
        self.Bcon2 -= np.einsum("ic,ic->",tmp,self.y1_B)

        tmp = np.einsum("jkbc,jibc->ki",self.x2_C,self.Loovv)
        tmp = np.einsum("ki,ia->ka",tmp,self.x1_A)
        self.Bcon2 -= np.einsum("ka,ka->",tmp,self.y1_B) 

        # <O|L2(B)[[Hbar(0),X1(A)],X2(C)]]|0>
        tmp = np.einsum("klcd,lkdb->cb",self.x2_C,self.y2_B)
        tmp = np.einsum("jb,cb->jc",self.x1_A,tmp)
        self.Bcon2 -= np.einsum("jc,jc->",tmp,self.Hov)

        tmp = np.einsum("klcd,ljdc->kj",self.x2_C,self.y2_B)
        tmp = np.einsum("kj,jb->kb",tmp,self.x1_A)
        self.Bcon2 -= np.einsum("kb,kb->",tmp,self.Hov)

        tmp = np.einsum('lkda,klcd->ac',self.y2_B,self.x2_C)
        tmp2 = np.einsum('jb,ajcb->ac',self.x1_A,self.Hvovv)
        self.Bcon2 += 2.*np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('lkda,klcd->ac',self.y2_B,self.x2_C)
        tmp2 = np.einsum('jb,ajbc->ac',self.x1_A,self.Hvovv)
        self.Bcon2 -= np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('jb,ljda->lbda',self.x1_A,self.y2_B)
        tmp2 = 2.*np.einsum('klcd,akbc->ldab',self.x2_C,self.Hvovv)
        tmp2 -= np.einsum('klcd,akcb->ldab',self.x2_C,self.Hvovv)
        self.Bcon2 += np.einsum('lbda,ldab->',tmp,tmp2)

        tmp = np.einsum('ia,fkba->fkbi',self.x1_A,self.Hvovv)
        tmp = np.einsum('fkbi,jifc->kjbc',tmp,self.y2_B)
        self.Bcon2 -= np.einsum('jkbc,kjbc->',self.x2_C,tmp)

        tmp = np.einsum('ia,fjac->fjic',self.x1_A,self.Hvovv)
        tmp = np.einsum('fjic,ikfb->jkbc',tmp,self.y2_B)
        self.Bcon2 -= np.einsum('jkbc,jkbc->',self.x2_C,tmp)

        tmp = np.einsum('ia,jkfa->jkfi',self.x1_A,self.y2_B)
        tmp2 = np.einsum('jkbc,fibc->jkfi',self.x2_C,self.Hvovv)
        self.Bcon2 -= np.einsum('jkfi,jkfi->',tmp2,tmp)

        tmp = np.einsum('jb,kjib->ki',self.x1_A,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_C,self.y2_B)
        self.Bcon2 -= 2.*np.einsum('ki,ki->',tmp,tmp2)       

        tmp = np.einsum('jb,jkib->ki',self.x1_A,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_C,self.y2_B)
        self.Bcon2 += np.einsum('ki,ki->',tmp,tmp2)          

        tmp  = 2.*np.einsum('jkic,klcd->jild',self.Hooov,self.x2_C)
        tmp -= np.einsum('kjic,klcd->jild',self.Hooov,self.x2_C)
        tmp  = np.einsum('jild,jb->bild',tmp,self.x1_A)
        self.Bcon2 -= np.einsum('bild,ilbd->',tmp,self.y2_B)  

        tmp  = np.einsum('ia,jkna->jkni',self.x1_A,self.Hooov)
        tmp2  = np.einsum('jkbc,nibc->jkni',self.x2_C,self.y2_B)
        self.Bcon2 += np.einsum('jkni,jkni->',tmp2,tmp)

        tmp  = np.einsum('ia,nkab->nkib',self.x1_A,self.y2_B)
        tmp  = np.einsum('jkbc,nkib->jnic',self.x2_C,tmp)
        self.Bcon2 += np.einsum('jnic,ijnc->',tmp,self.Hooov)

        tmp  = np.einsum('ia,nkba->nkbi',self.x1_A,self.y2_B)
        tmp  = np.einsum('jkbc,nkbi->jnci',self.x2_C,tmp)
        self.Bcon2 += np.einsum('jnci,jinc->',tmp,self.Hooov)

        # <O|L1(B)[[Hbar(0),X2(A)],X1(C)]]|0> 
        tmp  = 2.*np.einsum("jkbc,kc->jb",self.x2_A,self.y1_B)
        tmp -= np.einsum("jkcb,kc->jb",self.x2_A,self.y1_B)
        tmp = np.einsum('ijab,jb->ia',self.Loovv,tmp)
        self.Bcon2 += np.einsum("ia,ia->",tmp,self.x1_C)

        tmp = np.einsum("jkbc,jkba->ca",self.x2_A,self.Loovv)
        tmp = np.einsum("ia,ca->ic",self.x1_C,tmp)
        self.Bcon2 -= np.einsum("ic,ic->",tmp,self.y1_B)

        tmp = np.einsum("jkbc,jibc->ki",self.x2_A,self.Loovv)
        tmp = np.einsum("ki,ia->ka",tmp,self.x1_C)
        self.Bcon2 -= np.einsum("ka,ka->",tmp,self.y1_B)

        # <O|L2(B)[[Hbar(0),X2(A)],X1(C)]]|0>
        tmp = np.einsum("klcd,lkdb->cb",self.x2_A,self.y2_B)
        tmp = np.einsum("jb,cb->jc",self.x1_C,tmp)
        self.Bcon2 -= np.einsum("jc,jc->",tmp,self.Hov)

        tmp = np.einsum("klcd,ljdc->kj",self.x2_A,self.y2_B)
        tmp = np.einsum("kj,jb->kb",tmp,self.x1_C)
        self.Bcon2 -= np.einsum("kb,kb->",tmp,self.Hov)

        tmp = np.einsum('lkda,klcd->ac',self.y2_B,self.x2_A)
        tmp2 = np.einsum('jb,ajcb->ac',self.x1_C,self.Hvovv)
        self.Bcon2 += 2.*np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('lkda,klcd->ac',self.y2_B,self.x2_A)
        tmp2 = np.einsum('jb,ajbc->ac',self.x1_C,self.Hvovv)
        self.Bcon2 -= np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('jb,ljda->lbda',self.x1_C,self.y2_B)
        tmp2 = 2.*np.einsum('klcd,akbc->ldab',self.x2_A,self.Hvovv)
        tmp2 -= np.einsum('klcd,akcb->ldab',self.x2_A,self.Hvovv)
        self.Bcon2 += np.einsum('lbda,ldab->',tmp,tmp2)

        tmp = np.einsum('ia,fkba->fkbi',self.x1_C,self.Hvovv)
        tmp = np.einsum('fkbi,jifc->kjbc',tmp,self.y2_B)
        self.Bcon2 -= np.einsum('jkbc,kjbc->',self.x2_A,tmp)

        tmp = np.einsum('ia,fjac->fjic',self.x1_C,self.Hvovv)
        tmp = np.einsum('fjic,ikfb->jkbc',tmp,self.y2_B)
        self.Bcon2 -= np.einsum('jkbc,jkbc->',self.x2_A,tmp)

        tmp = np.einsum('ia,jkfa->jkfi',self.x1_C,self.y2_B)
        tmp2 = np.einsum('jkbc,fibc->jkfi',self.x2_A,self.Hvovv)
        self.Bcon2 -= np.einsum('jkfi,jkfi->',tmp2,tmp)

        tmp = np.einsum('jb,kjib->ki',self.x1_C,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_A,self.y2_B)
        self.Bcon2 -= 2.*np.einsum('ki,ki->',tmp,tmp2)       

        tmp = np.einsum('jb,jkib->ki',self.x1_C,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_A,self.y2_B)
        self.Bcon2 += np.einsum('ki,ki->',tmp,tmp2)          

        tmp  = 2.*np.einsum('jkic,klcd->jild',self.Hooov,self.x2_A)
        tmp -= np.einsum('kjic,klcd->jild',self.Hooov,self.x2_A)
        tmp  = np.einsum('jild,jb->bild',tmp,self.x1_C)
        self.Bcon2 -= np.einsum('bild,ilbd->',tmp,self.y2_B)  

        tmp  = np.einsum('ia,jkna->jkni',self.x1_C,self.Hooov)
        tmp2  = np.einsum('jkbc,nibc->jkni',self.x2_A,self.y2_B)
        self.Bcon2 += np.einsum('jkni,jkni->',tmp2,tmp)

        tmp  = np.einsum('ia,nkab->nkib',self.x1_C,self.y2_B)
        tmp  = np.einsum('jkbc,nkib->jnic',self.x2_A,tmp)
        self.Bcon2 += np.einsum('jnic,ijnc->',tmp,self.Hooov)

        tmp  = np.einsum('ia,nkba->nkbi',self.x1_C,self.y2_B)
        tmp  = np.einsum('jkbc,nkbi->jnci',self.x2_A,tmp)
        self.Bcon2 += np.einsum('jnci,jinc->',tmp,self.Hooov)
        
        # <0|L1(C)[[Hbar(0),X1(A),X1(B)]]|0>
        tmp  = -1.0*np.einsum('jc,kb->jkcb',self.Hov,self.y1_C)
        tmp -= np.einsum('jc,kb->jkcb',self.y1_C,self.Hov)
        tmp -= 2.0*np.einsum('kjib,ic->jkcb',self.Hooov,self.y1_C)
        tmp += np.einsum('jkib,ic->jkcb',self.Hooov,self.y1_C)
        tmp -= 2.0*np.einsum('jkic,ib->jkcb',self.Hooov,self.y1_C)
        tmp += np.einsum('kjic,ib->jkcb',self.Hooov,self.y1_C)

        tmp += 2.0*np.einsum('ajcb,ka->jkcb',self.Hvovv,self.y1_C)
        tmp -= np.einsum('ajbc,ka->jkcb',self.Hvovv,self.y1_C)
        tmp += 2.0*np.einsum('akbc,ja->jkcb',self.Hvovv,self.y1_C)
        tmp -= np.einsum('akcb,ja->jkcb',self.Hvovv,self.y1_C)

        tmp2 = np.einsum('miae,me->ia',tmp,self.x1_A)         
        self.Bcon3 = ndot('ia,ia->',tmp2,self.x1_B)

        # <0|L2(C)|[[Hbar(0),X1(A)],X1(B)]|0>
        tmp   = -1.0*np.einsum('janc,nkba->jckb',self.Hovov,self.y2_C) 
        tmp  -= np.einsum('kanb,njca->jckb',self.Hovov,self.y2_C)
        tmp  -= np.einsum('jacn,nkab->jckb',self.Hovvo,self.y2_C)
        tmp  -= np.einsum('kabn,njac->jckb',self.Hovvo,self.y2_C)
        tmp  += 0.5*np.einsum('fabc,jkfa->jckb',self.Hvvvv,self.y2_C)
        tmp  += 0.5*np.einsum('facb,kjfa->jckb',self.Hvvvv,self.y2_C)
        tmp  += 0.5*np.einsum('kjin,nibc->jckb',self.Hoooo,self.y2_C)
        tmp  += 0.5*np.einsum('jkin,nicb->jckb',self.Hoooo,self.y2_C)
        tmp2 = np.einsum('iema,me->ia',tmp,self.x1_A)      
        self.Bcon3 += ndot('ia,ia->', tmp2, self.x1_B)

        tmp = np.einsum('ijab,ijdb->ad',self.t2,self.y2_C)
        tmp = np.einsum('ld,ad->la',self.x1_B,tmp)
        tmp = np.einsum('la,klca->kc',tmp,self.Loovv)
        self.Bcon3 -= ndot('kc,kc->',tmp,self.x1_A)

        tmp = np.einsum('ijab,jlba->il',self.t2,self.y2_C)
        tmp2 = np.einsum('kc,kicd->id',self.x1_A,self.Loovv)
        tmp2 = np.einsum('id,ld->il',tmp2,self.x1_B)
        self.Bcon3 -= ndot('il,il->',tmp2,tmp)

        tmp = np.einsum('ijab,jkba->ik',self.t2,self.y2_C)
        tmp2 = np.einsum('ld,lidc->ic',self.x1_B,self.Loovv)
        tmp2 = np.einsum('ic,kc->ik',tmp2,self.x1_A)
        self.Bcon3 -= ndot('ik,ik->',tmp2,tmp)

        tmp = np.einsum('ijab,ijcb->ac',self.t2,self.y2_C)
        tmp = np.einsum('kc,ac->ka',self.x1_A,tmp)
        tmp2 = np.einsum('ld,lkda->ka',self.x1_B,self.Loovv)
        self.Bcon3 -= ndot('ka,ka->',tmp2,tmp)

        # <0|L2(C)[[Hbar(0),X2(A)],X2(B)]|0>
        tmp = np.einsum("klcd,ijcd->ijkl",self.x2_B,self.y2_C)   
        tmp = np.einsum("ijkl,ijab->klab",tmp,self.x2_A)        
        self.Bcon3 += 0.5*np.einsum('klab,klab->',tmp,self.Goovv)

        tmp = np.einsum("ijab,ikbd->jkad",self.x2_A,self.y2_C)  
        tmp = np.einsum("jkad,klcd->jlac",tmp,self.x2_B)         
        self.Bcon3 += np.einsum('jlac,jlac->',tmp,self.Goovv)

        tmp = np.einsum("klcd,ikdb->licb",self.x2_B,self.y2_C)   
        tmp = np.einsum("licb,ijab->ljca",tmp,self.x2_A)         
        self.Bcon3 += np.einsum('ljca,ljac->',tmp,self.Goovv)
          
        tmp = np.einsum("ijab,klab->ijkl",self.x2_A,self.y2_C)   
        tmp = np.einsum("ijkl,klcd->ijcd",tmp,self.x2_B)         
        self.Bcon3 += 0.5*np.einsum('ijcd,ijcd->',tmp,self.Goovv)
             
        tmp = np.einsum("ijab,ijac->bc",self.x2_A,self.Loovv)
        tmp = np.einsum("bc,klcd->klbd",tmp,self.x2_B)
        self.Bcon3 -= np.einsum("klbd,klbd->",tmp,self.y2_C)
        tmp = np.einsum("ijab,ikab->jk",self.x2_A,self.Loovv)
        tmp = np.einsum("jk,klcd->jlcd",tmp,self.x2_B)
        self.Bcon3 -= np.einsum("jlcd,jlcd->",tmp,self.y2_C)
        tmp = np.einsum("ikbc,klcd->ilbd",self.Loovv,self.x2_B)
        tmp = np.einsum("ilbd,ijab->jlad",tmp,self.x2_A)
        self.Bcon3 -= np.einsum("jlad,jlad->",tmp,self.y2_C) 
        tmp = np.einsum("ijab,jlbc->ilac",self.x2_A,self.y2_C)
        tmp = np.einsum("ilac,klcd->ikad",tmp,self.x2_B)
        self.Bcon3 -= np.einsum("ikad,ikad->",tmp,self.Loovv)
        tmp = np.einsum("klca,klcd->ad",self.Loovv,self.x2_B)
        tmp = np.einsum("ad,ijdb->ijab",tmp,self.y2_C)
        self.Bcon3 -= np.einsum("ijab,ijab->",tmp,self.x2_A)
        tmp = np.einsum("kicd,klcd->il",self.Loovv,self.x2_B)
        tmp = np.einsum("ijab,il->ljab",self.x2_A,tmp)
        self.Bcon3 -= np.einsum("ljab,ljab->",tmp,self.y2_C)

        tmp = np.einsum("klcd,ikac->lida",self.x2_B,self.y2_C)
        tmp = np.einsum("lida,jlbd->ijab",tmp,self.Loovv)
        self.Bcon3 += 2.*np.einsum("ijab,ijab->",tmp,self.x2_A)

        # <0|L1(C)[[Hbar(0),X1(A)],X2(B)]]|0> 
        tmp = 2.*np.einsum("jkbc,kc->jb",self.x2_B,self.y1_C)
        tmp -= np.einsum("jkcb,kc->jb",self.x2_B,self.y1_C)
        tmp = np.einsum('ijab,jb->ia',self.Loovv,tmp)
        self.Bcon3 += np.einsum("ia,ia->",tmp,self.x1_A)

        tmp = np.einsum("jkbc,jkba->ca",self.x2_B,self.Loovv)
        tmp = np.einsum("ia,ca->ic",self.x1_A,tmp)
        self.Bcon3 -= np.einsum("ic,ic->",tmp,self.y1_C)

        tmp = np.einsum("jkbc,jibc->ki",self.x2_B,self.Loovv)
        tmp = np.einsum("ki,ia->ka",tmp,self.x1_A)
        self.Bcon3 -= np.einsum("ka,ka->",tmp,self.y1_C)

        # <0|L2(C)[[Hbar(0),X1(A)],X2(B)]]|0> 
        tmp = np.einsum("klcd,lkdb->cb",self.x2_B,self.y2_C)
        tmp = np.einsum("jb,cb->jc",self.x1_A,tmp)
        self.Bcon3 -= np.einsum("jc,jc->",tmp,self.Hov)

        tmp = np.einsum("klcd,ljdc->kj",self.x2_B,self.y2_C)
        tmp = np.einsum("kj,jb->kb",tmp,self.x1_A)
        self.Bcon3 -= np.einsum("kb,kb->",tmp,self.Hov)

        tmp = np.einsum('lkda,klcd->ac',self.y2_C,self.x2_B)
        tmp2 = np.einsum('jb,ajcb->ac',self.x1_A,self.Hvovv)
        self.Bcon3 += 2.*np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('lkda,klcd->ac',self.y2_C,self.x2_B)
        tmp2 = np.einsum('jb,ajbc->ac',self.x1_A,self.Hvovv)
        self.Bcon3 -= np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('jb,ljda->lbda',self.x1_A,self.y2_C)
        tmp2 = 2.*np.einsum('klcd,akbc->ldab',self.x2_B,self.Hvovv)
        tmp2 -= np.einsum('klcd,akcb->ldab',self.x2_B,self.Hvovv)
        self.Bcon3 += np.einsum('lbda,ldab->',tmp,tmp2)

        tmp = np.einsum('ia,fkba->fkbi',self.x1_A,self.Hvovv)
        tmp = np.einsum('fkbi,jifc->kjbc',tmp,self.y2_C)
        self.Bcon3 -= np.einsum('jkbc,kjbc->',self.x2_B,tmp)

        tmp = np.einsum('ia,fjac->fjic',self.x1_A,self.Hvovv)
        tmp = np.einsum('fjic,ikfb->jkbc',tmp,self.y2_C)
        self.Bcon3 -= np.einsum('jkbc,jkbc->',self.x2_B,tmp)

        tmp = np.einsum('ia,jkfa->jkfi',self.x1_A,self.y2_C)
        tmp2 = np.einsum('jkbc,fibc->jkfi',self.x2_B,self.Hvovv)
        self.Bcon3 -= np.einsum('jkfi,jkfi->',tmp2,tmp)

        tmp = np.einsum('jb,kjib->ki',self.x1_A,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_B,self.y2_C)
        self.Bcon3 -= 2.*np.einsum('ki,ki->',tmp,tmp2)       

        tmp = np.einsum('jb,jkib->ki',self.x1_A,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_B,self.y2_C)
        self.Bcon3 += np.einsum('ki,ki->',tmp,tmp2)         

        tmp  = 2.*np.einsum('jkic,klcd->jild',self.Hooov,self.x2_B)
        tmp -= np.einsum('kjic,klcd->jild',self.Hooov,self.x2_B)
        tmp  = np.einsum('jild,jb->bild',tmp,self.x1_A)
        self.Bcon3 -= np.einsum('bild,ilbd->',tmp,self.y2_C) 

        tmp  = np.einsum('ia,jkna->jkni',self.x1_A,self.Hooov)
        tmp2  = np.einsum('jkbc,nibc->jkni',self.x2_B,self.y2_C)
        self.Bcon3 += np.einsum('jkni,jkni->',tmp2,tmp)

        tmp  = np.einsum('ia,nkab->nkib',self.x1_A,self.y2_C)
        tmp  = np.einsum('jkbc,nkib->jnic',self.x2_B,tmp)
        self.Bcon3 += np.einsum('jnic,ijnc->',tmp,self.Hooov)

        tmp  = np.einsum('ia,nkba->nkbi',self.x1_A,self.y2_C)
        tmp  = np.einsum('jkbc,nkbi->jnci',self.x2_B,tmp)
        self.Bcon3 += np.einsum('jnci,jinc->',tmp,self.Hooov)

        # <0|L1(C)[[Hbar(0),X2(A)],X1(B)]]|0> 
        tmp = 2.*np.einsum("jkbc,kc->jb",self.x2_A,self.y1_C)
        tmp -= np.einsum("jkcb,kc->jb",self.x2_A,self.y1_C)
        tmp = np.einsum('ijab,jb->ia',self.Loovv,tmp)
        self.Bcon3 += np.einsum("ia,ia->",tmp,self.x1_B)

        tmp = np.einsum("jkbc,jkba->ca",self.x2_A,self.Loovv)
        tmp = np.einsum("ia,ca->ic",self.x1_B,tmp)
        self.Bcon3 -= np.einsum("ic,ic->",tmp,self.y1_C)

        tmp = np.einsum("jkbc,jibc->ki",self.x2_A,self.Loovv)
        tmp = np.einsum("ki,ia->ka",tmp,self.x1_B)
        self.Bcon3 -= np.einsum("ka,ka->",tmp,self.y1_C)

        # <0|L1(C)[[Hbar(0),X2(A)],X1(B)]]|0> 
        tmp = np.einsum("klcd,lkdb->cb",self.x2_A,self.y2_C)
        tmp = np.einsum("jb,cb->jc",self.x1_B,tmp)
        self.Bcon3 -= np.einsum("jc,jc->",tmp,self.Hov)

        tmp = np.einsum("klcd,ljdc->kj",self.x2_A,self.y2_C)
        tmp = np.einsum("kj,jb->kb",tmp,self.x1_B)
        self.Bcon3 -= np.einsum("kb,kb->",tmp,self.Hov)

        tmp = np.einsum('lkda,klcd->ac',self.y2_C,self.x2_A)
        tmp2 = np.einsum('jb,ajcb->ac',self.x1_B,self.Hvovv)
        self.Bcon3 += 2.*np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('lkda,klcd->ac',self.y2_C,self.x2_A)
        tmp2 = np.einsum('jb,ajbc->ac',self.x1_B,self.Hvovv)
        self.Bcon3 -= np.einsum('ac,ac->',tmp,tmp2)

        tmp = np.einsum('jb,ljda->lbda',self.x1_B,self.y2_C)
        tmp2 = 2.*np.einsum('klcd,akbc->ldab',self.x2_A,self.Hvovv)
        tmp2 -= np.einsum('klcd,akcb->ldab',self.x2_A,self.Hvovv)
        self.Bcon3 += np.einsum('lbda,ldab->',tmp,tmp2)

        tmp = np.einsum('ia,fkba->fkbi',self.x1_B,self.Hvovv)
        tmp = np.einsum('fkbi,jifc->kjbc',tmp,self.y2_C)
        self.Bcon3 -= np.einsum('jkbc,kjbc->',self.x2_A,tmp)

        tmp = np.einsum('ia,fjac->fjic',self.x1_B,self.Hvovv)
        tmp = np.einsum('fjic,ikfb->jkbc',tmp,self.y2_C)
        self.Bcon3 -= np.einsum('jkbc,jkbc->',self.x2_A,tmp)

        tmp = np.einsum('ia,jkfa->jkfi',self.x1_B,self.y2_C)
        tmp2 = np.einsum('jkbc,fibc->jkfi',self.x2_A,self.Hvovv)
        self.Bcon3 -= np.einsum('jkfi,jkfi->',tmp2,tmp)

        tmp = np.einsum('jb,kjib->ki',self.x1_B,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_A,self.y2_C)
        self.Bcon3 -= 2.*np.einsum('ki,ki->',tmp,tmp2)     

        tmp = np.einsum('jb,jkib->ki',self.x1_B,self.Hooov)
        tmp2 = np.einsum('klcd,ilcd->ki',self.x2_A,self.y2_C)
        self.Bcon3 += np.einsum('ki,ki->',tmp,tmp2)          

        tmp  = 2.*np.einsum('jkic,klcd->jild',self.Hooov,self.x2_A)
        tmp -= np.einsum('kjic,klcd->jild',self.Hooov,self.x2_A)
        tmp  = np.einsum('jild,jb->bild',tmp,self.x1_B)
        self.Bcon3 -= np.einsum('bild,ilbd->',tmp,self.y2_C)  

        tmp  = np.einsum('ia,jkna->jkni',self.x1_B,self.Hooov)
        tmp2  = np.einsum('jkbc,nibc->jkni',self.x2_A,self.y2_C)
        self.Bcon3 += np.einsum('jkni,jkni->',tmp2,tmp)

        tmp  = np.einsum('ia,nkab->nkib',self.x1_B,self.y2_C)
        tmp  = np.einsum('jkbc,nkib->jnic',self.x2_A,tmp)
        self.Bcon3 += np.einsum('jnic,ijnc->',tmp,self.Hooov)

        tmp  = np.einsum('ia,nkba->nkbi',self.x1_B,self.y2_C)
        tmp  = np.einsum('jkbc,nkbi->jnci',self.x2_A,tmp)
        self.Bcon3 += np.einsum('jnci,jinc->',tmp,self.Hooov)
      
        self.hyper += self.Bcon1 + self.Bcon2 + self.Bcon3
  
        return self.hyper

# End HelperCCQuadraticResp class
