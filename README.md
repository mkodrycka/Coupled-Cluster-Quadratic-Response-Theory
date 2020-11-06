# Response theory

## Coupled-Cluster Linear Response (CCLR) and Quadratic Response (CCQR)

The implementation contains:
- `polar.py`: Script for calculating CCSD dipole polarizabilities using Coupled Cluster Linear Response (CCLR) theory.
- `hyper.py`: Script for calculating CCSD hyperpolarizability using Coupled Cluster Quadratic Response (CCQR) theory.
- `optrot.py`: Computing specific optical rotation using Coupled Cluster Linear Response
- `helper_ccpert.py`: Helper classes and functions for CCLR and CCQR implementations.
- `helper_ccenergy.py`: Helper classes and functions for computing CCSD energy.
- `helper_cclambda.py`: Helper classes and functions for computing CC Lambda equations.
- `helper_cceom.py`: Helper classes and functions for computing CCEOM.
- `helper_cchbar.py`: Helper classes and functions for computing HBAR intermediates.


### Getting Started

1. Obtain required software
    1. [Psi4NumPy](https://github.com/psi4/psi4numpy) (clone this repository; no install available)
    2. [Psi4](http://psicode.org/psi4manual/1.1/build_obtaining.html)
        * Option 1 (easiest): [Download installer](http://vergil.chemistry.gatech.edu/psicode-download/1.1.html) and install according to [instructions](http://psicode.org/psi4manual/1.1/conda.html#how-to-install-a-psi4-binary-with-the-psi4conda-installer-command-line).
          ```
          # Have Psi4conda installer (http://psicode.org/downloads.html)
          >>> bash psi4conda-{various}.sh
          # Check `psi4` command in path; adjust path if needed
          # **IF** using DFT tutorials (or a few newer specialized integrals), after above, create a separate environment within for newer psi4:
          >>> conda create -n p4env psi4 -c psi4/label/dev
          >>> source activate p4env
          ```
        * Option 2 (easy): Download Conda package according to [instructions](http://psicode.org/psi4manual/1.1/conda.html#how-to-install-a-psi4-binary-into-an-ana-miniconda-distribution)
          ```
          # Have Anaconda or Miniconda (https://conda.io/miniconda.html)
          >>> conda create -n p4env psi4 -c psi4
          >>> bash
          >>> source activate p4env
          ```
        * Option 3 (medium): [Clone source](https://github.com/psi4/psi4) and [compile](https://github.com/psi4/psi4/blob/master/CMakeLists.txt#L16-L143) according to [instructions](http://psicode.org/psi4manual/master/build_faq.html#configuring-building-and-installing-psifour-via-source)
          ```
          # Get Psi4 source
          >>> git clone https://github.com/psi4/psi4.git
          >>> git checkout v1.1
          >>> cmake -H. -Bobjdir -Doption=value ...
          >>> cd objdir && make -j`getconf _NPROCESSORS_ONLN`
          # Find `psi4` command at objdir/stage/<TAB>/<TAB>/.../bin/psi4; adjust path if needed
          ```
    3. [Python](https://python.org) 3.6+ (incl. w/ Psi4 Options 1 & 2)
    4. [NumPy](http://www.numpy.org) 1.7.2+ (incl. w/ Psi4 Options 1 & 2)
    5. [Scipy](https://scipy.org) 0.13.0+
2. Enable Psi4 & PsiAPI (if Psi4 was built from source)
   1. Find appropriate paths
        ```
        >>> psi4 --psiapi-path
        export PATH=/path/to/dir/of/python/interpreter/against/which/psi4/compiled:$PATH
        export PYTHONPATH=/path/to/dir/of/psi4/core-dot-so:$PYTHONPATH
        ```
    2. Export relevant paths
        ```
        >>> bash
        >>> export PATH=/path/to/dir/of/python/interpreter/against/which/psi4/compiled:$PATH
        >>> export PYTHONPATH=/path/to/dir/of/psi4/core-dot-so:$PYTHONPATH
        ```
3. Run scripts as conventional Python scripts
    * Example: Run `hyper.py`
        ```
        >>> python hyper.py
        ```

A tutorial that covers the basics of NumPy can be found
[here](http://wiki.scipy.org/Tentative_NumPy_Tutorial).
