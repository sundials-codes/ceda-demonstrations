# PIROCK Codes

[Note: this project is in active development.]

This package contains the example tests demonstrated in "PIROCK: A swiss-knife partitioned implicit-explicit orthogonal Runge-Kutta Chebyshev integrator for stiff diffusion-advection-reaction problems with or without noise" by Assyr Abdulle
and Gilles Vilmart. It also includes a one-dimensional advection-diffusion-reaction example with three species, which is the main example used in this work.

## Installation

The following steps describe how to run the examples in the PIROCK package.

### Gettting the Code

To obtain the code, clone this repository with Git:

```bash
  git clone https://github.com/sundials-codes/ceda-demonstrations.git
```

The PIROCK package can be found in the adr_1D folder.


### Usage

#### Python packages

Since Pandas is used, it is likely that it is already installed on your system.  However, if it is missing or needs to be updated, then we recommend installing it within a Python virtual environment using the following steps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r python_requirements.txt
```

You may "deactivate" this Python environment from your current shell with the command

```bash
deactivate
```

and in the future you can "reactivate" the python environment in your shell by running from the top-level directory of this repository

```bash
source .venv/bin/activate
```


#### Makefile
Run the makefile to build the executables for the example tests.

The Python script runtests_adr_1D.py runs the one-dimensional advection-diffusion-reaction example with three species. It allows testing various configurations by adjusting advection, diffusion, and reaction coefficients, as well as the spatial dimension. The script also includes options to enable or disable advection and reaction terms, and to choose between adaptive or fixed time stepping.
