This folder contains files for the implementation of the hybrid modeling of the 
Van der Pol oscillator.
The files work if the 00_Code folder is the working directory (check os.getcwd, open VSCode with this folder 
as top folder).

Before running the code install the needed python libraries via the requirements.txt file:
Go to the 00_Code directory and create a new python virtual environment with: $python3 -m venv (environment_name; e.g. 00_siemens_env)
Activate the environment with: $source environment_name/bin/activate
Install the libraries with: $pip install -r requirements.txt

01_ML_informed_Simulators
Contains files for the direct training and adjoint method approach.

02_FMPy
Contains files for the use with FMUs. All scripts load an Model Exchange FMU of the VdP Oscillator and solve it via 
Eulers method.
Also implements hybrid modelling approaches in combination with the FMUs

03_Amesim
Contains the implementation of hybrid modeling with FMU by a previous worker at Siemens. Given to me by
Dirk. Currently does not work, but shouldn't be too hard to fix.

04_PSO
Particle Swarm Optimization; currently contains one example to have a reference on how it works.
