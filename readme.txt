This folder contains files for the implementation of the hybrid modeling of the 
Van der Pol oscillator.
The files work if the 00_Code folder is the working directory (check os.getcwd, open VSCode with this folder 
as top folder).

01_ML_informed_Simulators
Contains files for the direct training and adjoint method approach.

02_FMPy
Contains files for the use with FMUs. All scripts load an Model Exchange FMU of the VdP Oscillator and solve it via 
Eulers method.
Also implements hybrid modelling approaches in combination with the FMUs

03_Amesim
Contains the implementation of hybrid modeling with FMU by a previous worker at Siemens. Given to me by
Dirk. Currently does not work, but shouldn't be too hard to fix.
