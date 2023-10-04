import platform
import subprocess
import os

def build_fmu(targetDirPath, modelName):

    # def testBuildFMU(self):
    # """Runs a cmake-based compilation of the generated FMU to check if the code compiles.
    # """

    # generate path to /build subdir
    buildDir = os.path.join(targetDirPath, "build")
    binDir = os.path.join(targetDirPath, "bin/release")

    print("We are now building the FMU.")
    try:

        # Different script handling based on platform
        if platform.system() == "Windows":

            # call batch file to build the FMI library
            pipe = subprocess.Popen(["build_VC_x64.bat"], shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd = buildDir, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            # retrieve output and error messages
            outputMsg, errorMsg = pipe.communicate()
            # get return code
            rc = pipe.returncode

            # if return code is different from 0, print the error message
            if rc != 0:
                print(str(outputMsg) + "\n" + str(errorMsg))
                raise RuntimeError("Error during compilation of FMU.")

            print("Compiled FMU successfully")

            # call batch file to build the FMI library
            pipe = subprocess.Popen(["deploy.bat"], shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd = buildDir, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            # retrieve output and error messages
            outputMsg, errorMsg = pipe.communicate()
            # get return code
            rc = pipe.returncode

            if rc != 0:
                print(str(outputMsg) + "\n" + str(errorMsg))
                raise RuntimeError("Error during compilation of FMU")

            print("Successfully created {}".format(modelName + ".fmu")	)

        else:
            # shell file execution for Mac & Linux
            pipe = subprocess.Popen(["bash", './build.sh'], cwd = buildDir, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            outputMsg,errorMsg = pipe.communicate()
            rc = pipe.returncode

            if rc != 0:
                print(errorMsg)
                raise RuntimeError("Error during compilation of FMU")

            print("Compiled FMU successfully")

            # Deployment

            # shell file execution for Mac & Linux
            deploy = subprocess.Popen(["bash", './deploy.sh'], cwd = buildDir, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            outputMsg,errorMsg = deploy.communicate()
            dc = deploy.returncode

            if dc != 0:
                print(errorMsg)
                raise RuntimeError("Error during assembly of FMU")

            print("Successfully created {}".format(modelName + ".fmu")	)

    except Exception as e:
        print(str(e))
        print("Error building FMU.")
        raise