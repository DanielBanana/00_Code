import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--targetDirName', type=str, default='NeuralNetwork', help='Name of the folder in which the FMU gets generated. Usually identical to modelName')
    parser.add_argument('--modelName', type=str, default='NeuralNetwork', help='Name of the FMU in which gets generated. Usually identical to targetDirPath')
    args = parser.parse_args()

    targetDirName = args.targetDirName
    modelName = args.modelName

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    targetDirPath = os.path.join(directory,targetDirName)

    cpp_path = os.path.join(targetDirPath,"src",modelName+".cpp")
    h_path = os.path.join(targetDirPath,"src",modelName+".h")
    h_path_tmp = os.path.join(targetDirPath,"src",modelName+"_tmp.h")

    h_string = """
	template <size_t rows, size_t cols>
	void matvecmul(double (&matrix)[rows][cols], double (&vec)[cols], double (&res)[rows]);

	template <size_t entries>
	void vecadd(double (&vec)[entries], double (&res)[entries]);

	template <size_t entries>
	void relu(double (&vec)[entries]);

	template <size_t rows, size_t cols>
	void relu_layer(double (&weights)[rows][cols], double (&bias)[rows], double (&input)[cols], double (&res)[rows], bool use_relu);"""

    # We want the h Code to be pasted after
    #  /*! Initializes model */
    # void init();
    # So we need to find that string inside the .h-file
    search_string_h = "void init();"

    with open(h_path, 'r') as h_file, open(h_path_tmp, 'w') as h_file_tmp:
        lines = h_file.readlines()
        for i, line in enumerate(lines):
            if line.find(search_string_h) != -1:
                print(f"Found search string for .h-file: {search_string_h}")
                print(f"Line number: {lines.index(line)}")
                h_file_tmp.write(h_string)
            h_file_tmp.write(line)
    os.remove(h_path)
    os.rename(h_path_tmp, h_path)



