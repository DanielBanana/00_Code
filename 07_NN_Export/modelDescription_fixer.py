import xml.etree.ElementTree as ET
import argparse
import os

if __name__ == "__main__":

    filename = "modelDescription.xml"
    path = os.path.abspath(__file__)
    filename = '/'.join(path.split('/')[:-1]) + '/' + filename

    tree = ET.parse(filename)
    root = tree.getroot()

    pass

    for child in root:
        if child.tag == "ModelVariables":
            for variable in child:
                if variable.get("causality") == "input" and variable.get("initial"):
                    del(variable.attrib["initial"])

    tree.write(filename)