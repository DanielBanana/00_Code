import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import os
import argparse


def create_modelDescription(targetDirPath, n_inputs, n_outputs):

    file_name = "modelDescription.xml"
    data_path = os.path.join(targetDirPath, "data")
    file_path = os.path.join(data_path, file_name)
    file_path_tmp = os.path.join(data_path, 'tmp.xml')

    with open(file_path, 'rb') as xml_file:

        tree = ET.parse(xml_file)
        root = tree.getroot()

        guid = root.attrib['guid']

        # Function to add a ScalarVariable with specified attributes
        def add_scalar_variable(parent, name, description, value_reference, variability, causality, initial=None):

            attribs = {
                'name': name,
                'description': description,
                'valueReference': str(value_reference),
                'variability': variability,
                'causality': causality,
            }
            if initial :
                attribs['initial'] = initial

            scalar_variable = ET.SubElement(parent, 'ScalarVariable', attribs)

            if causality == 'input':
                real = ET.SubElement(scalar_variable, 'Real', {'start': "1.0"})
            if causality == 'output' and (initial == 'exact' or initial == 'approx'):
                real = ET.SubElement(scalar_variable, 'Real', {'start': "0.0"})

            return scalar_variable

        n_variables = n_inputs + n_outputs

        # Add ScalarVariables with causality "input"
        input_variables = [
            (f"FMI_INPUT_{i}",f"Input Description {i}", i, "continuous", "input") for i in range(1, n_inputs+1)
        ]

        # Add ScalarVariables with causality "output"
        output_variables = [
            (f"FMI_OUTPUT_{j-n_inputs}", f"Output Description {j-n_inputs}", j, "continuous", "output", "exact") for j in range(n_inputs+1, n_variables+1)
        ]

        # Find the ModelVariables element
        model_variables = root.find('ModelVariables')

        # Find the ModelStructure element
        model_structure = root.find('ModelStructure')

        # Add ScalarVariables to ModelVariables
        for name, description, value_reference, variability, causality in input_variables:
            add_scalar_variable(model_variables, name, description, value_reference, variability, causality)

        for name, description, value_reference, variability, causality, initial in output_variables:
            add_scalar_variable(model_variables, name, description, value_reference, variability, causality, initial)

        # Add dependencies to ModelStructure for all outputs on all inputs
        if model_structure is not None:
            outputs = model_structure.find('Outputs')
            if outputs is None:
                outputs = ET.SubElement(model_structure, 'Outputs')

            for output_variable in output_variables:
                index = output_variable[2]
                dependencies = ' '.join(str(variable[2]) for variable in input_variables)
                unknown = ET.SubElement(outputs, 'Unknown', attrib={'index': str(index), 'dependencies': dependencies})

        # Save the modified XML to a new file
        tree.write(file_path_tmp, encoding='utf-8', xml_declaration=True, method='xml', short_empty_elements=False)

    os.remove(file_path)
    os.rename(file_path_tmp, file_path)

    print(f"Modified XML file saved as '{file_name}'")

    return guid









