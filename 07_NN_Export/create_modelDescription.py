import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
import os

filename = "modelDescription.xml"
path = os.path.abspath(__file__)
filename = '/'.join(path.split('/')[:-1]) + '/' + filename

tree = ET.parse(filename)
root = tree.getroot()

# Function to add a ScalarVariable with specified attributes
def add_scalar_variable(parent, name, description, value_reference, variability, causality, initial=None):
    scalar_variable = ET.SubElement(parent, 'ScalarVariable', {
        'name': name,
        'description': description,
        'valueReference': str(value_reference),
        'variability': variability,
        'causality': causality,
        'initial': initial if initial else ""
    })

    if causality == 'input' and initial:
        real = ET.SubElement(scalar_variable, 'Real', {'start': initial})

    return scalar_variable

# Add ScalarVariables with causality "input"
input_variables = [
    ("input_variable1", "Input Description 1", 10, "continuous", "input", "calculated"),
    ("input_variable2", "Input Description 2", 11, "continuous", "input", "exact"),
]

# Add ScalarVariables with causality "output"
output_variables = [
    ("output_variable1", "Output Description 1", 20, "continuous", "output", "approx"),
    ("output_variable2", "Output Description 2", 21, "continuous", "output", "exact"),
]

# Find the ModelVariables element
model_variables = root.find('.//ModelVariables')

# Find the ModelStructure element
model_structure = root.find('.//ModelStructure')

# Add ScalarVariables to ModelVariables
for name, description, value_reference, variability, causality, initial in input_variables:
    add_scalar_variable(model_variables, name, description, value_reference, variability, causality, initial)

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
tree.write('modified_modelDescription.xml', encoding='utf-8', xml_declaration=True, method='xml', short_empty_elements=False)

print("Modified XML file saved as 'modified_modelDescription.xml'")









