from jinja2 import Template
from XML_Parser import *
import os 
def generate_template():
    gen_model = {}
    gen_model = {
        'signals':signals_list(),
        'groupsignals':GroupSignals_list(),
        'groupIPDUs':IPDUs_list(),
        'signalgroups':SignalGroups_list()
    }
    return gen_model

def generate_output(generation_model,tempaltes_path,output_path):
    if generation_model is not None:
        for filename in os.scandir(tempaltes_path):
            if filename.is_file() and ".j2" in filename.path:
                out_file = os.path.join(output_path, filename.path.split('\\')[-1][:-3])
                with open(filename.path, "r") as tempalte_file:
                    template = tempalte_file.read()
                    j2_template = Template(template)
                    gen_model = {"gen_model": generation_model}
                    file_content = j2_template.render(gen_model)
                    with open(os.path.join(out_file), "w") as file:
                        file.write(file_content)

generation_model = generate_template()
generate_output(generation_model,'Templates','Gen')

