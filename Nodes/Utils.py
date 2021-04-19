from graphviz import Digraph
from PIL import Image
from io import BytesIO

def prefillPipeline(featureExtractor, prefillMs, sampleRate, prefillValue): # TODO this sucks
    frameCount = int(prefillMs / (1.0 / (sampleRate / 1000.0)))
    for i in range(0, frameCount - 1):
        featureExtractor.addData(prefillValue)
        
        
def dot_add_node(dot, node):
    shape = 'rect'
    if node.has_inputs == False:
        shape = 'invtrapezium'
    if node.has_outputs == False:
        shape = 'trapezium'
    dot.node(str(node), node.name, shape = shape, style = 'rounded')
    
def make_dot_graph(graph_inputs, scale = 0.5, svg=False):
    processing_list = graph_inputs
    if not isinstance(processing_list, list):
        processing_list = [processing_list]
    
    if svg == False:
        dot = Digraph(format = 'png', strict = True)
    else:
        dot = Digraph(format = 'svg', strict = True)
    while len(processing_list) != 0:
        node = processing_list.pop()
        dot_add_node(dot, node)
        for node_output in node.output_classes:
            dot_add_node(dot, node_output)
            dot.edge(str(node), str(node_output))
            processing_list.append(node_output)
    if svg == False:
        img = Image.open(BytesIO(dot.pipe()))
        return img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    else:
        return dot.pipe().decode('utf-8')
