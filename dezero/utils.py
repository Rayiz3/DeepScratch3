import os
import subprocess

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, tyle=filled]\n'
    
    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
        
    txt = dot_var.format(id(v), name)
    
    return txt

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, tyle=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)
    
    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))  # y is weakref
    
    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()
        
    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)
                
    add_func(output.creator)
    txt += _dot_var(output, verbose) # start node
    
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func) # creator node
        
        for x in func.inputs:
            txt += _dot_var(x, verbose) # creator's input node

            if x.creator is not None:
                add_func(x.creator)
    
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dot_graph = get_dot_graph(output, verbose)
    
    # save dot data
    tmp_dir = os.path.join(os.path.expanduser('~'), '.dezero')  # expanduser('~') : replace home directory '~' into absolute directory
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    
    with open(graph_path, 'w') as f:
        f.write(dot_graph)
    
    # call dot instruction
    extension = os.path.splitext(to_file)[1][1:]  # png, pdf, ...
    cmd = 'dot {}  -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)