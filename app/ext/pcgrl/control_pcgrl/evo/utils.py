import copy
import warnings

import numpy as np
import torch

def get_one_hot_map(int_map, n_tile_types, continuous=False):
    if continuous:
        return int_map  # AD HOC continuous af
    obs = (np.arange(n_tile_types) == int_map[..., None]).astype(int)
    env_is_3D = len(int_map.shape) == 3
    if not env_is_3D:
        obs = obs.transpose(2, 0, 1)
    else:
        obs = obs.transpose(3, 0, 1, 2)

    return obs


def draw_net(config: object, genome: object, view: object = False, filename: object = None, node_names: object = None, show_disabled: object = True,
             prune_unused: object = False,
             node_colors: object = None, fmt: object = 'svg') -> object:
    """ Receives a genome and draws a neural network with arbitrary topology. """
    # Attributes for network nodes.
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    if node_colors is None:
        node_colors = {}

    assert type(node_colors) is dict

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2',
        'length': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = set()
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                connections.add(cg.key)

        used_nodes = copy.copy(outputs)
        pending = copy.copy(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            #if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot

def set_fixed_type(x, fixed_tiles):
    """Takes a representation of a map and changes the map with changed fixed tiles
    Args:
        x (4d Tensor): (batch_size,num_channels,width,height)
        fixed_tiles (2d numpy array): Fixed tiles [[TYPE, X, Y]]
    """
    for coord in fixed_tiles:
        channel,xc,yc = coord
        batch, channels = x.shape[0], x.shape[1]
        for i in range(batch):
            for j in range(channels):
                if channel == j:
                    x[i,j, xc, yc] = 1.0
                else:
                    x[i,j, xc, yc] = 0.0
                    
def set_fixed_one(x, fixed_tiles):
    """Takes N amount of channels and changes each channel so the x,y location of the channel is 1.0 where we have should have a fixed tile

    Should provide information to the model, that at that location, something is happening

    Args:
        x (4d Tensor): (batch_size,num_channels,width,height)
        fixed_tiles (2d numpy array): Fixed tiles [[TYPE, X, Y]]
    """
    for coord in fixed_tiles:
        tile_type, xc, yc = coord
        batch, channels = x.shape[0], x.shape[1]
        for i in range(batch):
            for j in range(channels):
                x[i, j, xc, yc] = 1.0

def generate_binary_channel(fixed_tiles):
    """Generate binary channel based on the fixed tiles

    Generate binary channel based on fixed tiles, where (x,y) = 1 where fixed_tile exists

    Args:
        fixed_tiles (2d numpy array): Fixed tiles [[TYPE, X, Y]]

    Returns:
        2d numpy array: binary channel
    """
    binary_channel = np.zeros((16,16))

    for fixed_tile in fixed_tiles:
        tile_type, x, y = fixed_tile
        binary_channel[x,y] = 1.0

    return binary_channel

def add_binary_channel(x, binary_channel):
    """Add a binary channel to the existing input x

    Args:
        x (4d Tensor): (batch_size,num_tile_types,width,height)
        fixed_tiles (2d numpy array): Fixed tiles [[TYPE, X, Y]]

    Returns:
        tensor (4d Tensor): (batch_size, channels, width, height)
    """
    batch, channels = x.shape[0], x.shape[1]
    binary_channel = binary_channel.reshape((1, 16, 16))
    tensor = torch.empty(batch, channels+1, 16, 16) # + 1 for the binary channel

    for i in range(batch):
        # binary channel = (1, 16, 16), x[i] = (8, 16, 16), result = (9, 16, 16)
        result = torch.from_numpy(np.concatenate((binary_channel, x[i]), axis=0))
        tensor[i] = result
    return tensor 