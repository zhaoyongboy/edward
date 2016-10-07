import tensorflow as tf
import tensorflow as tf
distributions = tf.contrib.distributions
sg = tf.contrib.bayesflow.stochastic_graph
import numpy as np

import edward as ed
import edward.util as util
import edward.models.random_variables as rvs

def print_tree(op, depth=0, stop_nodes=None, stop_types=None):
    if stop_nodes is None: stop_nodes = set()
    if stop_types is None: stop_types = set()
    print ''.join(['-'] * depth), '%s...%s' % (op.type, op.name)
    if (op not in stop_nodes and op.type not in stop_types):
        for i in op.inputs:
            print_tree(i.op, depth+1, stop_nodes=stop_nodes, stop_types=stop_types)

def is_child(parent, possible_child, stop_nodes=None, stop_types=None):
    '''
    Determines if possible_child is a child of parent in the
    _reversed_ computation graph---that is, does possible_child
    get computed before parent.
    
    Args:
      parent: The node we're testing.
      possible_child: The child we're interested in.
      stop_nodes: A set of nodes that should stop the graph walk.
        This is useful for making sure we don't go past a random
        variable's Markov blanket.
      stop_types: A set of node types that should stop the graph walk.
        For example, "Shape" nodes.
    
    Returns:
      A list of input nodes of parent that depend on possible_child.
    '''
    if stop_nodes is None:
        stop_nodes = set()
    if stop_types is None:
        stop_types = set()
    result = []
    if (not parent) or (parent in stop_nodes) or (parent.type in stop_types):
        return []
    for c in parent.inputs:
        c = c.op
        if (c == possible_child):
            result += [c]
        else:
            result += is_child(c, possible_child, stop_nodes, stop_types)
    return result

def de_identity(node):
    '''
    Gets rid of Identity nodes.
    TODO: Relying on this might screw up device placement.
    '''
    while (node.type == 'Identity'):
        node = list(node.inputs)[0].op
    return node

def rv_value(node):
    '''
    Returns the "canonical" sample from a random variable
    node. Seems preferable to having a bunch of extraneous
    Identity nodes lying around?
    '''
    return de_identity(tf.identity(node).op).outputs[0]

# TODO: Deal with case where sufficient statistic involves multiplication: t(x) = f(x)*g(x).
# TODO: Deal with constrained support (mostly for beta/Dirichlet).
# TODO: Deal with multivariate normals.
# TODO: Deal with squared terms in Gaussians.
# TODO: Repeat replacements until convergence.
#       E.g., Log(Inv(Mul(a, b))) = -Log(Mul(a, b)) = -Log(a) + -Log(b)

def opify(node):
    if tf.is_numeric_tensor(node):
        return node.op
    else:
        return node


# def div_to_inv(node, inputs):
#     print 'replacing %s' % node.name
#     with tf.name_scope(node.name) as scope:
# #     with tf.name_scope(node.name[:node.name.rfind('/')]):
#         inv = tf.inv(inputs[1])
# #     return tf.mul(inputs[0], inv, name=node.name)
#         return tf.mul(inputs[0], inv, name=scope)

# def log_mul_to_add_log(node, root):
#     assert(is_log_mul(node))
#     new_g = tf.Graph()
#     old_inputs = node.inputs[0].op.inputs
#     new_input0 = tf.contrib.copy_graph.copy_op_to_graph(old_inputs[0].op, new_g, []).outputs[0]
#     new_input1 = tf.contrib.copy_graph.copy_op_to_graph(old_inputs[1].op, new_g, []).outputs[0]
#     with new_g.as_default():
#         with tf.name_scope(node.name) as scope:
#             new_node = tf.add(tf.log(new_input0), tf.log(new_input1),
#                               name=scope)
#     new_root = tf.contrib.copy_graph.copy_op_to_graph(root, new_g, [])
#     return new_g, new_root, new_node

# def div_to_inv(node):
#     assert(node.type == 'Div')
#     new_g = tf.Graph()
#     old_inputs = node.inputs
#     new_inputs = [tf.contrib.copy_graph.copy_op_to_graph(item.op, new_g, []).outputs[0]
#                   for item in old_inputs]
#     with new_g.as_default():
#         with tf.name_scope(node.name) as scope:
#             new_node = tf.mul(new_inputs[0], tf.inv(new_inputs[1]), name=scope)
#     new_root = tf.contrib.copy_graph.copy_op_to_graph(root, new_g, [])
#     return new_g, new_root, new_node

def copy_inputs(old_inputs, new_g):
    new_inputs = []
    for input in old_inputs:
        new_inputs.append(tf.contrib.copy_graph.copy_op_to_graph(input.op, new_g, []).outputs[0])
    return new_inputs

def replace_node(node, root, replace_f):
    new_g = tf.Graph()
    with new_g.as_default():
        with tf.name_scope(node.name) as scope:
            new_node = replace_f(node, new_g, scope)
    new_root = tf.contrib.copy_graph.copy_op_to_graph(root, new_g, [])
    return new_g, new_root, new_node

def make_is_type(t):
    return lambda node: node.type == t
           
def div_to_inv(node, new_g, scope):
    new_inputs = copy_inputs(node.inputs, new_g)
    return tf.mul(new_inputs[0], tf.inv(new_inputs[1]), name=scope)

def is_log_mul(node):
    return (node.type == 'Log') and (de_identity(node.inputs[0].op).type == 'Mul')

def log_mul_to_add_log(node, new_g, scope):
    new_inputs = copy_inputs(de_identity(node.inputs[0].op).inputs, new_g)
    return tf.add(tf.log(new_inputs[0]), tf.log(new_inputs[1]), name=scope)

def is_log_inv(node):
    return (node.type == 'Log') and (de_identity(node.inputs[0].op).type == 'Inv')

def log_inv_to_neg_log(node, new_g, scope):
    new_inputs = copy_inputs(de_identity(node.inputs[0].op).inputs, new_g)
    return tf.neg(tf.log(new_inputs[0]), name=scope)

def square_to_mul(node, new_g, scope):
    new_inputs = copy_inputs(node.inputs, new_g)
    return tf.mul(new_inputs[0], new_inputs[0], name=scope)

# def is_square_mul(node, new_g, scope):
#     return (node.type == 'Square') and (de_identity(node.inputs[0].op).type == 'Mul')

# def square_mul_to_mul_square(node, new_g, scope):
#     new_inputs = copy_inputs(de_identity(node.inputs[0].op).inputs, new_g)
#     return tf.mul(tf.square(new_inputs[0]), tf.square(new_inputs[1]), name=scope)

# def replace_node(node, root, replace_f):
#     new_g = tf.Graph()
#     inputs = [tf.contrib.copy_graph.copy_op_to_graph(i, new_g, [])
#               for i in node.inputs]
#     with new_g.as_default():
#         new_node = replace_f(node, inputs)

#         done_set = set()
#         consumers = [i for i in node.outputs[0].consumers()]
#         for i in consumers:
#             if i not in done_set:
#                 new_inputs = []
#                 for j in i.inputs:
#                     if j.op.name == node.name:
#                         new_inputs.append(new_node)
#                     else:
#                         j = tf.contrib.copy_graph.copy_op_to_graph(j.op, new_g, [])
#                         new_inputs.append(j.outputs[0])
#                 new_g.create_op(i.type, new_inputs,
#                                 [j.dtype for j in i.outputs], name=i.name)
#                 done_set.add(i)
#     new_root = tf.contrib.copy_graph.copy_op_to_graph(root, new_g, [])
#     return new_g, new_root, new_node


def _replace_until_done_rec(node, root, node_tester, replace_f, stop_nodes=None, incl_node=None):
#     print 'visiting %s-%s' % (node.type, node.name), node_tester(node)
    if stop_nodes is None:
        stop_nodes = set()
    if node in stop_nodes:
        return None
#     print len(is_child(node, incl_node, stop_nodes)), incl_node.name
    if not ((incl_node is None) or len(is_child(node, incl_node, stop_nodes)) > 0):
        return None
    if node_tester(node):
        return replace_node(node, root, replace_f)
    for c in node.inputs:
        result = _replace_until_done_rec(c.op, root, node_tester, replace_f, stop_nodes, incl_node)
        if result:
            return result
    return None


def replace_until_done(root, node_tester, replace_f, stop_nodes=None, incl_node=None):
    new_g = root.graph
    new_root = root

    while True:
        result = _replace_until_done_rec(new_root, new_root, node_tester, replace_f, stop_nodes, incl_node)
        if result is None:
            break

        new_g, new_root, _ = result
        stop_nodes = set([new_g.get_operation_by_name(i.name) for i in stop_nodes])
        if incl_node is not None:
            incl_node = new_g.get_operation_by_name(incl_node.name)

    result = [new_g, new_root]
    if stop_nodes is not None:
        result.append(stop_nodes)
    if incl_node is not None:
        result.append(incl_node)
    return tuple(result)

_linear_types = ['Add', 'AddN', 'Sub', 'Mul', 'Identity', 'Sum', 'Assert', 'Reshape', 'Neg']
def find_s_stat_nodes(root, node, blanket, depth=0):
    '''
    Given an Op ```root``` and an Op ```node```, finds the set of
    nodes that ```root``` depends on that depend on ```node```
    _nonlinearly_---that is, nodes whose path to ```root``` are
    all linear (adds, multiplies, sums, etc.), and whose path to
    ```node``` immediately hits a nonlinearity.
    
    Args:
      root: The final node being computed (typically a log-joint probability).
      node: The node whose sufficient statistics we're interested in (typically a realization of a random variable).
      blanket: The set of nodes that define ```node```'s Markov blanket (typically all r.v.s in the graphical model).
    
    Returns:
      nodes: A list of unique nodes that compute sufficient statistics of ```node```.
    '''
    if root == node:
        return [root]
    elif root in blanket:
        return []
    elif (root.type in _linear_types):
        result = []
        for c in root.inputs:
            new_nodes = find_s_stat_nodes(c.op, node, blanket, depth=depth+1)
            result += new_nodes
#         if (root.type == 'Mul') and (len(result) > 1):
#             result = [root]
        return list(set(result))
    else:
        if is_child(root, node, blanket):
            return [root]
        else:
            return []

def canonicalize(s_stat, rv):
    rv = opify(rv)
    if (s_stat == rv):
        return 'x'
    sub_canons = ','.join(np.sort([canonicalize(c.op, rv) for c in s_stat.inputs]))
    if (s_stat.type == 'Identity'):
        return sub_canons
    return '%s(%s)' % (s_stat.type, sub_canons)

def compute_n_params(logp, s_stats, blanket):
    g = logp.graph
    g2 = tf.Graph()

    g2_s_stats = []
    result = []
    with g2.as_default():
        # Put in placeholders for all sufficient statistics and rvs
        g2_s_stats = [tf.placeholder(np.float32, name=i.name) for i in s_stats]
        g2_blanket = [tf.placeholder(np.float32, name=i.name) for i in blanket]
        # Copy logp's graph to g2, stopping when we reach those placeholders.
        g2_logp = tf.contrib.copy_graph.copy_op_to_graph(logp, g2, [])
        # For each sufficient statistic s, natural parameter is dlogp/ds.
        # Compute it and copy it back to the original graph.
        for s_stat in g2_s_stats:
            g2_n_param = tf.gradients(g2_logp, s_stat, name='n_params/n_param')[0]
            result.append(tf.contrib.copy_graph.copy_op_to_graph(g2_n_param, g, []))

    return result

def complete_conditional(logp, rv, blanket, name=None):
    new_logp = logp.op
    new_rv = opify(rv)
    new_blanket = set([opify(i) for i in blanket])
    new_g, new_logp, new_blanket, new_rv = replace_until_done(new_logp, make_is_type('Square'), square_to_mul,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_logp, new_blanket, new_rv = replace_until_done(new_logp, make_is_type('Div'), div_to_inv,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_logp, new_blanket, new_rv = replace_until_done(new_logp, is_log_mul, log_mul_to_add_log,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_logp, new_blanket, new_rv = replace_until_done(new_logp, is_log_inv, log_inv_to_neg_log,
                                                              stop_nodes=new_blanket, incl_node=new_rv)

    with new_g.as_default():
        rv = new_g.get_operation_by_name(opify(rv).name)
        blanket = [new_g.get_operation_by_name(opify(i).name) for i in blanket]
        s_stats = find_s_stat_nodes(new_logp, rv, blanket)
        n_params = compute_n_params(new_logp.outputs[0], s_stats, blanket)

        stat_param_dict = {}
        for i in xrange(len(s_stats)):
            print_tree(s_stats[i], stop_nodes=blanket)
            s_stat = canonicalize(s_stats[i], rv)
            print s_stat
            n_param = n_params[i]
            if (s_stat not in stat_param_dict):
                stat_param_dict[s_stat] = n_param
            else:
                stat_param_dict[s_stat] = tf.add(stat_param_dict[s_stat], n_param)

        s_stats = list(stat_param_dict.keys())
        order = np.argsort(s_stats)
        s_stats = np.array(s_stats)[order]
        n_params = [stat_param_dict[k] for k in s_stats]
    if new_g != logp.graph:
        n_params = [tf.contrib.copy_graph.copy_op_to_graph(i, logp.graph, [])
                    for i in n_params]

    if name is None:
        name = rv.name + '/conditional'
    s_stats = ';'.join(s_stats)
    with logp.graph.as_default():
        if (s_stats == 'Log(x);x'):
            return rvs.Gamma(alpha=tf.add(n_params[0], 1, name='add1'),
                             beta=-n_params[1], name=name, validate_args=False)
        elif (s_stats == 'Inv(x);Log(x)'):
            return rvs.InverseGamma(alpha=tf.add(-n_params[1], -1, name='add1'),
                                    beta=-n_params[0], name=name, validate_args=False)
        else:
            raise Exception('No implementation available for exponential family with sufficient statistics %s' % 
                            s_stats)

