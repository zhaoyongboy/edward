import tensorflow as tf
import tensorflow as tf
distributions = tf.contrib.distributions
sg = tf.contrib.bayesflow.stochastic_graph
import numpy as np
import re

import edward as ed
import edward.util as util
import edward.models.random_variables as rvs
from edward.models.random_variable import RANDOM_VARIABLE_COLLECTION

def print_tree(op, depth=0, stop_nodes=None, stop_types=None):
    if stop_nodes is None: stop_nodes = set()
    if stop_types is None: stop_types = set()
    print ''.join(['-'] * depth), '%s...%s' % (op.type, op.name)
    if (op not in stop_nodes) and (op.type not in stop_types):
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
        if c == possible_child:
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
# TODO: Make transformations work with slicing, reshaping, adding 0. (Mul(x, 1) should already work.)


def opify(node):
    if tf.is_numeric_tensor(node):
        return node.op
    else:
        return node


# Functions for graph rewriting

def copy_inputs(old_inputs, new_g):
    new_inputs = []
    for input in old_inputs:
        new_inputs.append(tf.contrib.copy_graph.copy_op_to_graph(input.op, new_g, []).outputs[0])
    return new_inputs


def replace_node(node, root, replace_f):
    '''
    Creates a new graph, copies the inputs to `node` to that graph,
    applies `replace_f()` to transform `node` into a new (typically
    equivalent) set of operations with the same outputs. Returns the
    new graph, root, and node.
    '''
    # TODO: Move copy_inputs() logic into replace_node(). Maybe don't return graph?
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


def is_square_mul(node):
    return (node.type == 'Square') and (de_identity(node.inputs[0].op).type == 'Mul')


def square_mul_to_mul_square(node, new_g, scope):
    new_inputs = copy_inputs(de_identity(node.inputs[0].op).inputs, new_g)
    return tf.mul(tf.square(new_inputs[0]), tf.square(new_inputs[1]), name=scope)


def is_square_add(node):
    return (node.type == 'Square') and (de_identity(node.inputs[0].op).type == 'Add')


def square_add_expand(node, new_g, scope):
    new_inputs = copy_inputs(de_identity(node.inputs[0].op).inputs, new_g)
    return tf.add(tf.square(new_inputs[0]) + tf.square(new_inputs[1]),
                  2 * new_inputs[0] * new_inputs[1], name=scope)


def is_square_sub(node):
    return (node.type == 'Square') and (de_identity(node.inputs[0].op).type == 'Sub')


def square_sub_expand(node, new_g, scope):
    new_inputs = copy_inputs(de_identity(node.inputs[0].op).inputs, new_g)
    return tf.add(tf.square(new_inputs[0]) + tf.square(new_inputs[1]),
                  -2 * new_inputs[0] * new_inputs[1], name=scope)


_pow_types = {'Square':2., 'Sqrt':0.5, 'Inv':-1.}
_pow_type_fs = {'Square':tf.square, 'Sqrt':tf.sqrt, 'Inv':tf.inv}
_pow_types_inv = {2.:tf.square, 0.5:tf.sqrt, -1:tf.inv}
def is_pow_composition(node):
    return (node.type in _pow_types) and (de_identity(node.inputs[0].op).type in _pow_types)


def get_square_sqrt_inv_exponent(node):
    '''
    If node is a composition of Square, Sqrt, and Inv nodes, then this
    function returns the exponent of and input to that composition. E.g.,
    `get_square_sqrt_inv_exponent(tf.square(tf.inv(x)))`
    returns (-2, x), and
    `get_square_sqrt_inv_exponent(tf.square(tf.sqrt(tf.inv(tf.inv(x)))))`
    returns (1, x).
    '''
    if node.type in _pow_types:
        child = de_identity(node.inputs[0].op)
        if child.type in _pow_types:
            below_exp, bottom = get_square_sqrt_inv_exponent(child)
            return _pow_types[node.type] * below_exp, bottom
        else:
            return _pow_types[node.type], child
    return 1., None


def simplify_square_sqrt_inv(node, new_g, scope):
    exponent, bottom = get_square_sqrt_inv_exponent(node)
    new_bottom = tf.contrib.copy_graph.copy_op_to_graph(bottom, new_g, [])
    if exponent in _pow_types_inv:
        return _pow_types_inv[exponent](new_bottom.outputs[0], name=scope)
    else:
        return tf.pow(new_bottom.outputs[0], exponent, name=scope)


def is_pow_type_mul(node):
    return (node.type in _pow_types) and (de_identity(node.inputs[0].op).type == 'Mul')


def swap_pow_type_mul(node, new_g, scope):
    new_inputs = copy_inputs(de_identity(node.inputs[0].op).inputs, new_g)
    pow_type_f = _pow_type_fs[node.type]
    return tf.mul(pow_type_f(new_inputs[0]), pow_type_f(new_inputs[1]), name=scope)


def is_log_pow_type(node):
    return (node.type == 'Log') and (de_identity(node.inputs[0].op).type in _pow_types)


def simplify_log_pow_type(node, new_g, scope):
    child = de_identity(node.inputs[0].op)
    new_inputs = copy_inputs(child.inputs, new_g)
    return tf.mul(_pow_types[child.type], tf.log(new_inputs[0]), name=scope)


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


_linear_types = ['Add', 'AddN', 'Sub', 'Mul', 'Neg', 'Identity', 'Sum', 'Assert', 'Reshape', 'Slice', 'Gather', 'GatherNd']
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
    elif root.type in _linear_types:
        result = []
        for c in root.inputs:
            new_nodes = find_s_stat_nodes(c.op, node, blanket, depth=depth+1)
            result += new_nodes
        return list(set(result))
    else:
        if is_child(root, node, blanket):
            return [root]
        else:
            return []


def _get_const_value(op):
    value = op.get_attr('value')
    for field in value.ListFields():
        if field[0].name[-4:] == '_val':
            result = np.array(field[1], np.float32)
            if np.prod(result.shape) == 1:
                return result[0]
            else:
                return result
    assert(False)
    

_identity_types = ['Identity', 'Cast']
def canonicalize(s_stat, rv):
    rv = opify(rv)
    if s_stat == rv:
        return 'x'
    if s_stat.type == 'Const':
        return str(_get_const_value(s_stat))
    sub_canons = ','.join(np.sort([canonicalize(c.op, rv) for c in s_stat.inputs]))
    if s_stat.type in _identity_types:
        return sub_canons
    return '%s(%s)' % (s_stat.type, sub_canons)


def compute_n_params(log_prob, s_stats, blanket):
    g = log_prob.graph
    new_g = tf.Graph()

    new_g_s_stats = []
    result = []
    with new_g.as_default():
        # Put in placeholders for all sufficient statistics and rvs
        new_g_s_stats = [tf.placeholder(np.float32, name=i.name) for i in s_stats]
        new_g_blanket = [tf.placeholder(i.outputs[0].dtype, name=i.name) for i in blanket]
        # Copy log_prob's graph to new_g, stopping when we reach those placeholders.
        new_g_log_prob = tf.contrib.copy_graph.copy_op_to_graph(log_prob, new_g, [])
        # For each sufficient statistic s, natural parameter is dlog_prob/ds.
        # Compute it and copy it back to the original graph.
        for s_stat in new_g_s_stats:
            new_g_n_param = tf.gradients(new_g_log_prob, s_stat, name='n_params/n_param')[0]
            # TODO: assert(new_g_n_param is not None)
            result.append(tf.contrib.copy_graph.copy_op_to_graph(new_g_n_param, g, []))
#         # Put in placeholders for all sufficient statistics and rvs
#         new_g_s_stats = [tf.placeholder(np.float32, name=i.name) for i in s_stats]
#         new_g_blanket = [tf.placeholder(np.float32, name=i.name) for i in blanket]
#         # Copy log_prob's graph to new_g, stopping when we reach those placeholders.
#         new_g_log_prob = tf.contrib.copy_graph.copy_op_to_graph(log_prob, new_g, [])
#         # For each sufficient statistic s, natural parameter is dlog_prob/ds.
#         # Compute it and copy it back to the original graph.
#         for s_stat in new_g_s_stats:
#             new_g_n_param = tf.gradients(new_g_log_prob, s_stat, name='n_params/n_param')[0]
#             # TODO: assert(new_g_n_param is not None)
#             result.append(tf.contrib.copy_graph.copy_op_to_graph(new_g_n_param, g, []))

    return result


_supports = {rvs.Exponential.__name__:'nn_real',
             rvs.Gamma.__name__:'nn_real',
             rvs.InverseGamma.__name__:'nn_real',
             rvs.Normal.__name__:'real',
             rvs.Bernoulli.__name__:'binary',
             rvs.BernoulliWithSigmoidP.__name__:'binary',
             rvs.Beta.__name__:'unit',
             rvs.BetaWithSoftplusAB.__name__:'unit'}
def complete_conditional(rv, blanket, **kwargs):
    g = rv.value().graph
    blanket_values = [item.value() for item in blanket]
    log_joint_name = 'log_joint_of_' + '_and_'.join([item.name for item in blanket])
    log_joint_name = re.sub('/', '.', log_joint_name)
    support = _supports[type(rv).__name__]

    if log_joint_name in [item.name for item in g.get_operations()]:
        log_joint = g.get_operation_by_name(log_joint_name)
    else:
        log_probs = []
        with g.as_default():
            with tf.name_scope(log_joint_name) as scope:
                for rv_i in blanket:
                    log_prob_i = rv_i.conjugate_log_prob(rv_i)
#                     log_prob_i = rv_i.log_prob(rv_i)
                    log_probs.append(tf.reduce_sum(log_prob_i))
                log_joint = tf.add_n(log_probs, name=scope).op
    return _complete_conditional_imp(log_joint, rv.value(), blanket_values, support, **kwargs)


def _complete_conditional_imp(log_prob, rv, blanket, support, name=None, validate_args=True):
#    new_log_prob = log_prob.op
    new_log_prob = log_prob
    new_rv = opify(rv)
    new_blanket = set([opify(i) for i in blanket])
    # TODO: This isn't guaranteed to apply every transformation as many times as needed.
    new_g, new_log_prob, new_blanket, new_rv = replace_until_done(new_log_prob, make_is_type('Div'), div_to_inv,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_log_prob, new_blanket, new_rv = replace_until_done(new_log_prob, is_log_mul, log_mul_to_add_log,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_log_prob, new_blanket, new_rv = replace_until_done(new_log_prob, is_log_pow_type, simplify_log_pow_type,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_log_prob, new_blanket, new_rv = replace_until_done(new_log_prob, is_square_add, square_add_expand,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_log_prob, new_blanket, new_rv = replace_until_done(new_log_prob, is_square_sub, square_sub_expand,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_log_prob, new_blanket, new_rv = replace_until_done(new_log_prob, is_pow_type_mul, swap_pow_type_mul,
                                                              stop_nodes=new_blanket, incl_node=new_rv)
    new_g, new_log_prob, new_blanket, new_rv = replace_until_done(new_log_prob, is_pow_composition, simplify_square_sqrt_inv,
                                                              stop_nodes=new_blanket, incl_node=new_rv)

    if name is None:
        name = rv.name + '/conditional'
    with new_g.as_default():
        rv = new_g.get_operation_by_name(opify(rv).name)
        blanket = [new_g.get_operation_by_name(opify(i).name) for i in blanket]
        s_stats = find_s_stat_nodes(new_log_prob, rv, blanket)
        n_params = compute_n_params(new_log_prob.outputs[0], s_stats, blanket)

        stat_param_dict = {}
        for i in xrange(len(s_stats)):
            print_tree(s_stats[i], stop_nodes=blanket)
            s_stat = canonicalize(s_stats[i], rv)
            print s_stat
            n_param = n_params[i]
            if s_stat not in stat_param_dict:
                stat_param_dict[s_stat] = n_param
            else:
                with tf.name_scope(name) as scope:
                    stat_param_dict[s_stat] = tf.add(stat_param_dict[s_stat], n_param, name='n_param_sum')

        s_stats = list(stat_param_dict.keys())
        order = np.argsort(s_stats)
        s_stats = np.array(s_stats)[order]
        n_params = [stat_param_dict[k] for k in s_stats]
    if new_g != log_prob.graph:
        n_params = [tf.contrib.copy_graph.copy_op_to_graph(i, log_prob.graph, [])
                    for i in n_params]

    s_stats = ';'.join(s_stats)
    support_s_stats = support + '|' + s_stats
    print 's_stats are %s' % s_stats
    print 'n_params are ', [i.op.name for i in n_params]
    maker = _support_s_stats_to_dist_fn.get(support_s_stats)
    with log_prob.graph.as_default():
        with tf.name_scope(name) as scope:
            if maker is not None:
                rv_class, params = maker(n_params)
                params['name'] = scope
                params['validate_args'] = validate_args
                return rv_class(**params)
    raise Exception('No implementation available for exponential family with support %s and sufficient statistics %s' % 
                    (support, s_stats))

def _make_gamma(n_params):
    return rvs.Gamma, {'alpha':tf.add(n_params[0], 1, name='add1'),
                       'beta':-n_params[1]}
def _make_inverse_gamma(n_params):
    return rvs.InverseGamma, {'alpha':tf.add(-n_params[1], -1, name='add1'),
                              'beta':-n_params[0]}
def _make_normal(n_params):
    sigmasq = tf.inv(-2*n_params[0])
    mu = n_params[1] * sigmasq
    return rvs.Normal, {'sigma':tf.sqrt(sigmasq), 'mu':mu}
def _make_bernoulli(n_params):
    return rvs.BernoulliWithSigmoidP, {'p':n_params[0]}
def _make_beta(n_params):
    return rvs.Beta, {'a':tf.add(n_params[1], 1, name='a_add1'),
                      'b':tf.add(n_params[0], 1, name='b_add1')}
    
_support_s_stats_to_dist_fn = {}
_support_s_stats_to_dist_fn['nn_real|Log(x);x'] = _make_gamma
_support_s_stats_to_dist_fn['nn_real|Inv(x);Log(x)'] = _make_inverse_gamma
_support_s_stats_to_dist_fn['real|Square(x);x'] = _make_normal
_support_s_stats_to_dist_fn['binary|x'] = _make_bernoulli
_support_s_stats_to_dist_fn['unit|Log(Sub(1.0,x));Log(x)'] = _make_beta
