import logging
import pickle
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.PermutationBased.GRaSP import grasp
from tqdm import tqdm

import cslearn.ldag as ldag
import cslearn.stage as st
from cslearn import dependence

logger = logging.getLogger(__name__)


def write_minimal_context_graphs_to_files(context_dags, prefix="mygraphs"):
    """Write minimal context graphs to files.

    Each context graph is written to a separate file; the context is included
    in the filename and as a figure label.

    Args:
        context_dags (dict): A dictionary of context graphs. The keys are the
            contexts, and the values are the graphs.
        prefix (str, optional): Filename prefix. Defaults to "mygraphs".

    Example:
        >>> # tree is the Figure 1 CStree
        >>> gs = tree.to_minimal_context_graphs()
        >>> ct.write_minimal_context_graphs_to_files(gs, prefix="mygraphs")
    """

    for key, val in context_dags.items():
        agraph = nx.nx_agraph.to_agraph(val)
        agraph.graph_attr["fontsize"] = 10
        agraph.node_attr["fontsize"] = 10
        agraph.edge_attr["fontsize"] = 10
        agraph.layout("dot")
        agraph.draw(prefix + str(key) + ".png", args='-Glabel="' + str(key) + '"   ')


def plot(graph, layout="dot", fontsize=10):
    """Plots a graph using graphviz. Creates a pygraphviz graph from a NetworkX graph.

    Args:
        graph (nx.Graph): The graph to plot.
        layout (str, optional): Graphviz layout engine. Defaults to "dot".
        fontsize (int, optional): Font size (pt) for graph/node/edge labels.
            Defaults to 10 (graphviz's own default is 14pt).

    Returns:
        pygraphviz.agraph.AGraph: A pygraphviz graph.
    """
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.graph_attr["fontsize"] = fontsize
    agraph.node_attr["fontsize"] = fontsize
    agraph.edge_attr["fontsize"] = fontsize
    agraph.layout(layout)
    return agraph


class CStree:
    """A class representing a CStree, see [1]. It is initialized by a list of cardinalities of the variables at each level and a list of labels. The labels are optional and defaults (None) to [0,1,...,p-1].

    Args:
        cards (list): A list of integers representing the cardinality of each level (indexed from 0).
        labels (list, optional): A list of strings or integers representing the labels of each level. Defaults to [0,1,...,p-1].
    References:
        [1] E. Duarte and L. Solus. Representation of context-specific causal models with observational and interventional data, 2021, https://arxiv.org/abs/2101.09271.

    Example:
        >>> # Figure 1. from (Duarte & Solus 2022)
        >>> import cslearn.cstree as ct
        >>> import cslearn.stage as st
        >>> tree = ct.CStree([2, 2, 2, 2], labels=["X"+str(i) for i in range(1, 5)])
        >>> tree.update_stages({
        ...     0: [{"context": {0: 0}},
        ...         {"context": {0: 1}}],
        ...     1: [{"context": {1: 0}, "color": "green"},
        ...         {"context": {0: 0, 1: 1}},
        ...         {"context": {0: 1, 1: 1}}],
        ...     2: [{"context": {0: 0, 2: 0}, "color": "blue"},
        ...         {"context": {0: 0, 2: 1}, "color": "orange"},
        ...         {"context": {0: 1, 2: 0}, "color": "red"},
        ...         {"context": {0: 1, 1: 1, 2: 1}},
        ...         {"context": {0: 1, 1: 0, 2: 1}}]})
        >>> tree.sample_stage_parameters()
    """

    def __init__(self, cards=[], labels=None):
        self.tree = None
        self.stages = None
        self.cards = cards
        self.p = len(cards)
        if labels is None:
            self.labels = list(range(self.p))
        else:
            self.labels = labels

        # self.colors = list(set(mcolors.cnames.keys()))
        self.colors = [
            "peru",
            "blueviolet",
            "orange",
            "navy",
            "rebeccapurple",
            "darkseagreen",
            "aquamarine",
            "goldenrodyellow",
            "cornsilk",
            "azure",
            "chocolate",
            "red",
            "darkolivegreen",
            "chartreuse",
            "turquoise",
            "olive",
            "crimson",
            "goldenrod",
            "orchid",
            "firebrick",
            "lawngreen",
            "deeppink",
            "wheat",
            "teal",
            "mediumseagreen",
            "salmon",
            "palegreen",
            "navajowhite",
            "yellowgreen",
            "mediumaquamarine",
            "darkcyan",
            "dodgerblue",
            "brown",
            "powderblue",
            "mistyrose",
            "violet",
            "darkslategrey",
            "midnightblue",
            "aliceblue",
            "dimgrey",
            "palegoldenrod",
            "black",
            "darkgrey",
            "olivedrab",
            "linen",
            "lightblue",
            "thistle",
            "greenyellow",
            "indianred",
            "khaki",
            "lightslategrey",
            "slateblue",
            "purple",
            "deepskyblue",
            "magenta",
            "yellow",
            "ivory",
            "darkorchid",
            "mediumpurple",
            "snow",
            "dimgray",
            "palevioletred",
            "darkslateblue",
            "sandybrown",
            "lightgray",
            "lemonchiffon",
            "gray",
            "silver",
            "aqua",
            "tomato",
            "lightyellow",
            "seagreen",
            "darkmagenta",
            "beige",
            "cornflowerblue",
            "peachpuff",
            "ghostwhite",
            "cyan",
            "lightcoral",
            "hotpink",
            "lightpink",
            "lightskyblue",
            "slategrey",
            "tan",
            "oldlace",
            "steelblue",
            "springgreen",
            "fuchsia",
            "lime",
            "papayawhip",
            "mediumblue",
            "mediumspringgreen",
            "darkorange",
            "lightgreen",
            "blue",
            "slategray",
            "white",
            "saddlebrown",
            "mediumturquoise",
            "paleturquoise",
            "darkblue",
            "plum",
            "lightseagreen",
            "lightgrey",
            "blanchedalmond",
            "lavenderblush",
            "darkkhaki",
            "gainsboro",
            "lightsalmon",
            "darkturquoise",
            "moccasin",
            "darkgoldenrod",
            "mediumorchid",
            "honeydew",
            "mediumslateblue",
            "maroon",
            "forestgreen",
            "darkgray",
            "floralwhite",
            "darkgreen",
            "lightcyan",
            "darksalmon",
            "pink",
            "royalblue",
            "sienna",
            "green",
            "orangered",
            "bisque",
            "antiquewhite",
            "rosybrown",
            "whitesmoke",
            "darkred",
            "burlywood",
            "skyblue",
            "mediumvioletred",
            "mintcream",
            "limegreen",
            "lightsteelblue",
            "grey",
            "coral",
            "indigo",
            "gold",
            "cadetblue",
        ]
        self.color_no = 0
        self.stages = {i: [] for i in range(self.p)}

    def stage_proportion(self, stage: st.Stage):
        """The proportion of the outcome space this stage represents.

        Args:
            stage (st.Stage): A stage of this CStree.

        Returns:
            float: A number between 0 and 1.

        Example:
            >>> # Assuming all variables are binary
            >>> s = st.Stage([0, {0, 1}, 1])
            >>> stree.stage_proportion(s)
            0.25
        """
        prop = 1
        for i, val in enumerate(stage.list_repr):
            if not isinstance(val, set):
                prop *= 1 / self.cards[i]

        return prop

    def update_stages(self, stages: dict):
        """Update/set the stages of the CStree.

        Args:
            stages (dict): A dictionary of stage dicts. The keys are the levels, and the values are lists of dicts representing stages. The dicts representing stages should have the following keys: "context" and "color". The "context" should have a dict as value, where the keys are the levels and the values are the values of the variables at that level. The "color" key is optional but should have a string as value, representing the color of the stage.

        Example:
            >>> tree.update_stages({
            ...     0: [{"context": {0: 0}},
            ...         {"context": {0: 1}}],
            ...     1: [{"context": {1: 0}, "color": "green"},
            ...         {"context": {0: 0, 1: 1}},
            ...         {"context": {0: 1, 1: 1}}],
            ...     2: [{"context": {0: 0, 2: 0}, "color": "blue"},
            ...         {"context": {0: 0, 2: 1}, "color": "orange"},
            ...         {"context": {0: 1, 2: 0}, "color": "red"},
            ...         {"context": {0: 1, 1: 1, 2: 1}},
            ...         {"context": {0: 1, 1: 0, 2: 1}}]
            ...     })
        """

        # This should be filled with Stage objects.
        stages_to_add = {key: [] for key in stages.keys()}

        # If there are dicts, we convert them to Stages.
        # Stages are updated to contain a cardinality list, inherited from the
        # CStree.
        for lev, list_of_stage_repr in stages.items():
            # it can be either a Stage of a dict that should be converted to a
            # stage.

            for stage_repr in list_of_stage_repr:
                if isinstance(stage_repr, dict):
                    # If its a dict, we convert it to a stage.
                    stage_list_repr = []
                    for l in range(lev + 1):
                        if l in stage_repr["context"]:
                            stage_list_repr.append(stage_repr["context"][l])
                        else:
                            stage_list_repr.append(set(range(self.cards[l])))
                    # Create a stage from the stage_list_repr
                    s = st.Stage(stage_list_repr)
                    if "color" in stage_repr:
                        s.color = stage_repr["color"]

                    s.cards = self.cards
                    stages_to_add[lev].append(s)
                else:
                    # Just add the stage and set the cards
                    stage_repr.cards = self.cards
                    stages_to_add[lev].append(stage_repr)

        self.stages.update(stages_to_add)
        if -1 not in self.stages:
            self.stages[-1] = [st.Stage([], color="black")]

    def get_stage(self, node: tuple):
        """Get the stage of a node in the CStree.

        Args:
            node (tuple or list): A node in the CStree. It could be e.g. (0, 1, 0, 1).
        Example:
            >>> # tree is the fig. 1 CStree
            >>> stage = tree.get_stage([0, 0])
            >>> print(stage)
            [{0, 1}, 0]; probs: [0.38 0.62]; color: green
        """
        assert self.stages is not None
        lev = len(node) - 1

        stage = None
        if lev in self.stages:
            for s in self.stages[lev]:
                if node in s:
                    stage = s
                    break

        if stage is None:
            print("No stage found for {}".format(node))

        assert stage is not None
        return stage

    def to_df(self, write_probs=False):
        """Converts the CStree to a Pandas dataframe.
        The labels of the dataframe are the labels of the levels/variables in the CStree.
        The first row contains the cardinalities of the variables.
        The other rows contain the stages of the CStree. E.g. row number 3 contains the level 1 stage with context X2=0, etc.

        Args:
            write_probs (bool): If True, the probabilities of the stages are written to the dataframe. Defaults to False.
        Returns:
            df (pd.DataFrame): A Pandas dataframe with the stages of the CStree.
        Example:
            >>> tree.to_df()
                X1	X2	X3	X4
            0	2	2	2	2
            1	0	-	-	-
            2	1	-	-	-
            3	*	0	-	-
            4	0	1	-	-
            5	1	1	-	-
            6	0	*	0	-
            7	0	*	1	-
            8	1	*	0	-
            9	1	1	1	-
            10	1	0	1	-
            11	-	-	-	-
        """

        # cardinalities header
        d = {self.labels[i]: [c] for i, c in enumerate(self.cards)}
        max_card = max(self.cards)

        if write_probs:
            labs = self.labels + ["PROB_" + str(i) for i in range(max_card)]
        else:
            labs = self.labels
        df = pd.DataFrame(d, columns=labs)
        dfstmp = []
        for l, stages in self.stages.items():
            for s in stages:
                dftmp = s.to_df(labs, max_card=max_card, write_probs=write_probs)
                dfstmp += [dftmp]
        df = pd.concat([df] + dfstmp)
        df.reset_index(drop=True, inplace=True)

        return df

    def sample_stage_parameters(self, alpha=1):
        """Set the parameters of the stages of the CStree to be random samples
        from a Dirichlet distribution with hyper parameter alpha.

        Args:
            alpha (float): The hyper parameter for the Dirichlet distribution.

        Example:
            >>> t = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
            >>> t.sample_stage_parameters()
        """
        for lev, stages in self.stages.items():
            for i, stage in enumerate(stages):
                probs = np.random.dirichlet([alpha] * self.cards[lev + 1])
                stage.probs = probs
                if stage.color is None:
                    # Set color from stage if possible
                    stage.color = self.colors[i]

    def estimate_stage_parameters(self, data, method="BDeu", alpha_tot=1):
        """Estimate the parameters of the stages of the CStree under a Dirichlet model.

        Args:
            data (pd.DataFrame): A pandas dataframe with the data.
            method (str): The estimation method. Currently only "BDeu" [2] is implemented.
            alpha_tot (float): Total Dirichlet pseudo-count, split proportionally
                across stages by their size.

        Reference:
            [2] C. Hughes, P. Strong, and A. Shenvi. Score equivalence for staged trees, 2023,
            https://arxiv.org/abs/2206.15322

        Example:
            >>> t = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
            >>> t.sample_stage_parameters()
            >>> df = t.sample(500)
            >>> t.estimate_stage_parameters(df, alpha_tot=1.0, method="BDeu")
            >>> for lev, stagings in t.stages.items():
            ...    print("Level {}".format(lev))
            ...    for stage in stagings:
            ...        print(stage)
            ...    print()
        """
        import cslearn.scoring as sc

        # Set stage probabilities
        for lev, stages in self.stages.items():
            if lev == self.p - 1:
                continue

            stage_counts = sc._counts_at_level(self, lev + 1, data)

            for stage in stages:
                probs = sc._estimate_parameters(self, stage, stage_counts, method, alpha_tot)
                stage.probs = probs

    def _set_tree_probs(self, alpha=1):
        """This is dependent on if one has sampled from the tree already.
        I.e., if a probablity is already set for an edge, it
        should not be overwritten.
        """
        # First create the tree.
        # This has to do with that, if we allow for singleton stages, we need to
        # make sure it is not too big sort of.

        self._create_tree()

        # Check if the node is part part of a context (stage?)
        # if so we may overwrite probs. Otherwise, generate new ones.
        for node in self.tree.nodes():
            if len(node) == self.p:
                continue
            children = sorted(list(self.tree.successors(node)))
            probs = np.random.dirichlet([alpha] * self.cards[len(node)])
            # Seems like a stage at level l holds the probtable for the
            # variabel at level l+1.
            for i, ch in enumerate(children):
                stage = self.get_stage(node)

                if stage is not None:  # No singleton stages allowed!
                    prob = stage.probs[i]
                    self.tree[node][ch]["cond_prob"] = prob
                    self.tree[node][ch]["label"] = round(prob, 2)
                    self.tree[node][ch]["color"] = stage.color
                    self.tree.nodes[node]["color"] = stage.color
                else:  # This should never happen as all nodes should be part of a stage.
                    self.tree[node][ch]["cond_prob"] = probs[i]
                    self.tree[node][ch]["label"] = round(probs[i], 2)

    def _get_stage_no(self, node):
        """Get the stage number of a node in a level of the cstree.
        This is used when coloring the nodes.

        Args:
            node (tuple): A node in the CStree.
        """
        lev = len(node)
        for stages in self.stages[lev]:
            for i, stage_dict in enumerate(stages):
                if node in stage_dict:
                    return i
        return None

    def _create_tree(self):
        if self.tree is None:
            self.tree = nx.DiGraph()

        # All the levels of the first variable.
        tovisit = [(i,) for i in range(self.cards[0])]
        while len(tovisit) != 0:
            # Visit/create node in the tree
            node = tovisit.pop()
            lev = len(node) - 1  # added -1
            fr = node[:lev]
            to = node
            stage = self.get_stage(fr)
            self.tree.add_edge(fr, to)  # check if exists first

            if stage is not None:  # No singleton stages allowed!
                if stage.probs is not None:
                    # The last digit/element in the node is the variable value
                    prob = stage.probs[to[-1]]
                    self.tree[fr][to]["cond_prob"] = prob
                    self.tree[fr][to]["label"] = round(prob, 2)
                self.tree[fr][to]["color"] = stage.color
                self.tree.nodes[fr]["color"] = stage.color

            if fr == ():
                self.tree.nodes[fr]["label"] = "ø"
            self.tree.nodes[to]["label"] = to[-1]
            # Add more nodes to visit
            if lev < self.p - 1:
                for i in range(self.cards[lev + 1]):
                    tovisit.append(to + (i,))

    # Here is the only place we need labels/orders.
    def to_minimal_context_graphs(self):
        """This returns a dict of minimal context NetworkX DAGs.

        Returns:
            dict: The keys are the contexts, and the values are the NetworkX DAGs.

        Example:
            >>> # tree is the Figure 1 CStree
            >>> gs = tree.to_minimal_context_graphs()
            >>> for key, graph in gs.items():
            ...     print("{}: Edges {}".format(key, graph.edges()))
            X2=0: Edges [('X1', 'X4'), ('X3', 'X4')]
            X3=0: Edges [('X1', 'X2'), ('X1', 'X4')]
            X1=0: Edges [('X2', 'X3'), ('X3', 'X4')]

        """
        minl_csis_by_context = self.to_minimal_context_csis()
        cdags = dependence.csi_relations_to_dags(minl_csis_by_context, self.p, labels=self.labels)

        return cdags

    def to_minimal_context_agraphs(self, layout="dot"):
        """This returns a dict   of minimal context DAGs as pgraphviz graphs.

        Args:
            layout (str, optional): Graphviz engine. Defaults to "dot".

        Returns:
           pygraphviz.agraph.AGraph: pygraphviz graphs
        """

        graphs = self.to_minimal_context_graphs()
        agraphs = {}
        for key, graph in graphs.items():
            agraphs[key] = nx.nx_agraph.to_agraph(graph)
            agraphs[key].graph_attr["fontsize"] = 10
            agraphs[key].node_attr["fontsize"] = 10
            agraphs[key].edge_attr["fontsize"] = 10
            agraphs[key].layout(layout)

        return agraphs

    def to_minimal_context_csis(self):
        """This returns a dict of minimal context CSIs.

        Example:
            >>> minlcsis = tree.to_minimal_context_csis()
            >>> for key, csis in minlcsis.items():
            ...     for csi in csis:
            ...         print("{}: CSI {}".format(key, csi))
            X2=0: CSI X1 ⊥ X3 | X2=0
            X3=0: CSI X2 ⊥ X4 | X1, X3=0
            X1=0: CSI X2 ⊥ X4 | X3, X1=0

        """

        logger.debug("Stages")

        for key, val in self.stages.items():
            logger.debug("level {}".format(key))
            for s in val:
                logger.debug(s)
        logger.debug("Getting csi rels per level")
        rels = self.csi_relations_per_level()
        logger.debug("CSI relations per level")
        for key, rel in rels.items():
            logger.debug("level {}: ".format(key))

            for r in rel:
                logger.debug("the CSI")
                logger.debug(r)
                logger.debug("cards: {}".format(r.cards))

        paired_csis = dependence._csis_by_levels_2_by_pairs(rels, cards=self.cards)  # this could still be per level?
        logger.debug("\n###### Paired_csis")
        logger.debug(paired_csis)

        logger.debug("\n ######### minl cslisist")
        # The pairs may change during the process this is why we have by pairs.
        # However, the level part
        # should always remain the same actually, so may this is not be needed.
        minl_csislists = dependence.minimal_csis(paired_csis, self.cards)  # this could be by level still?
        logger.debug(minl_csislists)

        logger.debug("\n ############### get minl csis in list format")
        minl_csis = dependence._csi_lists_to_csis_by_level(
            minl_csislists, self.p, labels=self.labels
        )  # this would not be needed
        logger.debug(minl_csislists)
        for key in minl_csislists:
            for pair, val in key.items():
                logger.debug("{}: {}".format(pair, val))

        logger.debug("#### minimal csis")
        minl_csis_by_context = dependence._rels_by_level_2_by_context(minl_csis)
        logger.debug(minl_csis_by_context)
        for pair, val in minl_csis_by_context.items():
            for csi in val:
                logger.debug(csi)

        return minl_csis_by_context

    def csi_relations_per_level(self):
        """Get the context specific independence relations per level.

        Returns:
            dict: The CSI relations per level. The keys are the levels, and the values are lists of CSI relations.

        Example:

            >>> rels = tree.csi_relations_per_level()
            >>> print("CSI relations per level")
            >>> for key, rel in rels.items():
            ...     print("level {}: ".format(key))
            ...     for r in rel:
            ...         if (len(r.ci.a) > 0) and (len(r.ci.b) > 0):
            ...             # avoiding the singletons
            ...             print("the CSI")
            ...             print(r)
            CSI relations per level
            level 0:
            level 1:
            the CSI
            0 ⊥ 2 | 1=0
            level 2:
            the CSI
            1 ⊥ 3 | 0=0, 2=0
            the CSI
            1 ⊥ 3 | 0=0, 2=1
            the CSI
            1 ⊥ 3 | 0=1, 2=0
            level 3:

        """

        return {l: [s.to_csi() for s in stages] for l, stages in self.stages.items() if l >= 0}

    def csi_relations(self, level="all"):
        """Returns the context specific indepencende (CSI) relations for the CStree.

        Examples:
            >>> rels = tree.csi_relations()
            >>> for cont, rels in rels.items():
            ...     for rel in rels:
            ...         print(rel)
            X1 ⊥ X3 | X2=0
            X2 ⊥ X4 | X1=0, X3=0
            X2 ⊥ X4 | X1=0, X3=1
            X2 ⊥ X4 | X1=1, X3=0
        """
        csi_rels = {}

        for key, stage_list in self.stages.items():
            for stage in stage_list:
                if stage.is_singleton():
                    # As these dont encode any independence relations.
                    continue
                csi_rel = stage.to_csi(labels=self.labels)

                if csi_rel.context not in csi_rels.keys():
                    csi_rels[csi_rel.context] = [csi_rel]
                else:
                    csi_rels[csi_rel.context].append(csi_rel)

        return csi_rels

    def sample(self, n: int):
        """Draws random samples from the CStree. It also dynamically generates nodes in the underlying tree and associated parameters on the fly in order to avoid creating the whole
        tree, which is O(2^p) (if singeltone stages are allowed), just to sample data. This is also useful when plotting the tree, as it will only contain the nodes that are actually used in the sampling.

        Args:
            n (int): number of random samples.

        Returns:
            pandas.DataFrame: A pandas dataframe containing the data. The header contains labels of the
            variables and row 0 contains the cardinalities. The other rows contain the samples.
        Examples:
            >>> df = tree.sample(5)
            >>> print(df)
               0  1  2  3
            0  2  2  2  2
            1  0  0  1  1
            2  0  0  0  0
            3  0  0  0  1
            4  0  0  1  1
            5  0  0  0  1
        """

        # TODO: Check that the stage parameters are set.
        if self.tree is None:
            self.tree = nx.DiGraph()
        xs = []

        xs.append(self.cards)  # cardinalities at the first row

        for _ in range(n):
            node = ()
            x = []
            while len(x) < self.p:
                # Create tree dynamically if isnt already crated.
                if (node not in self.tree) or len(self.tree.out_edges(node)) == 0:
                    lev = len(node) - 1
                    edges = [(node, node + (ind,)) for ind in range(self.cards[lev + 1])]

                    self.tree.add_edges_from(edges)

                    # We set the parametres at to whats specifies in the stage.
                    s = self.get_stage(node)
                    color = ""
                    if s is not None:  # not singleton
                        probs = s.probs
                        if s.color is None:  # Color if the color isnt set already
                            s.color = self.colors[self.color_no]
                            self.color_no += 1

                        color = s.color

                    # Hope this gets in the right order
                    edges = list(self.tree.out_edges(node))

                    # Set parameters
                    for i, e in enumerate(edges):
                        self.tree[e[0]][e[1]]["cond_prob"] = probs[i]
                        self.tree[e[0]][e[1]]["label"] = round(probs[i], 2)
                        self.tree.nodes[e[1]]["label"] = e[1][-1]
                        self.tree[e[0]][e[1]]["color"] = color
                        self.tree.nodes[e[0]]["color"] = color

                        # Coloring the child too
                        if len(e[1]) < self.p:
                            child_stage = self.get_stage(e[1])
                            self.tree.nodes[e[1]]["color"] = child_stage.color

                        # We should color all children here.

                edges = list(self.tree.out_edges(node))

                probabilities = [self.tree[e[0]][e[1]]["cond_prob"] for e in edges]

                ind = np.random.choice(len(edges), 1, p=probabilities)[0]
                node = edges[ind][1]  # This is a typle like (0,1,1,0)
                x.append(node[-1])  # Take the last element, 0, above

            xs.append(x)

        df = pd.DataFrame(xs)
        df.columns = self.labels
        return df

    def plot(self, full=False):
        """Plot the CStree.

        Args:
            full (bool): If True, the complete tree is drawn with all nodes.
                Defaults to False, in which case only nodes visited during
                sampling are shown (pass ``full=True`` for a fresh tree with
                no sampling history).

        Returns:
            pygraphviz.agraph.AGraph: A pygraphviz graph.

        Examples:
            >>> tree.sample_stage_parameters()
            >>> agraph = tree.plot(full=True)
            >>> agraph.draw("cstree.png")
        """

        # If no samples has been drawn, create the full tree.
        if full:  # or (self.tree is None):
            self._create_tree()
        else:
            print("Use plot(full=True) to draw the full tree.")

        if self.tree is None:
            return
        agraph = plot(self.tree)
        # agraph.node_attr["shape"] = "circle"

        ###############################################################
        ### This is a hack to plot labels. Just an invisible graph. ###
        for i, lab in enumerate(self.labels):
            agraph.add_node(lab, color="white")

        agraph.add_node("-", color="white")
        agraph.add_edge("-", self.labels[0], color="white")

        for i, lab in enumerate(self.labels[:-1]):
            agraph.add_edge(self.labels[i], self.labels[i + 1], color="white")

        agraph.layout("dot")
        return agraph

    def predict(
        self,
        partial_observations: pd.DataFrame,
        return_probs: bool = False,
    ) -> pd.DataFrame:
        """Predict the most likely completion of partially-observed rows.

        Given a CStree over random variables :math:`X_{[p]}` and a DataFrame
        of partial observations, returns the MAP completion of the unobserved
        variables for each row.

        Args:
            partial_observations: DataFrame whose columns are the labels of the
                observed variables and whose rows are individual observations.
                Pass a single-row DataFrame with no columns
                (``pd.DataFrame(index=[0])``) to predict all variables
                unconditionally.
            return_probs: If ``True``, append a ``PROB`` column containing the
                normalized conditional probability of the MAP completion.

        Returns:
            DataFrame with one column per predicted variable (in label order).
            If ``return_probs=True``, a ``PROB`` column is appended.

        Example:

            >>> import random
            >>> import numpy as np
            >>> import pandas as pd
            >>> import cslearn.cstree as ct
            >>> np.random.seed(22)
            >>> random.seed(22)
            >>> t = ct.sample_cstree([3, 2, 2, 3], max_cvars=2, prob_cvar=0.5, prop_nonsingleton=1)
            >>> t.sample_stage_parameters(alpha=2)
            >>> t.sample(100)
            >>> t.predict(pd.DataFrame({0: [1]}))
               1  2  3
            0  0  1  1
        """
        partial_labels = list(partial_observations.columns)
        to_predict_labels = [l for l in self.labels if l not in partial_labels]

        # Corner case: all variables are observed — return observations unchanged.
        if not to_predict_labels:
            result = partial_observations[self.labels].copy()
            if return_probs:
                result["PROB"] = 1.0
            return result

        label_order = partial_labels + to_predict_labels

        def pmf(outcome):
            return self.pmf(outcome, label_order)

        outcome_completions = list(product(*(range(self.cards[self.labels.index(l)]) for l in to_predict_labels)))

        num_preds = len(partial_observations)
        preds = np.empty((num_preds, len(to_predict_labels)), int)
        if return_probs:
            probs = np.empty(num_preds, float)

        obs_iter = partial_observations.values if partial_labels else [[] for _ in range(num_preds)]
        for idx, partial in enumerate(obs_iter):
            outcomes = (list(partial) + list(c) for c in outcome_completions)
            preds[idx] = max(outcomes, key=pmf)[-len(to_predict_labels) :]
            if return_probs:
                outcomes = (list(partial) + list(c) for c in outcome_completions)
                outcome_probs = list(map(pmf, outcomes))
                probs[idx] = max(outcome_probs) / sum(outcome_probs)

        preds_df = pd.DataFrame(preds, columns=to_predict_labels)
        if return_probs:
            preds_df["PROB"] = probs
        return preds_df

    def fit(
        self,
        data: pd.DataFrame,
        gibbs_samples=10000,
        poss_cvars=None,
        grasp_depth=3,
        grasp_score="local_score_BDeu",
        max_cvars=2,
        alpha_tot=1.0,
        method="BDeu",
        save_as="",
    ):
        """Learn a CStree from data using GRaSP + Gibbs order sampling (CSlearn).

        Runs the three-phase CSlearn pipeline:
        1. DAG pre-screening via GRaSP (or uses caller-supplied ``poss_cvars``).
        2. Gibbs order MCMC to find the MAP variable ordering.
        3. Exact staging search to find the optimal CStree for that ordering.

        Updates ``self`` in place and returns it.

        Args:
            data (pd.DataFrame): Training data; first column may be cardinalities
                (matching the format returned by :meth:`sample`).
            gibbs_samples (int): Number of Gibbs iterations for order sampling.
                Defaults to 10000.
            poss_cvars (dict, optional): Pre-computed dict mapping each variable
                label to its list of possible context variables. If ``None``
                (default), GRaSP is run on ``data`` to produce this dict.
            grasp_depth (int): GRaSP search depth. Defaults to 3.
            grasp_score (str): Scoring function for GRaSP. Defaults to
                ``"local_score_BDeu"``.
            max_cvars (int): Maximum context-set size β. Defaults to 2.
            alpha_tot (float): Total BDeu pseudo-count. Defaults to 1.0.
            method (str): Scoring method, currently only ``"BDeu"`` is supported.
            save_as (str): If non-empty, save intermediate results (``poss_cvars``,
                score tables, learned tree, …) to ``{save_as}-<name>.pkl`` files.

        Returns:
            CStree: ``self``, updated with the learned structure and parameters.

        Example:
            >>> import pandas as pd
            >>> import cslearn.cstree as ct
            >>> tree = ct.sample_cstree([2, 2, 2, 2], max_cvars=1, prob_cvar=0.5)
            >>> tree.sample_stage_parameters()
            >>> df = tree.sample(500)
            >>> learned = ct.CStree(tree.cards).fit(df)
        """
        import cslearn.learning as ctl
        import cslearn.scoring as sc

        if poss_cvars is None:
            # estimate possible context variables and create score tables
            graph = grasp(
                data.values[1:],
                score_func=grasp_score,
                depth=grasp_depth,
            )
            poss_cvars = ctl.causallearn_graph_to_posscvars(graph, labels=data.columns, alg="grasp")

        score_table, context_scores, _ = sc.order_score_tables(
            data,
            max_cvars=max_cvars,
            alpha_tot=alpha_tot,
            method=method,
            poss_cvars=poss_cvars,
        )

        # run Gibbs sampler to get MAP order
        orders, scores = ctl.gibbs_order_sampler(gibbs_samples, score_table)
        map_order = orders[scores.index(max(scores))]

        # estimate CStree and stage params
        opt_tree = ctl._optimal_cstree_given_order(map_order, context_scores)
        opt_tree.estimate_stage_parameters(data, alpha_tot=2.0, method="BDeu")

        if save_as != "":
            to_saves = [
                poss_cvars,
                score_table,
                context_scores,
                orders,
                scores,
                opt_tree,
            ]
            names = [
                "poss_cvars",
                "score_table",
                "context_scores",
                "orders",
                "scores",
                "opt_tree",
            ]
            for to_save, name in zip(to_saves, names):
                with open(f"{save_as}-{name}.pkl", "wb") as f:
                    pickle.dump(to_save, f)

        old_labels = self.labels
        self.__dict__.update(opt_tree.__dict__)
        if len(old_labels):
            self.labels = [old_labels[idx] for idx in self.labels]
        return self

    def to_LDAG(self, return_contexts=False):
        """Return the LDAG representation of the CStree.

        Args:
            return_contexts: if True, also return a dict mapping each edge to
                its CSI relations as ``(variable, value)`` pairs, rather than
                just the bare context values the edge's own ``"label"``
                attribute carries. Useful for building a human-readable
                context table.

        Returns:
            nx.DiGraph: A DAG whose edges carry CSI-relation labels. Node labels
            match ``self.labels``. If ``return_contexts`` is True, returns a
            ``(LDAG, contexts)`` tuple instead, where ``contexts`` maps each
            edge to a list of its CSI relations, each relation a list of
            ``(variable, value)`` pairs.
        """

        df = self.to_df()
        # convert variable names to integer values for purposes of finding LDAG representation
        # These are later converted back to the variable names using an option in plotLDAG
        df.columns = range(len(df.columns))

        num_nodes = len(list(df.columns))
        adjMat = ldag._getDAGmap(df)
        labels = ldag._collectLabels(df)

        LDAG = ldag.LDAG(adjMat)
        varorder = self.labels

        newEdges = ldag._updateEdges(labels, varorder)
        newLabels = dict.fromkeys(newEdges)
        labelkeys = list(labels.keys())
        for i in range(len(labelkeys)):
            newLabels[newEdges[i]] = labels[labelkeys[i]]

        newVertices = {}
        for i in range(num_nodes):
            newVertices[i] = varorder[i]

        LDAG = nx.relabel_nodes(LDAG, newVertices)
        nx.set_edge_attributes(LDAG, newLabels, "label")

        if not return_contexts:
            return LDAG

        contexts = ldag._collectLabels(df, keep_vars=True)
        contextEdges = ldag._updateEdges(contexts, varorder)
        contextkeys = list(contexts.keys())
        namedContexts = {}
        for i in range(len(contextkeys)):
            relations = contexts[contextkeys[i]]
            namedContexts[contextEdges[i]] = [
                [(varorder[var_idx], value) for var_idx, value in relation] for relation in relations
            ]

        return LDAG, namedContexts

    def pmf(self, x, label_order=None):
        """Calculate the probability mass function of a given outcome.

        Args:
            x (list): Outcome values, one per variable. Order follows ``label_order``
                if provided, otherwise ``self.labels``.
            label_order (list, optional): Permutation of ``self.labels`` describing
                the order of values in ``x``. Defaults to ``self.labels``.

        Returns:
            float: The probability mass of the outcome.
        """
        if label_order is None:
            label_order = self.labels

        x = [x[label_order.index(l)] for l in self.labels]

        prob = 1
        for i, val in enumerate(x):
            stage = self.get_stage(x[:i])
            prob *= stage.probs[val]

        return prob

    def pmf_log(self, x, label_order=None):
        """Calculate the log probability mass function of a given outcome.

        Args:
            x (list): A list of values representing the outcome.

        Returns:
            float: The log probability mass of the outcome.
        """
        if label_order is None:
            label_order = self.labels

        # relabel the outcome
        x = [x[label_order.index(l)] for l in self.labels]

        log_prob = 0
        for i, val in enumerate(x):
            stage = self.get_stage(x[:i])
            if stage.probs[val] == 0:
                return -np.inf
            log_prob += np.log(stage.probs[val])

        return log_prob

    def to_joint_distribution(self, label_order=None, with_outcomes=True):
        """Return the joint distribution of the CStree.

        Args:
            label_order (list, optional): A list of labels from self.labels in the desired order. Defaults to None which means self.labels.

        Returns:
            Pandas Dataframe: The joint distribution of the CStree.

        Example:
            >>> df = tree.to_joint_distribution()
            >>> print(df)
                X1	X2	X3	X4	prob	log_prob
            0	0	0	0	0	0.283088	-1.261999
            1	0	0	0	1	0.219686	-1.515555
            2	0	0	1	0	0.061755	-2.784584
            3	0	0	1	1	0.307910	-1.177948
            4	0	1	0	0	0.004319	-5.444837
            5	0	1	0	1	0.003351	-5.698393
            6	0	1	1	0	0.005404	-5.220655
            7	0	1	1	1	0.026943	-3.614019
            8	1	0	0	0	0.003439	-5.672556
            9	1	0	0	1	0.025955	-3.651409
            10	1	0	1	0	0.020695	-3.877848
            11	1	0	1	1	0.000916	-6.995125
            12	1	1	0	0	0.002284	-6.081627
            13	1	1	0	1	0.017241	-4.060480
            14	1	1	1	0	0.012563	-4.377006
            15	1	1	1	1	0.004451	-5.414586
        """

        ""
        if label_order is None:
            label_order = self.labels

        # Iterate over all possible outcomes and calculate the probability mass function.
        # Store the outcomes together with the probabilities in a Pandas Dataframe.
        outcomes = product(*[range(card) for card in self.cards])
        n_outcomes = np.prod(self.cards)

        # Create an empty dataframe with the correct column names
        df_outcomes = pd.DataFrame(columns=label_order)
        # store all the outcomes and probabilities
        pmfs = [None] * np.prod(self.cards)
        pmfs_log = [None] * np.prod(self.cards)

        for i, outcome in tqdm(enumerate(outcomes), total=n_outcomes, desc="Calculating joint distribution"):
            df_outcomes.loc[i] = outcome
            pmfs[i] = self.pmf(outcome, label_order)
            pmfs_log[i] = self.pmf_log(outcome, label_order)

        df_pmf = pd.DataFrame(pmfs, columns=["prob"])
        df_pmf_log = pd.DataFrame(pmfs_log, columns=["log_prob"])
        # join the two dataframes
        if with_outcomes:
            df = pd.concat([df_outcomes, df_pmf, df_pmf_log], axis=1)
        else:
            df = pd.concat([df_pmf, df_pmf_log], axis=1)

        return df


def sample_cstree(
    cards: list[int],
    max_cvars: int,
    prob_cvar: float,
    prop_nonsingleton: float = 1.0,
    labels: list | None = None,
) -> CStree:
    """Sample a random CStree with given cardinalities.

    Args:
        cards (list): cardinalities of the levels.
        max_cvars (int): maximum number of context variables.
        prob_cvar (float): probability a potential context variable (in the algorithm) will a context variable.
        prop_nonsingleton (float): proportion of non-singleton stages. Defaults to 1.0, meaning no singletons, which is also what the algorithms in this library are designed for.
        labels (list, optional): A list of strings or integers representing the labels of each level. Defaults to None which means [0,1,...,p-1].

    Returns:
        CStree: A CStree.
    Examples:
        >>> np.random.seed(1)
        >>> random.seed(1)
        >>> tree = ct.sample_cstree([2,2,2,2], max_cvars=1, prob_cvar=0.5, prop_nonsingleton=1)
        >>> tree.to_df()
            0	1	2	3
        0	2	2	2	2
        1	0	-	-	-
        2	1	-	-	-
        3	1	*	-	-
        4	0	*	-	-
        5	*	*	0	-
        6	*	*	1	-
    """
    p = len(cards)

    if labels is None:
        labels = list(range(p))
    ct = CStree(cards, labels=labels)

    stagings = {}
    for level, val in enumerate(cards[:-1]):  # not the last level
        tmpstage = st.Stage([set(range(cards[l])) for l in range(level + 1)], cards=cards)

        stage_space = [tmpstage]

        full_stage_space_size = stage_space[0].size()
        singleton_space_size = full_stage_space_size

        # Need to adjust the max_cvars for the low levels.
        mc = max_cvars
        if level < mc:
            mc = level + 1  # should allow for "singleton" stages at level 0 as this
            # this correponds to the half of the space.
        # in general, when the level is the liming factor, the number of stages
        # should be as many as possible. Hence, mc = level +1
        # It should not make abig difference. But in this way, we actually show
        # all the stages expicitly, even when saving to file, without risking
        # to blow up the space.

        minimal_stage_size = np.prod(
            sorted(cards[: level + 1])[:-mc], dtype=int
        )  # take the product of all cards, expept for the mc largest ones

        # special case when the granularity is to coarse. Randomly add anyway?
        # Setting the upper boudn likje this may generalize. However, the stage
        # should not always be accepted.. prop_nonsingleton =
        # max(prop_nonsingleton, minimal_stage_size / full_state_space_size)

        # Allowing only non-singleton stages, this should never happen.
        if prop_nonsingleton < minimal_stage_size / full_stage_space_size:
            logger.info(
                "The size (proportion {}) of a minimal is larger than {}.".format(
                    minimal_stage_size / full_stage_space_size, prop_nonsingleton
                )
            )
            b = np.random.multinomial(1, [prob_cvar, 1 - prob_cvar], size=1)[0][0]
            stagings[level] = []

            if b == 0:
                print(b)
                # it should be the whole space to begin with
                stage_restr = stage_space.pop(0)
                new_stage = st.sample_stage_restr_by_stage(stage_restr, mc, 1.0, cards)
                stagings[level].append(new_stage)
            continue  # cant add anything anyway so just go to the next level.
        stagings[level] = []

        while (1 - (singleton_space_size / full_stage_space_size)) < prop_nonsingleton:
            colored_size_old = full_stage_space_size - singleton_space_size
            # Choose randomly a stage space

            space_int = np.random.randint(len(stage_space))
            stage_restr = stage_space.pop(space_int)
            new_stage = st.sample_stage_restr_by_stage(stage_restr, mc, prob_cvar, cards)

            new_space = stage_restr - new_stage  # this is a list of substages after removal
            stage_space += new_space  # adds a list of stages
            # subtract the size of the sampled stage
            singleton_space_size -= new_stage.size()
            colored_prop = 1 - singleton_space_size / full_stage_space_size
            # Check it its over full after adding the new space
            if colored_prop > prop_nonsingleton:
                # Check if it would be possible to add the smallest stage.
                # Restore the old space and maybe try again

                if (minimal_stage_size + colored_size_old) / full_stage_space_size <= prop_nonsingleton:
                    stage_space = stage_space[: -len(new_space)]
                    stage_space += [stage_restr]
                    singleton_space_size += new_stage.size()
                else:
                    # This is when it is impossible to push in a new stages
                    # since all are be too big.
                    break
            else:
                stagings[level].append(new_stage)

    stagings[-1] = [st.Stage([], color="black")]

    # Color each stage in the optimal staging. Singletons are black.
    # This should be done somewhere else probably.
    colors = [
        "peru",
        "blueviolet",
        "orange",
        "navy",
        "rebeccapurple",
        "darkseagreen",
        "darkslategray",
        "lightslategray",
        "aquamarine",
        "lightgoldenrodyellow",
        "cornsilk",
        "azure",
        "chocolate",
        "red",
        "darkolivegreen",
    ]

    for level, staging in stagings.items():
        for i, stage in enumerate(staging):
            if (level == -1) or ((level > 0) and all([isinstance(i, int) for i in stage.list_repr])):
                stage.color = "black"
            else:
                stage.color = colors[i]

    ct.update_stages(stagings)

    return ct


def df_to_cstree(df, read_probs=True):
    """Convert a dataframe to a CStree. The dataframe should have the following format:
    The labels should be the level labels.
    The first row should be the cardinalities, the second row should be the first stage, the third row the second stage etc. (see :meth:`cslearn.cstree.CStree.to_df()`).

    Args:
        df (Pandas DataFrame): The dataframe to convert.

    Returns:
        CStree: A CStree.

    Example:
        >>> df = tree.to_df()
        >>> print(df)
        >>> t2 = ct.df_to_cstree(df)
        >>> df2 = t2.to_df()
        >>> print("The same tree:")
        >>> print(df)
            X1 X2 X3 X4
        0   2  2  2  2
        1   0  -  -  -
        2   1  -  -  -
        3   *  0  -  -
        4   0  1  -  -
        5   1  1  -  -
        6   0  *  0  -
        7   0  *  1  -
        8   1  *  0  -
        9   1  1  1  -
        10  1  0  1  -
        11  -  -  -  -
        The same tree:
            X1 X2 X3 X4
        0   2  2  2  2
        1   0  -  -  -
        2   1  -  -  -
        3   *  0  -  -
        4   0  1  -  -
        5   1  1  -  -
        6   0  *  0  -
        7   0  *  1  -
        8   1  *  0  -
        9   1  1  1  -
        10  1  0  1  -
        11  -  -  -  -
    """
    collabs = list(df.columns)

    # Gets the number of varaibles.
    # If no probs are given, then the number of variables is
    # the number of columns.
    if "PROB_0" in collabs:
        has_probs = True
        nvars = collabs.index("PROB_0")
    else:
        has_probs = False
        nvars = len(collabs)

    cards = df.iloc[0, :nvars].values.astype(int)  # [int(x[1]) for x in df.columns]

    stagings = {i: [] for i in range(-1, len(cards))}

    cstree = CStree(cards)
    cstree.labels = list(df.columns[:nvars])

    color_ind = 0
    for row in df.iloc[1:, :nvars].iterrows():
        stage_list = []
        for level, val in enumerate(row[1]):
            if level == len(cards):
                break
            if val == "*":
                stage_list.append(set(range(cards[level])))
            elif val == "-":
                # Reached stop mark "-", so create a stage of stage_list.
                s = st.Stage(stage_list)

                if s.size() == 1:  # Singleton stage? Or maybe root stage?.
                    s.color = "black"
                else:
                    s.color = cstree.colors[color_ind]
                    color_ind = (color_ind + 1) % len(cstree.colors)
                # Now we reed the probabilities, if there are any
                if has_probs:
                    s.probs = df.iloc[row[0], nvars:].values

                stagings[level - 1].append(s)

                break
            else:
                stage_list.append(int(val))

    cstree.update_stages(stagings)

    return cstree
