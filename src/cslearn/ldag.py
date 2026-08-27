import warnings
from string import ascii_letters

import networkx as nx
import numpy as np
import pandas as pd

# pygraphviz shells out to `dot`/`circo` both for AGraph.layout() and again,
# separately, whenever Jupyter displays an AGraph (AGraph._repr_mimebundle_
# calls draw() a second time to render SVG). Both invocations surface the
# host's Fontconfig warnings as a RuntimeWarning; it reflects the font
# config, not anything about the graph being drawn, so it's noise regardless
# of which call triggers it -- hence a module-level filter rather than one
# scoped to plot_graphviz alone.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pygraphviz")


class LDAG(nx.DiGraph):
    def plot_graphviz(
        self,
        prog="dot",
        args="",
        with_legend=False,
        fontsize=10,
        compact=True,
        nodesep="0.05",
        ranksep="0.1",
        margin="0.02,0.01",
    ):
        """Render this LDAG via graphviz.

        Args:
            prog: graphviz layout engine.
            args: extra raw graphviz args, passed through to ``AGraph.layout``.
            with_legend: if True, replace each edge's (possibly long) context
                label with a single letter and return a dict mapping letters
                back to the original labels, instead of drawing the labels in
                full. Long context strings otherwise dominate the layout's
                footprint -- e.g. the Mushroom LDAG has edge labels up to 79
                characters, which alone pushed its native width to ~1400pt,
                far past what fits on a printed page even after shrinking
                every other lever.
            fontsize: pt size for graph/node/edge text (JCGS requires >=10pt
                *as printed*; since these figures are always embedded scaled
                down from native size, set the authored size here rather than
                relying on graphviz's 14pt default).
            compact: if True (default), use a tight box node shape/margin and
                nodesep/ranksep instead of graphviz's defaults (padded
                ellipses at 0.25in/0.5in spacing). Ellipses need extra room to
                keep text inside the oval boundary; for graphs with many
                nodes per rank this alone can be the difference between
                fitting on a page and not (confirmed on ALARM: default
                spacing/shape landed at 820pt native width, box+tight spacing
                at 426pt).
            nodesep, ranksep, margin: graphviz spacing/padding to use when
                ``compact`` is True. The defaults are enough for most graphs;
                denser ones (e.g. Mushroom's 22 nodes) may need tighter
                values to fit a printed page even after ``with_legend``.
        """
        if with_legend:
            strs = (letter for letter in ascii_letters)
            legend_dict = {}
            for pa, ch, attr in self.edges(data=True):
                if "label" not in attr:
                    continue
                new_label = next(strs)
                legend_dict[new_label] = attr["label"]
                attr["label"] = new_label
        agraph = nx.nx_agraph.to_agraph(self)
        # graphviz defaults to 14pt Times; JCGS requires figure text >=10pt, and
        # since these figures are always embedded at less than native size (see
        # \includegraphics[width=...] in the manuscript), the *effective* printed
        # size is smaller still unless the authored size is set explicitly here.
        # Node AND edge attrs both matter -- LDAG context labels live on edges.
        agraph.graph_attr["fontsize"] = fontsize
        agraph.node_attr["fontsize"] = fontsize
        agraph.edge_attr["fontsize"] = fontsize
        if compact:
            agraph.graph_attr["nodesep"] = nodesep
            agraph.graph_attr["ranksep"] = ranksep
            agraph.node_attr["shape"] = "box"
            agraph.node_attr["margin"] = margin
            agraph.node_attr["width"] = "0"
            agraph.node_attr["height"] = "0"
        agraph.layout(prog=prog, args=args)
        return (agraph, legend_dict) if with_legend else agraph


# Functions needed for generating LDAG representation from a dataframe representation of the staging
# of a CStree


def _convertToNumeric(df, alarmdf):
    npdf = df.to_numpy()
    vars = list(df.columns)
    n = len(df)
    for v in vars:
        j = vars.index(v)
        states = list(alarmdf[v].drop_duplicates().to_numpy())
        for i in range(n):
            npdf[i, j] = states.index(alarmdf[v].iloc[i])
    numdf = pd.DataFrame(npdf)
    return numdf


def _nodemask(v, w, df):
    dashmask = df[w] == "-"
    mask = dashmask & (df[v] != "-")
    return mask


def _getCSI(v, w, df):
    dfs = df[_nodemask(v, w, df)]
    n = len(dfs)
    vars = list(dfs.columns)
    d = len(vars)
    CSIs = []

    for i in range(n):
        A = []
        B = []
        Bcontexts = []
        for j in range(d):
            if dfs.iloc[i, j] == "*":
                A += [vars[j]]
            elif dfs.iloc[i, j] != "-":
                B += [vars[j]]
                Bcontexts += [dfs.iloc[i, j]]
        CSIs += [[w, A, B, Bcontexts]]

    return CSIs


def _collectCSIs(v, df):
    vars = list(df.columns)
    vidx = vars.index(v)
    prevvar = vars[vidx - 1]
    CSIs = _getCSI(prevvar, v, df)

    return CSIs


def _collectParents(v, df):
    CSIs = _collectCSIs(v, df)
    m = len(CSIs)
    parents = []
    for i in range(m):
        parents += CSIs[i][2]

    parents = list(dict.fromkeys(parents))

    return parents


def _collectVertexLabels(v, df):
    CSIs = _collectCSIs(v, df)
    m = len(CSIs)

    parents = _collectParents(v, df)
    padict = dict.fromkeys(parents)

    for i in range(m):
        CSIs[i].pop(0)
        CSIs[i].pop(0)

    labels = dict.fromkeys(parents)
    edgeLabels = {}

    for k in parents:
        padict[k] = []
        labels[k] = []
        for i in range(m):
            if CSIs[i][0].count(k) == 0:
                padict[k] += [CSIs[i]]
                labels[k] += [CSIs[i][1]]
        edgeLabels[(k, v)] = labels[k]

        kvanish = [len(x[0]) for x in padict[k]]
        if len(kvanish) != 0:
            if len(list(dict.fromkeys(kvanish))) != 1:
                print("Warning: different sized vanishing sets")

    edges = list(edgeLabels.keys())

    for e in edges:
        if edgeLabels[e] == []:
            del edgeLabels[e]

    return edgeLabels


def _collectLabels(df):
    num_nodes = len(list(df.columns))
    labels = {}
    for i in range(num_nodes):
        labels.update(_collectVertexLabels(i, df))

    return labels


def _updateEdges(dic, varorder):
    edges = list(dic.keys())
    for i in range(len(edges)):
        edges[i] = (varorder[edges[i][0]], varorder[edges[i][1]])

    return edges


def _getDAGmap(df):
    nodes = list(df.columns)
    num_nodes = len(nodes)
    adjmat = np.zeros([num_nodes, num_nodes], int)

    for v in nodes:
        v_parents = _collectParents(v, df)
        for w in v_parents:
            adjmat[w, v] = 1
    return adjmat
