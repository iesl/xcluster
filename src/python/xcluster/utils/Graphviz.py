class Graphviz(object):
    def __init__(self):
        self.internal_color = "lavenderblush4"
        self.colors = [
            "aquamarine", "bisque", "blue", "blueviolet", "brown", "cadetblue",
            "chartreuse", "coral", "cornflowerblue", "crimson", "darkgoldenrod",
            "darkgreen", "darkkhaki", "darkmagenta", "darkorange", "darkred",
            "darksalmon", "darkseagreen", "darkslateblue", "darkslategrey",
            "darkviolet", "deepskyblue", "dodgerblue", "firebrick",
            "forestgreen", "gainsboro", "ghostwhite", "gold", "goldenrod",
            "gray", "grey", "green", "greenyellow", "honeydew", "hotpink",
            "indianred", "indigo", "ivory", "khaki", "lavender",
            "lavenderblush", "lawngreen", "lemonchiffon", "lightblue",
            "lightcoral", "lightcyan", "lightgoldenrodyellow", "lightgray",
            "lightgreen", "lightgrey", "lightpink", "lightsalmon",
            "lightseagreen", "lightskyblue", "lightslategray", "lightslategrey",
            "lightsteelblue", "lightyellow", "limegreen", "linen", "magenta",
            "maroon", "mediumaquamarine", "mediumblue", "mediumorchid",
            "mediumpurple", "mediumseagreen", "mediumslateblue",
            "mediumturquoise", "midnightblue", "mintcream", "mistyrose",
            "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab",
            "orange", "orangered", "orchid", "palegoldenrod", "palegreen",
            "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru",
            "pink", "powderblue", "purple", "red", "rosybrown", "royalblue",
            "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell",
            "sienna", "silver", "skyblue", "slateblue", "slategray",
            "slategrey", "snow", "springgreen", "steelblue", "tan", "teal",
            "thistle", "tomato", "violet", "wheat", "burlywood", "chocolate"]
        self.color_map = {}
        self.color_counter = 0

    def format_id(self, ID):
        if not ID.startswith("id"):
            return ("id%s" % ID).replace('-', '')\
                .replace('#', '_HASH_').replace('.', '_DOT_')
        else:
            return ("%s" % ID).replace('-', '')\
                .replace('#', '_HASH_').replace('.', '_DOT_')

    def clean_label(self, s):
        return s.replace("[/:.]", "_")

    def get_node_label(self, node):
        lbl = []
        lbl.append(self.format_id(node.id))
        lbl.append('<BR/>')
        lbl.append('num pts: %d' % len(node.leaves()))
        lbl.append('<BR/>')
        try:
            lbl.append('purity: %f' % node.purity())
        except Exception:
            pass
        try:
            lbl.append('<BR/>')
            lbl.append('across: %s' % node.best_across_debug)
        except Exception:
            pass
        return ''.join(lbl)

    def get_color(self, lbl):
        if lbl in self.color_map:
            return self.color_map[lbl]
        else:
            self.color_map[lbl] = self.colors[self.color_counter]
            self.color_counter = (self.color_counter + 1) % len(self.colors)
            return self.color_map[lbl]

    def format_graphiz_node(self, node):
        """Format a graphviz node for printing."""
        s = []
        color = self.internal_color
        try:
            if node.purity() == 1.0:
                if hasattr(node, 'pts') and len(node.pts) > 0:
                    w_gt = [x for x in node.pts if x[1] and x[1] != "None"]
                    if w_gt:
                        color = self.get_color(w_gt[0][1])
                    else:
                        color = self.get_color('None')
        except Exception:
            pass
        shape = 'egg'

        if node.parent is None:
            s.append(
                '\n%s[shape=%s;style=filled;color=%s;label=<%s<BR/>%s<BR/>>]'
                % (self.format_id(node.id), shape, color,
                   self.get_node_label(node), color))
            s.append(
                '\nROOTNODE[shape=star;style=filled;color=gold;label=<ROOT>]')
            s.append('\nROOTNODE->%s' % self.format_id(node.id))
        else:
            leaf_m = ''
            if hasattr(node, 'pts') and node.pts and len(node.pts) > 0:
                if hasattr(node.pts[0][0], 'mid'):
                    leaf_m = '%s|%s' % (node.pts[0][0].mid, node.pts[0][0].gt) \
                        if node.is_leaf() else ''
                else:
                    leaf_m = '%s|%s' % (node.pts[0][2], node.pts[0][1]) \
                        if node.is_leaf() else ''
            s.append('\n%s[shape=%s;style=filled;color=%s;label=<%s<BR/>'
                     '%s<BR/>%s<BR/>>]'
                % (self.format_id(node.id), shape, color,
                   self.get_node_label(node), color, leaf_m))
            s.append('\n%s->%s' % (self.format_id(node.parent.id),
                                   self.format_id(node.id)))
        return ''.join(s)

    def graphviz_tree(self, root,):
        """Return a graphviz tree as a string."""
        s = []
        s.append('digraph TreeStructure {\n')
        s.append(self.format_graphiz_node(root))
        for d in root.descendants():
            s.append(self.format_graphiz_node(d))
        s.append('\n}')
        return ''.join(s)

    @staticmethod
    def write_tree(filename, root):
        """Write a graphviz tree to a file."""
        gv = Graphviz()
        tree = gv.graphviz_tree(root)
        with open(filename, 'w') as fout:
            fout.write(tree)
