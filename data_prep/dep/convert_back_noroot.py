import sys
import nltk
from nltk.parse import DependencyGraph
from nltk.parse.transitionparser import TransitionParser, Transition, Configuration

for l in sys.stdin:
    words = []
    operation = Transition(TransitionParser.ARC_STANDARD)
    dg = DependencyGraph()
    conf = Configuration(dg)
    i = 1
    actions = l.strip().split()
    actions.extend(["ROOT", "@R_ROOT"]) # these are now implicit
    for action in actions:
        if action[0] == "@" and action[1] == "R":
            label = action.split("_")[1] if len(action.split("_")) > 1 else "R"
            operation.right_arc(conf, label) #action.split("_")[1])
        elif action[0] == "@" and action[1] == "L":
            label = action.split("_")[1] if len(action.split("_")) > 1 else "L"
            operation.left_arc(conf, label) #action.split("_")[1])
        else:
            words.append(action)
            conf.buffer.append(i)
            if i != 1:
                operation.shift(conf)
            i += 1
    final_label = action.split("_")[1] if len(action.split("_")) > 1 else "R"
    operation.right_arc(conf, label) #action.split("_")[1])
    arcs = {}
    for arc in conf.arcs:
        arcs[arc[2]] = (arc[0], arc[1] )
    for i, w in enumerate(words, 1):
        if i == len(words): continue
        print "\t".join([str(i), w, "_", "_", "_", "_", str(arcs[i][0]), arcs[i][1].lower(), "_", "_"])
    print
