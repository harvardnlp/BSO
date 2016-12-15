import sys
import nltk
from nltk.parse import DependencyGraph
from nltk.parse.transitionparser import TransitionParser, Transition, Configuration



data = [] 
parse = ""
for l in open(sys.argv[1]):
    if not l.strip(): 
        data.append(parse)
        parse = ""
    else:
        t = l.strip().split()
        # print " ".join([t[1], t[3], t[6], t[7].upper()]) 
        parse += " ".join([t[1], t[3], t[6], t[7].upper()]) + "\n"
        # parse += " ".join([t[1], t[4], t[8], t[10].upper()]) + "\n"

data.append(parse)


d = [ DependencyGraph(q) for q in data] 



class MyTransitionParser(TransitionParser):
    def _create_training_examples_arc_std(self, depgraphs, input_file):
        """
        ADAPTED FROM NTLK
        Create the training example in the libsvm format and write it to the input_file.
        Reference : Page 32, Chapter 3. Dependency Parsing by Sandra Kubler, Ryan McDonal and Joakim Nivre (2009)
        """
        operation = Transition(self.ARC_STANDARD)
        count_proj = 0
        training_seq = []

        for depgraph in depgraphs[:-1]:
            
            if not self._is_projective(depgraph):
                print >>sys.stderr, "fail non proj"
                continue

            count_proj += 1
            conf = Configuration(depgraph)
            print >>input_file, depgraph.nodes[conf.buffer[0]]["word"],
            while len(conf.buffer) > 0:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)


                if len(conf.stack) > 0:
                    s0 = conf.stack[len(conf.stack) - 1]
                    # Left-arc operation
                    rel = self._get_dep_relation(b0, s0, depgraph)
                    if rel is not None:
                        key = Transition.LEFT_ARC + ':' + rel
                        print >>input_file, "@L_"+rel,
                        # self._write_to_file(key, binary_features, input_file)
                        operation.left_arc(conf, rel)
                        training_seq.append(key)
                        continue

                    # Right-arc operation
                    rel = self._get_dep_relation(s0, b0, depgraph)
                    if rel is not None:
                        precondition = True
                        # Get the max-index of buffer
                        maxID = conf._max_address

                        for w in range(maxID + 1):
                            if w != b0:
                                relw = self._get_dep_relation(b0, w, depgraph)
                                if relw is not None:
                                    if (b0, relw, w) not in conf.arcs:
                                        precondition = False

                        if precondition:
                            if rel == "ROOT":
                                print >>input_file, "ROOT",
                            print >>input_file, "@R_"+rel,
                            # self._write_to_file(
                            #     key,
                            #     binary_features,
                            #     input_file)
                            operation.right_arc(conf, rel)
                            training_seq.append(key)
                            continue
                # Shift operation as the default
                key = Transition.SHIFT
                # print conf.buffer
                if len(conf.buffer) > 1:
                    print >>input_file, depgraph.nodes[conf.buffer[1]]["word"],

                # self._write_to_file(key, binary_features, input_file)
                operation.shift(conf)

                training_seq.append(key)
            print >>input_file, ""

        print(" Number of training examples : " + str(len(depgraphs)))
        print(" Number of valid (projective) examples : " + str(count_proj))
        return training_seq




MyTransitionParser(TransitionParser.ARC_STANDARD)._create_training_examples_arc_std(d, open("/tmp/targetparses.txt", "w"))

#java -cp stanford-parser-3.3.0.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -treeFile  ~/Projects/group/data/wsj/test.txt     -conllx -basic > test.conll.txt
