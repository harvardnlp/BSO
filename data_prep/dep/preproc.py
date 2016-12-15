import sys
import os

def replace_digits(word):
    for i in xrange(10):
        word = word.replace(str(i), "#")
    return word

def prep_targ(direc, fi):
    """
    replaces digits, and gets rid of 'ROOT' and '@ROOT'
    """
    newfi = "targ-" + fi
    with open(os.path.join(direc, newfi), "w+") as g:
        with open(os.path.join(direc, fi)) as f:
            for line in f:
                tokens = [replace_digits(word) for word in line.strip().split()[:-2]]
                g.write("%s\n" % " ".join(tokens))

def prep_src(direc, fi):
    """
    replaces digits, and just keeps words
    """
    newfi = "src-" + fi
    with open(os.path.join(direc, newfi), "w+") as g:
        with open(os.path.join(direc, fi)) as f:
            for line in f:
                tokens = [replace_digits(toke) for toke in line.strip().split()[:-2] 
                           if not toke.startswith("@L_") and not toke.startswith("@R_")]
                g.write("%s\n" % " ".join(tokens))


direc, base = os.path.split(sys.argv[1])

prep_src(direc, base)
prep_targ(direc, base)
