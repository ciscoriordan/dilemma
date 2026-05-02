"""Label definitions for POS tagging and dependency parsing.

Unified label sets covering both Modern Greek (el) and Ancient Greek (grc).
el (Modern Greek) labels from gr-nlp-toolkit configs. grc (Ancient Greek) labels expanded from UD Perseus
and PROIEL treebanks.
"""

# UPOS tags + morphological feature label maps.
# Each key maps to an ordered list of labels (index -> label string).
# el-original labels are listed first to preserve index compatibility
# with gr-nlp-toolkit weights. AG additions are appended at the end.
pos_labels = {
    'Abbr': ['_', 'Yes'],
    'Aspect': ['Perf', '_', 'Imp'],
    'Case': ['Dat', '_', 'Acc', 'Gen', 'Nom', 'Voc', 'Loc'],
    'Definite': ['Ind', 'Def', '_'],
    'Degree': ['Cmp', 'Sup', '_', 'Pos'],
    'Foreign': ['_', 'Yes'],
    'Gender': ['Fem', 'Masc', '_', 'Neut', 'Fem,Masc', 'Masc,Neut'],
    'Mood': ['Ind', '_', 'Imp', 'Opt', 'Sub'],
    'NumType': ['Mult', 'Card', '_', 'Ord', 'Sets'],
    'Number': ['Plur', '_', 'Sing', 'Dual'],
    'Person': ['3', '1', '_', '2'],
    'Polarity': ['_', 'Neg'],
    'Poss': ['_', 'Yes'],
    'PronType': ['Ind', 'Art', '_', 'Rel', 'Dem', 'Prs', 'Ind,Rel', 'Int', 'Rcp'],
    'Reflex': ['_', 'Yes'],
    'Tense': ['Pres', 'Past', '_', 'Fut', 'Pqp'],
    'VerbForm': ['Part', 'Conv', '_', 'Inf', 'Fin', 'Gdv'],
    'Voice': ['Pass', 'Act', '_', 'Mid', 'Mid,Pass'],
    'upos': [
        'X', 'PROPN', 'PRON', 'ADJ', 'AUX', 'PART', 'ADV', '_',
        'DET', 'SYM', 'NUM', 'CCONJ', 'PUNCT', 'NOUN', 'SCONJ',
        'ADP', 'VERB', 'INTJ',
    ],
}

# Valid morphological features per UPOS tag.
# Unified for el + grc. grc adds Polarity, Reflex, and extra values.
pos_properties = {
    'ADJ': ['Degree', 'Number', 'Gender', 'Case'],
    'ADP': ['Number', 'Gender', 'Case'],
    'ADV': ['Degree', 'Abbr', 'Polarity'],
    'AUX': ['Mood', 'Aspect', 'Tense', 'Number', 'Person', 'VerbForm', 'Voice'],
    'CCONJ': [],
    'DET': ['Number', 'Gender', 'PronType', 'Definite', 'Case'],
    'INTJ': [],
    'NOUN': ['Number', 'Gender', 'Abbr', 'Case'],
    'NUM': ['NumType', 'Number', 'Gender', 'Case'],
    'PART': ['Polarity'],
    'PRON': ['Number', 'Gender', 'Person', 'Poss', 'PronType', 'Case', 'Reflex'],
    'PROPN': ['Number', 'Gender', 'Case'],
    'PUNCT': [],
    'SCONJ': [],
    'SYM': [],
    'VERB': ['Mood', 'Aspect', 'Tense', 'Number', 'Gender', 'Person',
             'VerbForm', 'Voice', 'Case'],
    'X': ['Foreign'],
    '_': [],
}

# Dependency relation labels (index -> label string).
# el-original labels first, AG additions appended.
dp_labels = [
    'obl', 'obj', 'dep', 'mark', 'case', 'flat', 'nummod', 'obl:arg',
    'punct', 'cop', 'acl:relcl', 'expl', 'nsubj', 'csubj:pass', 'root',
    'advmod', 'nsubj:pass', 'ccomp', 'conj', 'amod', 'xcomp', 'aux',
    'appos', 'csubj', 'fixed', 'nmod', 'iobj', 'parataxis', 'orphan',
    'det', 'advcl', 'vocative', 'compound', 'cc', 'discourse', 'acl',
    'obl:agent',
    # grc additions
    'advcl:cmp', 'aux:pass', 'dislocated', 'flat:name', 'nsubj:outer',
]

# Number of el-original labels (for gr-nlp-toolkit weight compatibility).
# gr-nlp-toolkit weights have this many outputs per head.
EL_POS_LABEL_COUNTS = {
    'Abbr': 2, 'Aspect': 3, 'Case': 6, 'Definite': 3, 'Degree': 3,
    'Foreign': 2, 'Gender': 4, 'Mood': 3, 'NumType': 5, 'Number': 3,
    'Person': 4, 'Poss': 2, 'PronType': 8, 'Tense': 3, 'VerbForm': 5,
    'Voice': 3, 'upos': 17,
}
EL_DP_LABEL_COUNT = 37
