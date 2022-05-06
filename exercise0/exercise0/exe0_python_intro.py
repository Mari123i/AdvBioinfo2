import string
from typing import Dict, List, Tuple
import numpy as np
from numpy import argmax

AAs: str = 'ALGVSREDTIPKFQNYMHWCUZO'


def get_aa_composition(protein_seq: str) -> Dict[str, int]:
    counter = {}
    for aa in protein_seq:
        if aa not in string.ascii_uppercase:
            raise ValueError('Not a valid AA')
        if aa in counter.keys():
            counter[aa] = counter[aa] + 1
        else:
            counter[aa] = 1
    return counter


def gen(alphabet):  # helper for k_mers
    while (True):
        for a in alphabet:
            yield a
        yield "$"


def k_mers(alphabet: str, k: int) -> List[str]:
    alphabet = alphabet[:: -1]
    k_mers = []
    generate = []
    pointer = k - 1
    tmp = []
    # initialize
    for i in range(k):
        generator = gen(alphabet)
        generate.append(generator)
        tmp.append(next(generator))
    k_mers.append("".join(tmp))
    got = ""
    while pointer != -1 or "$" not in got:
        got = next(generate[pointer])
        if "$" not in got:
            tmp[pointer] = got
            k_mers.append("".join(tmp))
            pointer = k - 1
        else:
            tmp[pointer] = next(generate[pointer])
            pointer -= 1
    return k_mers


# helper function
def getAlphabet(seq):
    alphabet = []
    for a in seq:
        if a in alphabet:
            pass
        else:
            alphabet.append(a)
    return "".join(alphabet)


def get_kmer_composition(protein_seq: str, k: int) -> Dict[str, int]:
    alphabet = getAlphabet(protein_seq)
    k_mer_list = k_mers(alphabet, k)
    k_mer_dict = { k_mer:0 for k_mer in k_mer_list}
    for i in range(0,len(protein_seq)-k+1):
        tmp = protein_seq[i:i+k]
        k_mer_dict[tmp]=k_mer_dict[tmp]+1

    return k_mer_dict


def get_alignment(protein_seq_1: str, protein_seq_2: str,
                  gap_penalty: int, substitution_matrix: np.ndarray) -> Tuple[str, str]:

    align1=""
    align2=""
    scoring_matrix = np.zeros((len(protein_seq_1) + 1,len(protein_seq_2)+1))
    traceback_matrix = np.zeros((len(protein_seq_1) + 1, len(protein_seq_2) + 1))
    for i in range(0,len(protein_seq_1)):
        for j in range(0,len(protein_seq_2)):
            gap1 = scoring_matrix[i][j]+gap_penalty
            gap2 = scoring_matrix[i][j]+gap_penalty
            diag = scoring_matrix[i][j]+substitution_matrix[protein_seq_1[i]][protein_seq_2[j]]

            if argmax([0,scoring_matrix[i+1][j],gap1])==2:
                traceback_matrix[i+1][j]=1
            scoring_matrix[i + 1][j] = max([0, scoring_matrix[i + 1][j], gap1])

            if argmax([0,scoring_matrix[i][j+1],gap2])==2:
                traceback_matrix[i][j+1]=2
            scoring_matrix[i][j + 1] = max([0, scoring_matrix[i][j + 1], gap2])

            if argmax([0,scoring_matrix[i+1][j+1],diag]) == 2:
                traceback_matrix[i+1][j+1]=3
            scoring_matrix[i+1][j+1] = max([0,scoring_matrix[i+1][j+1],diag])
    #align
    maximum = argmax(scoring_matrix)
    ind = np.unravel_index(maximum,scoring_matrix.shape)
    print(ind)
    i_current=ind[0]
    j_current=ind[1]
    print(protein_seq_1)
    print(protein_seq_2)
    while scoring_matrix[i_current][j_current]!=0:
        print(scoring_matrix[i_current][j_current])
        trace=traceback_matrix[i_current][j_current]
        if trace==1:
            i_current-=1
            align1=protein_seq_1[i_current]+align1
            align2="-"+align2

        if trace == 2:
            j_current-=1
            align1="-"+align1
            align2=protein_seq_2[j_current]+align2

        if trace==3:
            i_current-=1
            j_current-=1
            align1=protein_seq_1[i_current]+align1
            align2=protein_seq_2[j_current]+align2

        print(align1)
        print(align2)





    return align1, align2


