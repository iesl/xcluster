import numpy as np
import sys

def load_result_file(fn):
    """
    Result file format:
        algorithm <tab> dataset <tab> dendrogram purity

    Args:
        fn: filename

    Returns: dictionary: alg -> dataset -> (mean(dp),std(dp)

    """
    alg2dataset2score = {}
    with open(fn,'r') as fin:
        for line in fin:
            try:
                splt = line.strip().split("\t")
                alg,dataset,dp = splt
                if alg not in alg2dataset2score:
                    alg2dataset2score[alg] = {}
                if dataset not in alg2dataset2score[alg]:
                    alg2dataset2score[alg][dataset] = []
                alg2dataset2score[alg][dataset].append(float(dp))
            except:
                pass

    for alg in alg2dataset2score:
        for dataset in alg2dataset2score[alg]:
            mean = np.mean(alg2dataset2score[alg][dataset])
            std = np.std(alg2dataset2score[alg][dataset])
            alg2dataset2score[alg][dataset] = (mean,std)

    return alg2dataset2score

def escape_latex(s):
    s = s.replace("_","\\_")
    return s

def latex_table(alg2dataset2score):
    table_string = """\\begin{table}\n\\begin{center}\n\\begin{tabular}"""
    num_ds = max([len(alg2dataset2score[x]) for x in alg2dataset2score])
    formatting = "{|c" + "|c" * num_ds + "|" + "}"
    table_string += format(formatting)
    table_string += "\n\\hline\n"
    ds_names = list(set([name for x in alg2dataset2score for name in alg2dataset2score[x]]))
    table_string += "\\bf Algorithm & \\bf " + " & \\bf ".join([escape_latex(x) for x in ds_names]) + "\\\\\n"
    table_string += "\\hline\n"
    alg_names = alg2dataset2score.keys()
    alg_names = sorted(alg_names)
    for alg in alg_names:
        scores = [ "%.5f $\\pm$ %.5f" % (alg2dataset2score[alg][ds][0],alg2dataset2score[alg][ds][1]) if ds in alg2dataset2score[alg] else "-" for ds in ds_names]
        table_string += "%s & %s \\\\\n" % (alg," & ".join(scores))
    table_string += "\\hline\n\\end{tabular}\n\\end{center}\n\\end{table}"
    return table_string


if __name__ == "__main__":
    print(latex_table(load_result_file(sys.argv[1])))