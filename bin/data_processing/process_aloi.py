
import sys
import numpy as np

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    dim=128
    id = 0
    with open(input_file,'r') as fin:
        with open(output_file,'w') as fout:
            for line in fin:
                splt = line.strip().split(" ")
                vec = np.zeros(dim)
                # write id
                fout.write(str(id))
                fout.write("\t")
                fout.write(str(splt[0]))
                for i in range(1,len(splt)):
                    if len(splt[i]) > 0:
                        dim_val = splt[i].split(":")
                        if len(dim_val) != 2:
                            print(dim_val)
                        vec[int(dim_val[0])-1] = float(dim_val[1])
                for i in range(0,len(vec)):
                    fout.write("\t")
                    fout.write(str(vec[i]))
                fout.write("\n")
                id += 1
