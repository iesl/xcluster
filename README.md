# xcluster #
**xcluster** contains algorithms and evaluation tools for _extreme clustering_, i.e., instances of clustering in which the number of points to be clustered _and_ the number of clusters is large.  Most notably, xcluster contains an implementation of **PERCH** (Purity Enhancing Rotations for Cluster Hierachies). PERCH is an online extreme clustering algorithm that incrementally builds a tree with data points at its leaves.  During the data point insertion procedure, PERCH performs _rotations_ to keep the tree accurate and as balanced as possible. Empirical experiments show that PERCH produces purer trees faster than other algorithms; theoretical analysis shows that for _separable_ data, PERCH builds trees with perfect dendrogram purity regardless of the order of the data.  Technical details and analysis of the algorithm can be found in our paper: [A Hierarchical Algorithm for Extreme Clustering](https://dl.acm.org/authorize?N34117)


## Setup ##

If running the python code, download and Install Anaconda's Python3

```
https://docs.continuum.io/anaconda/install
```

If running python code, install numba

```
conda install numba
```

Set environment variables:

```
source bin/setup.sh
```

Install maven if you don't already have it installed:

```
./bin/util/install_mvn.sh
```

Build Scala code:

```
./bin/build.sh
```

Download data

```
./bin/download_data.sh
```

## Run ##

#### Scala ####

Run Test on Separated Data:

```
 ./bin/test/test_perch_dendrogram_purity.sh
```

Run PERCH on Small Scale Data (glass dataset):

```
# Hierarchical clustering
./bin/hierarchical/glass/run_perch.sh

# Flat clustering
./bin/flat/glass/run_perch.sh
```

Run PERCH on ALOI (see notes below for suggested system environment):

```
# Hierarchical clustering
./bin/hierarchical/aloi/run_perch.sh

# Flat clustering
./bin/flat/aloi/run_perch.sh
```

#### Python ####

Run Test on Separated Data:

```
 ./bin/test/test_perch_dendrogram_purity_py.sh
```

Run PERCH on Small Scale Data (glass dataset):

```
# Hierarchical clustering
./bin/hierarchical/glass/run_perch_py.sh
```

Run PERCH on ALOI:

```
# Hierarchical clustering
./bin/hierarchical/aloi/run_perch_py.sh
```

## Notes ##

  - The ALOI scripts are set up to run on a machine with about 24 cores and 60GB of memory. Most of the computation required is to compute Dendrogram Purity. You can run the Perch algorithm with much less computational resources efficiently (even 1 thread and a few gigabytes of memory.)
  - You'll need perl installed on your system to run experiment shell scripts as is. perl is used to shuffle the data. If you can't run perl, you can change this to another shuffling method of your choice.
  - The scripts in this project use environment variables set in the setup script. You'll need to source this set up script in each shell session running this project.
  - Java Version 1.8 and Scala 2.11.7 are used in this project. Java 1.8 must be installed on your system. It is not necessary to have Scala installed.
