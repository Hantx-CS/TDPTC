# TDPTC

### Directory Structure
- data/			&emsp;datasets.
- experiment_Fraction/		&emsp;Python File of the experiment for fraction.
- experiment_InnerCount/		&emsp;Python File of the method InnerCount. 
- experiment_Louvain/	  &emsp;Python File of the TDPTC on datasets partitioned by Louvain.
- experiment_Random19/		&emsp;Python File of the TDPTC on datasets partitioned by Random(1:9).
- experiment_Random55/		&emsp;Python File of the TDPTC on datasets partitioned by Random(5:5).
- experiment_Scale/	  &emsp;Python File of the TDPTC on datasets with different scales.
- LICENSE.txt		&emsp;MPL license.
- README.md		&emsp;This file.

### Prepare for IMDB
Download the [IMDB dataset](https://www.cise.ufl.edu/research/sparse/matrices/Pajek/IMDB.html) and place the dataset in data/, and run the following commands:
```
  cd data/
  python3 ReadIMDB.py IMDB.mtx edges.csv deg.csv
  cd ../
```
