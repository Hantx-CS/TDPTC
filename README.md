# TDPTC

### Directory Structure
- data/			&emsp;datasets.
  - create_cpp &emsp; C++ File to create different scale of datasets.
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

### Prepare for Scale experiment
Generate a dataset of a specified scale using the following command, which extracts a subset from the original dataset according to the defined scale. This allows for testing the performance of algorithms at different scales.
```
  cd data/create_cpp
  make
  ./CreateDatabase ../edges.csv $scale $eps 1 10 0
  cd ../../
```

### Run Experiments
Copy the dataset files from the `data/` directory to each `experiment_*` folder. Modify the `sourceFile` in the Makefile to match the name of the dataset and set `EPSILON` to the desired value. For the Fraction and Scale experiments, this value can be set in the Makefile. For other experiments, modify `EPSILON` directly in `TriangleCount.py`. Adjust the `SLEEPTIME` appropriately to prevent CPU overload during the 300 cycle experiments, with each cycle consisting of 10 sets, based on the performance capabilities of your device.

For each `experiment_*` folder, the Python files can be executed using the commands `make run`, `make one`, or `make run_IMDB` specified in the Makefile. The default number of runs is set to 300, but this can be adjusted in the Makefile as needed.


Copy `convert.py` and `mean.py` into each `experiment_*` folder. Running `make mean` will calculate the average results from repeated experiments.
