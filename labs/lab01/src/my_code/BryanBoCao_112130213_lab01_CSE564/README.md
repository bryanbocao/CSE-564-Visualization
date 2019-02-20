# CSE 564 Visualization Lab 1

Student: Bryan Bo Cao

SBU ID: 112130213

Email: bo.cao.1@stonybrook.edu or boccao@stonybrook.edu

# Demo

This code is live on https://blockbuilder.org/BryanBo-Cao/1e2f511ee3bbdae742777dc73cf7a441

PLEASE switch to side-by-side mode to see the whole functionality of this code instead of using fullscreen.

# File Structure

All files include
```
index.html
College.csv
compute_min_spanning_tree.py
minimum_spanning_tree_mtx.json
```
where 1)```index.html``` is the main file to run and all the d3 code is in this file; 2) ```College.csv``` is the dataset downloaded from the ```college``` dataset from https://vincentarelbundock.github.io/Rdatasets/datasets.html, the original dataset has College 777 data points, 18 dimensions; 3) ```compute_min_spanning_tree.py``` is the python code to compute the minimum spanning tree for the first 70 data points and save the data as 4) ```minimum_spanning_tree_mtx.json```, then ```index.html``` visualize it in force-directed layout graph. Note that in the force-directed layout graph, the distances between two nodes are the euclidean distance using all the attribute.

