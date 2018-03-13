# Preparing for Training Data

Following the intructions to build "*.npy" files for training.
When selecting data, we employ some tricky technichs and the details are elobrated in each steps.
Note that theses pre-processing codes are for *refernece only*.
We didn't re-orgnaize them so it's a little messy.
However, we think let the codes opened can help researchers to get familiar with the field of music generation more quickly.



### Step 1. run "tracks_parsing.py"

Given a *msd_id* list (Rock_C_id here), This file can parse the crawled data into five tracks: paino, drum, guitar, bass and other. If there are more than two the same types of instrument, they will be compressed into one. Additionally, the type "other" includes "string" or any other types of instrument.

* LMD_dataset => {drum, piano, guitar, bass other}
* placed in "tracks" folder with intruments name
* tracks/(instru_name)/(msd_id).npz
    Ex: "tracks/Drum/TRAAEEH128E0795DFE.npz"
* 2844 selected in 6646

### Step 2. run "song_analysis.py"
There are several tasks in this file:
* Generate Piano Roll for segmentation (in .mat for matlab)
    * "Piano_Roll": (128 x ?) mat matrix

* Search Non-Empty Bar and label them
    * "act_all":   list of bool. activation of instruments > thres (3 here)
    * "act_instr": 2d np array in shape (numOfBar, 5). If the bar of epecific track is not empty, it will be denoted to 1.

All new created folders are placed in the *"trakcs"* folder

### Step 3. run "main_seg.m" & "main_lab.m"
**Segmentation** and **Labeling**. The algorithm we used is "Structural Feature" [1] [2]. The algorithm is originally working on raw audio. However, we found it also works well on symbolic data.


The codes are modified from my old project (in matlab):
https://github.com/wayne391/Music-Structure-Analysis-in-Matlab <br/>
For python, you can use [MSAF](https://github.com/urinieto/msaf).

the results are placed in the *"structure"* folder


### Step 4. run "Gen_data_bar.py"
Output 6 tracks: Original(5) + 'act_instr'
This step file to generate 6 important npy data for NN. The bars are filtered by act_all".



### References
[1] Unsupervised Detection of Music Boundaries by Time Series Structure Features.

[2] Unsupervised Music Structure Annotation by Time Series Structure Features and Segment Similarity.






