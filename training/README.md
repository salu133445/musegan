Pre-processing for Dataset

== 1. run "tracks_parsing.py" ================

    This file is to parse the crawled data into five tracks: paino, drum, guitar, bass
    and other given a msd_id list (Rock_C_id here). If there are more than two the same
    types of instrument, they will be compressed into one. Additionally, the type "other"
    will also include "string" or any other types of instrument.

    * LMD_dataset => {drum, piano, guitar, bass other}
    * placed in "tracks" folder with intruments name
    * tracks/(instru_name)/(msd_id).npz
        Ex: "tracks/Drum/TRAAEEH128E0795DFE.npz"
    * 2844 selected in 6646

== 2. run "song_analysis.py" =================
    There are several tasks for this file:
    1. Generate Piano Roll for segmentation (in .mat for matlab)
        * "Piano_Roll": (128 x ?) mat matrix

    2. Search Non-Empty Bar and label them
        * "act_all":   list of bool. activation of instruments > thres (3 here)
        * "act_instr": 2d np array in shape (numOfBar, 5). If the bar of epecific track is not
                     empty, it will be denoted to 1.

    All  new folders are under "trakcs" folder

== 3. run "main_seg.m" & "main_lab.m" ==========
    Segmentation and Labeling
    the results are in new folder "structure"

== 4. run "Gen_data_bar.py" =================
    Output 6 tracks: Original(5) + 'act_instr'
    This step file to generate 6 important npy data for NN. The bars are filtered by act_all".








