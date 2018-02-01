Preprocessing
=============

Source code for deriving [Lakh Pianoroll Dataset](https://salu133445.github.io/musegan/dataset) (LPD) from [Lakh MIDI dataset](http://colinraffel.com/projects/lmd/) (LMD).

- download LMD [here](http://colinraffel.com/projects/lmd/)
- modify `settings.yaml` for settings
- run `cleansing.sh`

By default, it will generate the whole LPD, including lpd_full, lpd_matched, lpd_cleansed, lpd_5_full, lpd_5_matched and lpd_5_cleansed.

If you want to use LPD directly, the one derived with the default settings is available [here](https://salu133445.github.io/musegan/dataset).
