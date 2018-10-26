# Shell scripts for downloading training data

## Download particular training data

Run

```sh
./download.sh [filename]
```

This will download the training data to the current working directory. Available
training data are listed as follows:

- [`lastfm_alternative_5b_phrase.npy`](https://drive.google.com/uc?export=download&id=1F7J5n9uOPqViBYpoPT5GvE4PjCWhOyWc) 
  contains 12,444 four-bar phrases from 2,074 songs with *alternative* tags. The
  shape is (2074, 6, 4, 96, 84, 5). The five tracks are *Drums*, *Piano*,
  *Guitar*, *Bass* and *Strings*.

- [`lastfm_alternative_8b_phrase.npy`](https://drive.google.com/uc?export=download&id=1x3CeSqE6ElWa6V7ueNl8FKPFmMoyu4ED)
  contains 13,746 four-bar phrases from 2,291 songs with *alternative* tags. The
  shape is (2291, 6, 4, 96, 84, 8). The eight tracks are *Drums*, *Piano*,
  *Guitar*, *Bass*, *Ensemble*, *Reed*, *Synth Lead* and *Synth Pad*.

## Download all training data

Run

```sh
./download_all.sh
```

This will download all training data to the current working directory.
