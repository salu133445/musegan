# Shell scripts for downloading training data

## Download particular training data

Run

```sh
./download.sh [filename]
```

This will download the training data to the current working directory. Available
training data are listed as follows:

- [`lastfm_alternative_5b_phrase.npy`](https://drive.google.com/uc?export=download&id=1QKKWJ9t7K8nwYRD_AbNQDUie4xSiv1ph)
  contains 12,444 four-bar phrases from 2,074 songs with _alternative_ tags. The
  shape is (2074, 6, 4, 96, 84, 5). The five tracks are _Drums_, _Piano_,
  _Guitar_, _Bass_ and _Strings_.

- [`lastfm_alternative_8b_phrase.npy`](https://drive.google.com/uc?export=download&id=1f9NKbhIxIbedHR370sc_hF9730985Xre)
  contains 13,746 four-bar phrases from 2,291 songs with _alternative_ tags. The
  shape is (2291, 6, 4, 96, 84, 8). The eight tracks are _Drums_, _Piano_,
  _Guitar_, _Bass_, _Ensemble_, _Reed_, _Synth Lead_ and _Synth Pad_呒.

## Download all training data

Run

```sh
./download_all.sh
```

This will download all training data to the current working directory.
