# Shell scripts for downloading pretrained model

## Download a particular pretrained model

Run

```sh
sh download.sh [model] [filename]
```

This will download a particular pretrained model to the current working
directory. Available files are listed as follows:

- MuseGAN (`musegan`)
  - `lastfm_alternative_g_hybrid_d_proposed.tar.gz`

- BinaryMuseGAN (`bmusegan`)
  - `lastfm_alternative_first_stage_d_proposed.tar.gz`
  - `lastfm_alternative_first_stage_d_abalted.tar.gz`
  - `lastfm_alternative_first_stage_d_baseline.tar.gz`

## Download all pretrained models

Run

```sh
sh download_all.sh
```

This will download all pretrained models to the current working directory.
