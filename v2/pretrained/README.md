# Shell scripts for downloading pretrained model

## Download a particular pretrained model

Run

```sh
./download.sh [model] [filename]
```

This will download a particular pretrained model to the current working
directory. Available files are listed as follows:

- MuseGAN (`musegan`)
  - [`lastfm_alternative_g_composer_d_proposed.tar.gz`](https://drive.google.com/uc?export=download&id=1-KgUFsVvRXSWtpiB5zrRjTvFnss-6E5T)
  - [`lastfm_alternative_g_jamming_d_proposed.tar.gz`](https://drive.google.com/uc?export=download&id=1D0vwE-DaPafRd5HM849Qbs3VmqSZI5I3)
  - [`lastfm_alternative_g_hybrid_d_proposed.tar.gz`](https://drive.google.com/uc?export=download&id=1mFG3LwazgQ3YgoXoEm_blta8p5Gf2LaR)

- BinaryMuseGAN (`bmusegan`)
  - [`lastfm_alternative_first_stage_d_proposed.tar.gz`](https://drive.google.com/uc?export=download&id=16LKWjiEjDjgiTjMLFcgnzZdCT-v8fp3T)
  - [`lastfm_alternative_first_stage_d_ablated.tar.gz`](https://drive.google.com/uc?export=download&id=1YyKAiPV0AuGuQB1K05dQAkPnsRMAqjtJ)
  - [`lastfm_alternative_first_stage_d_baseline.tar.gz`](https://drive.google.com/uc?export=download&id=1ZVASqhTApVWSvtM0N-952BAEbfUqRTfK)

## Download all pretrained models

Run

```sh
./download_all.sh
```

This will download all pretrained models to the current working directory.
