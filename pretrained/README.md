# Shell scripts for downloading pretrained model

## Download a particular pretrained model

Run

```sh
./download.sh [model] [filename]
```

This will download a particular pretrained model to the current working
directory. Available files are listed as follows:

- MuseGAN (`musegan`)
  - [`lastfm_alternative_g_composer_d_proposed.tar.gz`](https://drive.google.com/uc?export=download&id=1QzTL4So-oRWrif4gVKqQM5yQ48y2X5gM)
  - [`lastfm_alternative_g_jamming_d_proposed.tar.gz`](https://drive.google.com/uc?export=download&id=1-Q_krj4VKOWbpFU1jTdfihKYxV6hKmlm)
  - [`lastfm_alternative_g_hybrid_d_proposed.tar.gz`](https://drive.google.com/uc?export=download&id=1b1bwTzW09QPFbRn2Hy9X8yU1fbTc3S1k)

- BinaryMuseGAN (`bmusegan`)
  - [`lastfm_alternative_first_stage_d_proposed.tar.gz`](https://drive.google.com/uc?export=download&id=12tEzs-Qa-qi59hLJB8TlD-vcZgVEQZu6)
  - [`lastfm_alternative_first_stage_d_abalted.tar.gz`](https://drive.google.com/uc?export=download&id=1GolkoE2ktmHF2Pt7POd8TBBYZARu6ih8)
  - [`lastfm_alternative_first_stage_d_baseline.tar.gz`](https://drive.google.com/uc?export=download&id=1qWWWU6UTMJvzdK6y4bvh3PRXF5Xbk09v)

## Download all pretrained models

Run

```sh
./download_all.sh
```

This will download all pretrained models to the current working directory.
