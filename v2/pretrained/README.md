# Shell scripts for downloading pretrained model

## Download a particular pretrained model

Run

```sh
./download.sh [model] [filename]
```

This will download a particular pretrained model to the current working
directory. Available files are listed as follows:

- MuseGAN (`musegan`)
  - [`lastfm_alternative_g_composer_d_proposed.tar.gz`](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/h3dong_ucsd_edu/Efh0S5J_HUZDq18MO3Kh7RkBALExTJHI7t3Y0ws0Qezc2Q?e=TD1E2E)
  - [`lastfm_alternative_g_jamming_d_proposed.tar.gz`](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/h3dong_ucsd_edu/Ee4tiLWzkYBAqdd4KHqQu3EBBYJuDIbJfVJBKBm4Wq2kjQ?e=Zd6ozV)
  - [`lastfm_alternative_g_hybrid_d_proposed.tar.gz`](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/h3dong_ucsd_edu/Ee7d7AJDyWxFlIehzRCQm7QBwemWcMU6ANPgpwydCdIg1A?e=USjAJr)

- BinaryMuseGAN (`bmusegan`)
  - [`lastfm_alternative_first_stage_d_proposed.tar.gz`](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/h3dong_ucsd_edu/ER1uybyEUtVOllTWBBbJ9XEBaUJsaNXQU7vAh5wHy57eHw?e=KxxtKS)
  - [`lastfm_alternative_first_stage_d_ablated.tar.gz`](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/h3dong_ucsd_edu/EfEkEu02cjhFvJCTOJLcxogBcDm-0fAg5TRsEbVrWXkmSQ?e=WmpJit)
  - [`lastfm_alternative_first_stage_d_baseline.tar.gz`](https://ucsdcloud-my.sharepoint.com/:u:/g/personal/h3dong_ucsd_edu/EdwIT2Cg2EdKpeq70GZgYikB49T7d0_FH_BsGcTbgGZn8g?e=whJXnw)

## Download all pretrained models

Run

```sh
./download_all.sh
```

This will download all pretrained models to the current working directory.
