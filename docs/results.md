# Results

## Pianoroll Visualizations

<img src="figs/evolution.png" alt="evolution" style="max-width:none;">
<p class="caption">Evolution of the generated pianorolls as a function of update steps</p>

<img src="figs/hybrid.png" alt="hybrid" style="max-width:none;">
<p class="caption">Randomly-chosen generated pianorolls (generated from scratch)</p>

## Audio Samples

> __Note that all the audio samples presented below have been downsampled to 4
time steps per beat (originally 24 time steps per beat).__

### Best samples

{% include audio_player.html filename="best_samples.mp3" %}

### Generation from scratch

> No cherry-picking. Some might sound unpleasant. __Lower the volume first!__

| Model    | Sample                                                               |
|:--------:|:--------------------------------------------------------------------:|
| composer | {% include audio_player.html filename="from_scratch_composer.mp3" %} |
| jamming  | {% include audio_player.html filename="from_scratch_jamming.mp3" %}  |
| hybrid   | {% include audio_player.html filename="from_scratch_hybrid.mp3" %}   |

### Track-conditional generation

> No cherry-picking. Some might sound unpleasant. __Lower the volume first!__

| Model    | Sample                                                                    |
|:--------:|:-------------------------------------------------------------------------:|
| composer | {% include audio_player.html filename="track_conditional_composer.mp3" %} |
| jamming  | {% include audio_player.html filename="track_conditional_jamming.mp3" %}  |
| hybrid   | {% include audio_player.html filename="track_conditional_hybrid.mp3" %}   |
