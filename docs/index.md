<img src="figs/logo.png" alt="logo" width="200" height="200" style="margin-left:0 margin-right:0"/>


<p style="color:#222;">
  <em><a class="invisible-link" href="pdf/musegan-aaai2018-slides.pdf">NEWS: The slides for the oral representation at AAAI 2018 have been uploaded.<span style="color:#727272"> &mdash; Feb 17th, 2018</span></a></em><br>
  <em>NEWS: <strong>Please come to find us at AAAI!</strong> (oral presentation at 11:30, Feb 5th at Grand Salon A, Hilton Riverside, New Orleans)<span style="color:#727272"> &mdash; Feb 1st, 2018</span></em><br>
  <em><a class="invisible-link" href="dataset">NEWS: The Lakh Pianoroll Dataset is back!<span style="color:#727272"> &mdash; Feb 1st, 2018</span></a></em><br>
  <em><a class="invisible-link" href="http://arxiv.org/abs/1709.06298">NEWS: A new version of the paper has been uploaded.<span style="color:#727272"> &mdash; Nov 27th, 2017</span></a></em><br>
  <em>NEWS: Our paper has been accepted by <strong>AAAI2018</strong>.<span style="color:#727272"> &mdash; Nov 9th, 2017</span></em>
</p>

[MuseGAN](https://salu133445.github.io/musegan/) is a project on music
generation. In essence, we aim to generate polyphonic music of multiple tracks
(instruments) with harmonic and rhythmic structure, multi-track interdependency
and temporal structure. To our knowledge, our work represents the first approach
that deal with these issues altogether.

Listen to some of the best phrases.
([more results](https://salu133445.github.io/musegan/results))

{% include audio_player.html filename="best_samples.mp3" %}

The models are trained with
[Lakh Pianoroll Dataset](https://salu133445.github.io/lakh-pianoroll-dataset/)
(LPD), a new [multi-track piano-roll](https://salu133445.github.io/musegan/data)
dataset, in an unsupervised approach. The proposed models are able to generate
music either from scratch, or by accompanying a track given by user.
Specifically, we use the model to generate pop song phrases consisting of bass,
drums, guitar, piano and strings tracks.

It's an open source project on [Github](https://github.com/salu133445/musegan).
You can play with it if you're interested.
