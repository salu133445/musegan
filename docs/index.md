<img src="figs/logo.png" alt="logo" width="200" height="200" />

<p style="color:#222;">
  <em>NEWS: <strong>Please come to find us at AAAI!</strong> (oral presentation at 11:30, Feb 5th at Grand Salon A, Hilton Riverside, New Orleans)<span style="color:#727272"> &mdash; Feb 1th, 2018</span></em><br>
  <em>NEWS: The Lakh Pianoroll Dataset is back!<span style="color:#727272"> &mdash; Feb 1th, 2018</span></em><br>
  <em>NEWS: A new version of the paper has been uploaded.<span style="color:#727272"> &mdash; Nov 27th, 2017</span></em><br>
  <em>NEWS: Our paper has been accepted by <strong>AAAI2018</strong>.<span style="color:#727272"> &mdash; Nov 9th, 2017</span></em>
</p>

MuseGAN is a project on music generation.
In essence, we aim to generate *multi-track polyphonic* music with harmonic and rhythmic structure, multi-track interdependency and temporal structure.
To our knowledge, our work represents the first approach that deal with these issues altogether.

The models are trained with a large collection of [multi-track piano-rolls](#multitrack) in an unsupervised approach.
The proposed models are able to generate music either from scratch, or by accompanying a track given by user.
Specifically, we use the model to generate pop song phrases consisting of bass, drums, guitar, piano and strings tracks.

Listen to some of the best phrases.

{% include audio_player.html filename="best_samples.mp3" %}

It's an open source project.
All the source code is available at [Github](https://github.com/salu133445/musegan).
You can play with it if you're interested.
