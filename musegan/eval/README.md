## Evaluation on MuseGAN

### Available Metrics
- metric_is_empty_bar (**EB**)
- metric_num_pitch_used
- metric_qualified_note_ratio (**QN**)
- metric_polyphonic_ratio
- metric_in_scale
- metric_drum_pattern (**DP**)
- metric_num_chroma_used (**UPC**)
- metrics_harmonicity (or tonal distance, **TD**)

### Usage

new a metrics object
```python
m = Metrics()
```

evaluate a dataset
```python
m.eval(dataset ,quiet=True,  save_fig=True, fig_dir='data_bar/tra/statistic')
```



