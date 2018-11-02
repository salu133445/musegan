"""Class and utilities for metrics
"""
import os
import warnings
import numpy as np

def get_tonal_matrix(r1=1.0, r2=1.0, r3=0.5):
    """Compute and return a tonal matrix for computing the tonal distance [1].
    Default argument values are set as suggested by the paper.

    [1] Christopher Harte, Mark Sandler, and Martin Gasser. Detecting harmonic
    change in musical audio. In Proc. ACM MM Workshop on Audio and Music
    Computing Multimedia, 2006.
    """
    tonal_matrix = np.empty((6, 12))
    tonal_matrix[0] = r1 * np.sin(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[1] = r1 * np.cos(np.arange(12) * (7. / 6.) * np.pi)
    tonal_matrix[2] = r2 * np.sin(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[3] = r2 * np.cos(np.arange(12) * (3. / 2.) * np.pi)
    tonal_matrix[4] = r3 * np.sin(np.arange(12) * (2. / 3.) * np.pi)
    tonal_matrix[5] = r3 * np.cos(np.arange(12) * (2. / 3.) * np.pi)
    return tonal_matrix

def get_num_pitch_used(pianoroll):
    """Return the number of unique pitches used in a piano-roll."""
    return np.sum(np.sum(pianoroll, 0) > 0)

def get_qualified_note_rate(pianoroll, threshold=2):
    """Return the ratio of the number of the qualified notes (notes longer than
    `threshold` (in time step)) to the total number of notes in a piano-roll."""
    padded = np.pad(pianoroll.astype(int), ((1, 1), (0, 0)), 'constant')
    diff = np.diff(padded, axis=0)
    flattened = diff.T.reshape(-1,)
    onsets = (flattened > 0).nonzero()[0]
    offsets = (flattened < 0).nonzero()[0]
    num_qualified_note = (offsets - onsets >= threshold).sum()
    return num_qualified_note / len(onsets)

def get_polyphonic_ratio(pianoroll, threshold=2):
    """Return the ratio of the number of time steps where the number of pitches
    being played is larger than `threshold` to the total number of time steps"""
    return np.sum(np.sum(pianoroll, 1) >= threshold) / pianoroll.shape[0]

def get_in_scale(chroma, scale_mask=None):
    """Return the ratio of chroma."""
    measure_chroma = np.sum(chroma, axis=0)
    in_scale = np.sum(np.multiply(measure_chroma, scale_mask, dtype=float))
    return in_scale / np.sum(chroma)

def get_drum_pattern(measure, drum_filter):
    """Return the drum_pattern metric value."""
    padded = np.pad(measure, ((1, 0), (0, 0)), 'constant')
    measure = np.diff(padded, axis=0)
    measure[measure < 0] = 0

    max_score = 0
    for i in range(6):
        cdf = np.roll(drum_filter, i)
        score = np.sum(np.multiply(cdf, np.sum(measure, 1)))
        if score > max_score:
            max_score = score

    return  max_score / np.sum(measure)

def get_harmonicity(bar_chroma1, bar_chroma2, resolution, tonal_matrix=None):
    """Return the harmonicity metric value"""
    if tonal_matrix is None:
        tonal_matrix = get_tonal_matrix()
        warnings.warn("`tonal matrix` not specified. Use default tonal matrix",
                      RuntimeWarning)
    score_list = []
    for r in range(bar_chroma1.shape[0]//resolution):
        start = r * resolution
        end = (r + 1) * resolution
        beat_chroma1 = np.sum(bar_chroma1[start:end], 0)
        beat_chroma2 = np.sum(bar_chroma2[start:end], 0)
        score_list.append(tonal_dist(beat_chroma1, beat_chroma2, tonal_matrix))
    return np.mean(score_list)

def to_chroma(pianoroll):
    """Return the chroma features (not normalized)."""
    padded = np.pad(pianoroll, ((0, 0), (0, 12 - pianoroll.shape[1] % 12)),
                    'constant')
    return np.sum(np.reshape(padded, (pianoroll.shape[0], 12, -1)), 2)

def tonal_dist(chroma1, chroma2, tonal_matrix=None):
    """Return the tonal distance between two chroma features."""
    if tonal_matrix is None:
        tonal_matrix = get_tonal_matrix()
        warnings.warn("`tonal matrix` not specified. Use default tonal matrix",
                      RuntimeWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        chroma1 = chroma1 / np.sum(chroma1)
        chroma2 = chroma2 / np.sum(chroma2)
    result1 = np.matmul(tonal_matrix, chroma1)
    result2 = np.matmul(tonal_matrix, chroma2)
    return np.linalg.norm(result1 - result2)

def plot_histogram(hist, fig_dir=None, title=None, max_hist_num=None):
    """Plot the histograms of the statistics"""
    import matplotlib.pyplot as plt
    hist = hist[~np.isnan(hist)]
    u_value = np.unique(hist)

    hist_num = len(u_value)
    if max_hist_num is not None:
        if len(u_value) > max_hist_num:
            hist_num = max_hist_num

    fig = plt.figure()
    plt.hist(hist, hist_num)
    if title is not None:
        plt.title(title)
    if fig_dir is not None and title is not None:
        fig.savefig(os.path.join(fig_dir, title))
    plt.close(fig)

class Metrics(object):
    """Class for metrics.
    """
    def __init__(self, config):
        self.metric_map = config['metric_map']
        self.tonal_distance_pairs = config['tonal_distance_pairs']
        self.track_names = config['track_names']
        self.beat_resolution = config['beat_resolution']
        self.drum_filter = config['drum_filter']
        self.scale_mask = config['scale_mask']
        self.tonal_matrix = get_tonal_matrix(
            config['tonal_matrix_coefficient'][0],
            config['tonal_matrix_coefficient'][1],
            config['tonal_matrix_coefficient'][2]
        )

        self.metric_names = [
            'empty_bar',
            'pitch_used',
            'qualified_note',
            'polyphonicity',
            'in_scale',
            'drum_pattern',
            'chroma_used',
        ]

    def print_metrics_mat(self, metrics_mat):
        """Print the intratrack metrics as a nice formatting table"""
        print(' ' * 12, ' '.join(['{:^14}'.format(metric_name)
                                  for metric_name in self.metric_names]))

        for t, track_name in enumerate(self.track_names):
            value_str = []
            for m in range(len(self.metric_names)):
                if np.isnan(metrics_mat[m, t]):
                    value_str.append('{:14}'.format(''))
                else:
                    value_str.append('{:^14}'.format('{:6.4f}'.format(
                        metrics_mat[m, t])))

            print('{:12}'.format(track_name), ' '.join(value_str))

    def print_metrics_pair(self, pair_matrix):
        """Print the intertrack metrics as a nice formatting table"""
        for idx, pair in enumerate(self.tonal_distance_pairs):
            print("{:12} {:12} {:12.5f}".format(
                self.track_names[pair[0]], self.track_names[pair[1]],
                pair_matrix[idx]))

    def eval(self, bars, verbose=False, mat_path=None, fig_dir=None):
        """Evaluate the input bars with the metrics"""
        score_matrix = np.empty((len(self.metric_names), len(self.track_names),
                                 bars.shape[0]))
        score_matrix.fill(np.nan)
        score_pair_matrix = np.zeros((len(self.tonal_distance_pairs),
                                      bars.shape[0]))
        score_pair_matrix.fill(np.nan)

        for b in range(bars.shape[0]):
            for t in range(len(self.track_names)):
                is_empty_bar = ~np.any(bars[b, ..., t])
                if self.metric_map[0, t]:
                    score_matrix[0, t, b] = is_empty_bar
                if is_empty_bar:
                    continue
                if self.metric_map[1, t]:
                    score_matrix[1, t, b] = get_num_pitch_used(bars[b, ..., t])
                if self.metric_map[2, t]:
                    score_matrix[2, t, b] = get_qualified_note_rate(
                        bars[b, ..., t])
                if self.metric_map[3, t]:
                    score_matrix[3, t, b] = get_polyphonic_ratio(
                        bars[b, ..., t])
                if self.metric_map[4, t]:
                    score_matrix[4, t, b] = get_in_scale(
                        to_chroma(bars[b, ..., t]), self.scale_mask)
                if self.metric_map[5, t]:
                    score_matrix[5, t, b] = get_drum_pattern(bars[b, ..., t],
                                                             self.drum_filter)
                if self.metric_map[6, t]:
                    score_matrix[6, t, b] = get_num_pitch_used(
                        to_chroma(bars[b, ..., t]))

            for p, pair in enumerate(self.tonal_distance_pairs):
                score_pair_matrix[p, b] = get_harmonicity(
                    to_chroma(bars[b, ..., pair[0]]),
                    to_chroma(bars[b, ..., pair[1]]), self.beat_resolution,
                    self.tonal_matrix)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            score_matrix_mean = np.nanmean(score_matrix, axis=2)
            score_pair_matrix_mean = np.nanmean(score_pair_matrix, axis=1)

        if verbose:
            print("{:=^120}".format(' Evaluation '))
            print('Data Size:', bars.shape)
            print("{:-^120}".format('Intratrack Evaluation'))
            self.print_metrics_mat(score_matrix_mean)
            print("{:-^120}".format('Intertrack Evaluation'))
            self.print_metrics_pair(score_pair_matrix_mean)

        if fig_dir is not None:
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            if verbose:
                print('[*] Plotting...')
            for m, metric_name in enumerate(self.metric_names):
                for t, track_name in enumerate(self.track_names):
                    if self.metric_map[m, t]:
                        temp = '-'.join(track_name.replace('.', ' ').split())
                        title = '_'.join([metric_name, temp])
                        plot_histogram(score_matrix[m, t], fig_dir=fig_dir,
                                       title=title, max_hist_num=20)
            if verbose:
                print("Successfully saved to", fig_dir)

        if mat_path is not None:
            if not mat_path.endswith(".npy"):
                mat_path = mat_path + '.npy'
            info_dict = {
                'score_matrix_mean': score_matrix_mean,
                'score_pair_matrix_mean': score_pair_matrix_mean}
            if verbose:
                print('[*] Saving score matrices...')
            np.save(mat_path, info_dict)
            if verbose:
                print("Successfully saved to", mat_path)

        return score_matrix_mean, score_pair_matrix_mean

def eval_dataset(filepath, result_dir, location, config):
    """Run evaluation on a dataset stored in either shared array (if `location`
    is 'sa') or in hard disk (if `location` is 'hd') and save the results to the
    given directory.

    """
    def load_data(filepath, location):
        """Load and return the training data."""
        print('[*] Loading data...')

        # Load data from SharedArray
        if location == 'sa':
            import SharedArray as sa
            data = sa.attach(filepath)

        # Load data from hard disk
        elif location == 'hd':
            if os.path.isabs(filepath):
                data = np.load(filepath)
            else:
                root = os.path.dirname(os.path.dirname(
                    os.path.realpath(__file__)))
                data = np.load(os.path.abspath(os.path.join(
                    root, 'training_data', filepath)))

        else:
            raise ValueError("Unrecognized value for `location`")

        # Reshape data
        data = data.reshape(-1, config['num_timestep'], config['num_pitch'],
                            config['num_track'])

        return data

    print('[*] Loading data...')
    data = load_data(filepath, location)

    print('[*] Running evaluation')
    metrics = Metrics(config)
    _ = metrics.eval(data, verbose=True,
                     mat_path=os.path.join(result_dir, 'score_matrices.npy'),
                     fig_dir=result_dir)

def print_mat_file(mat_path, config):
    """Print the score matrices stored in a file."""
    metrics = Metrics(config)
    with np.load(mat_path) as loaded:
        metrics.print_metrics_mat(loaded['score_matrix_mean'])
        metrics.print_metrics_pair(loaded['score_pair_matrix_mean'])
