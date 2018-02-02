from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

## 193 settings
import warnings
import matplotlib
import SharedArray as sa
import tensorflow as tf

## notebook
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import signal
import librosa
import os
import time

class Metrics(object):
    def __init__(self,
                 eval_map,
                 inter_pair,
                 drum_filter,
                 scale_mask,
                 track_names,
                 is_build_graph=False):

        # Basic Metrics Setting
        self.eval_map = eval_map
        self.inter_pair = inter_pair
        self.drum_filter = drum_filter
        self.scale_mask = scale_mask
        self.track_names = track_names
        self.tonal_matrix = self.get_tonal_matrix()

        # metrics
        self.metric_names = ['metric_is_empty_bar',
                             'metric_num_pitch_used',
                             'metric_qualified_note_ratio',
                             'metric_polyphonic_ratio',
                             'metric_in_scale',
                             'metric_drum_pattern',
                             'metric_num_chroma_used']

        # amount info
        self.metrics_num = len(self.metric_names)
        self.track_num = len(self.track_names)
        self.pair_num = len(self.inter_pair)

        # training history
        self.score_matrics_history = []
        self.score_parirs_history = []
        self.loss_history = []

    ### Utils ###

    def to_chroma(self, bar, is_normalize=True):
        chroma = bar.reshape(bar.shape[0], 12, -1).sum(axis=2)
        return chroma

    def get_tonal_matrix(self, r1=1.0, r2=1.0, r3=0.5):
        tm = np.empty((6, 12), dtype=np.float32)
        tm[0, :] = r1*np.sin(np.arange(12)*(7./6.)*np.pi)
        tm[1, :] = r1*np.cos(np.arange(12)*(7./6.)*np.pi)
        tm[2, :] = r2*np.sin(np.arange(12)*(3./2.)*np.pi)
        tm[3, :] = r2*np.cos(np.arange(12)*(3./2.)*np.pi)
        tm[4, :] = r3*np.sin(np.arange(12)*(2./3.)*np.pi)
        tm[5, :] = r3*np.cos(np.arange(12)*(2./3.)*np.pi)
        return tm

    ### Recording ###

    def tonal_dist(self, beat_chroma1, beat_chroma2):
        beat_chroma1 = beat_chroma1 / np.sum(beat_chroma1)
        c1 = np.matmul(self.tonal_matrix, beat_chroma1)
        beat_chroma2 = beat_chroma2 / np.sum(beat_chroma2)
        c2 = np.matmul(self.tonal_matrix, beat_chroma2)
        return np.linalg.norm(c1-c2)

    def plot_histogram(self, hist, fig_dir, title='', max_hist_num=20):

        hist = hist[~np.isnan(hist)]
        u_value = np.unique(hist)
        hist_num = len(u_value)
        if len(u_value) > 20:
            hist_num = 20

        fig = plt.figure()
        plt.hist(hist, hist_num)
        plt.title(title)
        fig.savefig(os.path.join(fig_dir, title))
        plt.close(fig)

    def print_metrics_mat(self, metrics_mat):
        print('{:=^90}'.format(' Eval: Intra '))
        title_str = '{:36}'.format('')
        for n in self.track_names:
            title_str += '{:12}'.format(n)
        print(title_str)
        for i in range(metrics_mat.shape[0]):
            row_str = '{:30}'.format(self.metric_names[i])
            for j in range(metrics_mat.shape[1]):
                tmp = metrics_mat[i, j]
                if np.isnan(tmp):
                    row_str += '{:12}'.format('')
                else:
                    row_str += '{:12.5f}'.format(tmp)
            print(row_str)

    def print_metrics_pair(self, pair):

        print('{:=^90}'.format(' Eval: Inter '))
        for pidx in range(len(self.inter_pair)):
            p = self.inter_pair[pidx]
            str1 = self.track_names[p[0]]
            str2 = self.track_names[p[1]]
            row_str = '{:8}'.format(str1) + '{:8}'.format(str2) + '{:12.5f}'.format(pair[pidx])
            print(row_str)

    def collect(self, matrix, pair):
        self.score_matrics_history.append(matrix)
        self.score_parirs_history.append(pair)

    def collect_loss(self, loss):
        self.loss_history.append(loss)

    def save_history(self, log_dir):
        np.save(os.path.join(log_dir, 'metrics_matrix.npy'), self.score_matrics_history)
        np.save(os.path.join(log_dir, 'metrics_pair.npy'), self.score_parirs_history)
        np.save(os.path.join(log_dir, 'loss.npy'), self.loss_history)

    ### Metrics ###

    def metric_is_empty_bar(self, bar):
        return not np.sum(bar)

    def metric_num_pitch_used(self, bar):
        activation_span = np.sum(bar, axis=0)
        return np.sum(activation_span > 0)

    def metric_qualified_note_ratio(self, bar, threshold=2):
        span = bar.shape[1]
        bar_diff = np.concatenate((np.zeros((1, span)), bar, np.zeros((1, span))))
        bar_search = np.diff(bar_diff, axis=0)

        num_notes = 0
        num_short_notes = 0

        for p in range(bar.shape[1]):
            st_idx = (bar_search[:,p] > 0).nonzero()[0]
            ed_idx = (bar_search[:,p] < 0).nonzero()[0]
            for idx in range(len(st_idx)):
                tmp_len = ed_idx[idx] - st_idx[idx]
                if(tmp_len >=  threshold):
                    num_short_notes += 1
                num_notes += 1

        return num_short_notes / num_notes

    def metric_polyphonic_ratio(self, bar, threshold=2):
        return sum(np.sum(bar, axis=1) >= threshold) / 96

    def metric_in_scale(self, chroma):
        all_notes = np.sum(chroma)
        bar_chroma = np.sum(chroma, axis=0)
        in_scale_notes = np.sum(np.multiply(bar_chroma, self.scale_mask))
        return in_scale_notes / all_notes

    def metric_drum_pattern(self, bar):
        span = bar.shape[1]
        bar_diff = np.concatenate((np.zeros((1, span)), bar))
        bar = np.diff(bar_diff, axis=0)
        bar[bar<0] = 0

        temporal = np.sum(bar, axis=1)
        all_notes = np.sum(bar)
        max_score = 0
        for i in range(6):
            cdf = np.roll(self.drum_filter, i)
            score = np.sum(np.multiply(cdf, temporal))
            if score > max_score:
                max_score = score

        return  max_score / all_notes

    def metrics_harmonicity(self, chroma1, chroma2, resolution=24):
        score_list = []
        for r in range(chroma1.shape[0]//resolution):
             chr1 = np.sum(chroma1[resolution*r: resolution*(r+1)], axis=0)
             chr2 = np.sum(chroma2[resolution*r: resolution*(r+1)], axis=0)
             score_list.append(self.tonal_dist(chr1, chr2))
        return np.mean(score_list)


    ### Eval ###

    def eval(self, batch, output_type=0, quiet=False, save_fig=False, fig_dir='./'):
        """
        Evaluate one batch of bars according to eval_map and eval_pair
        Args:
            batch (tensor): The input tensor.
            output_type (int): 0 for scalar (mean of list), 1 for list
            quiet (bool): if true, print the values
            save_fig (bool): if true, plot figures and save them under 'fig_dir'
            fig_dir (str): dir to store images
        Returns:
           score_matrix: result of eval map
           score_pair_matrix: result of eval pair
        """

        batch =  np.reshape(batch,(-1, 96, 84, 5))
        num_batch = len(batch)
        score_matrix = np.zeros((self.metrics_num, self.track_num, num_batch)) * np.nan
        score_pair_matrix = np.zeros((self.pair_num, num_batch)) * np.nan

        for idx in range(num_batch):
            bar = batch[idx]

            # compute eval map
            for t in range(self.track_num):
                if(self.eval_map[0, t]):
                     bar_act = self.metric_is_empty_bar(batch[idx, :, :, t])
                     score_matrix[0, t, idx] = bar_act

                if(self.eval_map[1, t] and not bar_act):
                    score_matrix[1, t, idx] = self.metric_num_pitch_used(batch[idx, :, :, t])

                if(self.eval_map[2, t] and not bar_act):
                    score_matrix[2, t, idx]= self.metric_qualified_note_ratio(batch[idx, :, :, t])

                if(self.eval_map[3, t] and not bar_act):
                    score_matrix[3, t, idx]= self.metric_polyphonic_ratio(batch[idx, :, :, t])

                if(self.eval_map[4, t] and not bar_act):
                    score_matrix[4, t, idx]= self.metric_in_scale(self.to_chroma(batch[idx, :, :, t]))

                if(self.eval_map[5, t] and not bar_act):
                    score_matrix[5, t, idx]= self.metric_drum_pattern(batch[idx, :, :, t])

                if(self.eval_map[6, t] and not bar_act):
                    score_matrix[6, t, idx]= self.metric_num_pitch_used(self.to_chroma(batch[idx, :, :, t]))

            # compute eval pair
            for p in range(self.pair_num):
                pair = self.inter_pair[p]
                score_pair_matrix[p, idx] = self.metrics_harmonicity(self.to_chroma(batch[idx, :, :, pair[0]]),
                                                                     self.to_chroma(batch[idx, :, :, pair[1]]))
        score_matrix_mean = np.nanmean(score_matrix, axis=2)
        score_pair_matrix_mean = np.nanmean(score_pair_matrix, axis=1)

        if not quiet:
            print('# Data Size:', batch.shape, '       # num of Metrics:', np.sum(self.eval_map))
            self.print_metrics_mat(score_matrix_mean)
            self.print_metrics_pair(score_pair_matrix_mean)

        # save figures and save info as npy files
        if save_fig:
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

            print('[*] Plotting Figures...')

            # plot figures for each metric
            for i in range(len(self.metric_names)):
                for j in range(len(self.track_names)):
                    if self.eval_map[i, j]:
                        self.plot_histogram(score_matrix[i, j], fig_dir=fig_dir,
                                            title='['+self.metric_names[i]+']_'+self.track_names[j])

            # info dict
            info = {'score_matrix_mean': score_matrix_mean,
                    'score_pair_matrix_mean': score_pair_matrix_mean,
                    'score_matrix': score_matrix,
                    'score_pair_matrix': score_pair_matrix}

            np.save(os.path.join(fig_dir, 'info.npy'), info)
            print('[*] Done!! saved in %s' %(fig_dir))

        # return vlaues
        if output_type is 0: # mean vlaue, scalar
            return score_matrix_mean, score_pair_matrix_mean

        if output_type is 1: # list of values
            return score_matrix, score_pair_matrix

# check info.npy
def read_info(fig_dir):
    m = Metrics()
    info = np.load(os.path.join(fig_dir, 'info.npy'))
    m.print_metrics_mat(info[()]['score_matrix_mean'])