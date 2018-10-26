"""Network architecture of the baseline discriminator
"""
NET_D = {}

NET_D['pitch_time_private'] = None

NET_D['time_pitch_private'] = None

NET_D['merged_private'] = None

NET_D['shared'] = None

NET_D['onset'] = None

NET_D['chroma'] = None

NET_D['merged'] = [
    ('conv3d', (128, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),   # 0 (4, 96, 7)
    ('conv3d', (128, (1, 1, 3), (1, 1, 2)), None, 'lrelu'),     # 1 (4, 96, 3)
    ('conv3d', (256, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),     # 2 (4, 16, 3)
    ('conv3d', (256, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),     # 3 (4, 4, 3)
    ('conv3d', (512, (1, 1, 3), (1, 1, 3)), None, 'lrelu'),     # 4 (4, 4, 1)
    ('conv3d', (512, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),     # 5 (4, 1, 1)
    ('conv3d', (1024, (2, 1, 1), (1, 1, 1)), None, 'lrelu'),    # 6 (3, 1, 1)
    ('reshape', (3*1024)),
    ('dense', 1),
]
