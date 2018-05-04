"""Network architecture of the proposed discriminator
"""
NET_D = {}

NET_D['pitch_time_private'] = [
    ('conv3d', (32, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 0 (4, 96, 7)
    ('conv3d', (64, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 1 (4, 16, 7)
]

NET_D['time_pitch_private'] = [
    ('conv3d', (32, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 0 (4, 16, 84)
    ('conv3d', (64, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 1 (4, 16, 7)
]

NET_D['merged_private'] = [
    ('conv3d', (64, (1, 1, 1), (1, 1, 1)), None, 'lrelu'),      # 0 (4, 16, 7)
]

NET_D['shared'] = [
    ('conv3d', (128, (1, 4, 3), (1, 4, 2)), None, 'lrelu'),     # 0 (4, 4, 3)
    ('conv3d', (256, (1, 4, 3), (1, 4, 3)), None, 'lrelu'),     # 1 (4, 1, 1)
]

NET_D['onset'] = [
    ('sum', (3), True),                                         # 0 (4, 96, 1)
    ('conv3d', (32, (1, 6, 1), (1, 6, 1)), None, 'lrelu'),      # 1 (4, 16, 1)
    ('conv3d', (64, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),      # 2 (4, 4, 1)
    ('conv3d', (128, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),     # 3 (4, 1, 1)
]

NET_D['chroma'] = [
    ('conv3d', (64, (1, 1, 12), (1, 1, 12)), None, 'lrelu'),    # 0 (4, 4, 1)
    ('conv3d', (128, (1, 4, 1), (1, 4, 1)), None, 'lrelu'),     # 1 (4, 1, 1)
]

NET_D['merged'] = [
    ('conv3d', (512, (2, 1, 1), (1, 1, 1)), None, 'lrelu'),     # 0 (3, 1, 1)
    ('reshape', (3*512)),
    ('dense', 1),
]
