"""Network architecture of the proposed generator.
"""
NET_G = {}

NET_G['z_dim'] = 128

NET_G['shared'] = [
    ('dense', (3*256), 'bn', 'relu'),                           # 0
    ('reshape', (3, 1, 1, 256)),                                # 1 (3, 1, 1)
    ('transconv3d', (256, (2, 1, 1), (1, 1, 1)), 'bn', 'relu'), # 2 (4, 1, 1)
    ('transconv3d', (128, (1, 4, 1), (1, 4, 1)), 'bn', 'relu'), # 3 (4, 4, 1)
    ('transconv3d', (128, (1, 1, 3), (1, 1, 3)), 'bn', 'relu'), # 4 (4, 4, 3)
    ('transconv3d', (64, (1, 4, 1), (1, 4, 1)), 'bn', 'relu'),  # 5 (4, 16, 3)
    ('transconv3d', (64, (1, 1, 3), (1, 1, 2)), 'bn', 'relu'),  # 6 (4, 16, 7)
]

NET_G['pitch_time_private'] = [
    ('transconv3d', (64, (1, 1, 12), (1, 1, 12)), 'bn', 'relu'),# 0 (4, 16, 84)
    ('transconv3d', (32, (1, 6, 1), (1, 6, 1)), 'bn', 'relu'),  # 1 (4, 96, 84)
]

NET_G['time_pitch_private'] = [
    ('transconv3d', (64, (1, 6, 1), (1, 6, 1)), 'bn', 'relu'),  # 0 (4, 96, 7)
    ('transconv3d', (32, (1, 1, 12), (1, 1, 12)), 'bn', 'relu'),# 1 (4, 96, 84)
]

NET_G['merged_private'] = [
    ('transconv3d', (1, (1, 1, 1), (1, 1, 1)), 'bn', 'sigmoid'),# 0 (4, 96, 84)
]
