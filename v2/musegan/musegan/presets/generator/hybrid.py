"""Network architecture for the generator of the hybrid model.
"""
NET_G = {}

# Input latent sizes (NOTE: use 0 instead of None)
NET_G['z_dim_shared'] = 32
NET_G['z_dim_private'] = 32
NET_G['z_dim_temporal_shared'] = 32
NET_G['z_dim_temporal_private'] = 32
NET_G['z_dim'] = (NET_G['z_dim_shared'] + NET_G['z_dim_private']
                  + NET_G['z_dim_temporal_shared']
                  + NET_G['z_dim_temporal_private'])

# Temporal generators
NET_G['temporal_shared'] = [
    ('dense', (3*256), 'bn', 'lrelu'),
    ('reshape', (3, 1, 1, 256)),                                 # 1 (3, 1, 1)
    ('transconv3d', (NET_G['z_dim_shared'],                      # 2 (4, 1, 1)
                     (2, 1, 1), (1, 1, 1)), 'bn', 'lrelu'),
    ('reshape', (4, NET_G['z_dim_shared'])),
]

NET_G['temporal_private'] = [
    ('dense', (3*256), 'bn', 'lrelu'),
    ('reshape', (3, 1, 1, 256)),                                 # 1 (3, 1, 1)
    ('transconv3d', (NET_G['z_dim_private'],                     # 2 (4, 1, 1)
                     (2, 1, 1), (1, 1, 1)), 'bn', 'lrelu'),
    ('reshape', (4, NET_G['z_dim_private'])),
]

# Bar generator
NET_G['bar_generator_type'] = 'private'

NET_G['bar_main'] = [
    ('reshape', (4, 1, 1, NET_G['z_dim'])),
    ('transconv3d', (512, (1, 4, 1), (1, 4, 1)), 'bn', 'lrelu'), # 1 (4, 4, 1)
    ('transconv3d', (256, (1, 1, 3), (1, 1, 3)), 'bn', 'lrelu'), # 2 (4, 4, 3)
    ('transconv3d', (128, (1, 4, 1), (1, 4, 1)), 'bn', 'lrelu'), # 3 (4, 16, 3)
    ('transconv3d', (64, (1, 1, 3), (1, 1, 2)), 'bn', 'lrelu'),  # 4 (4, 16, 7)
]

NET_G['bar_pitch_time'] = [
    ('transconv3d', (32, (1, 1, 12), (1, 1, 12)), 'bn', 'lrelu'),# 0 (4, 16, 84)
    ('transconv3d', (16, (1, 6, 1), (1, 6, 1)), 'bn', 'lrelu'),  # 1 (4, 96, 84)
]

NET_G['bar_time_pitch'] = [
    ('transconv3d', (32, (1, 6, 1), (1, 6, 1)), 'bn', 'lrelu'),  # 0 (4, 96, 7)
    ('transconv3d', (16, (1, 1, 12), (1, 1, 12)), 'bn', 'lrelu'),# 1 (4, 96, 84)
]

NET_G['bar_merged'] = [
    ('transconv3d', (1, (1, 1, 1), (1, 1, 1)), 'bn', 'sigmoid'),
]
