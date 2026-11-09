DEBUG = True

config = {
    "T": 1000,
    "batch_size": 32,

    # Noise schedule
    "beta_start": 1e-4,
    "beta_end": 2e-2,

    # Dataset
    # "test_img_dir": "../data/test_images/celeba_02.png",

    # Attention
    # "q_dim": 64,
    # "v_dim": 64,

    # Timestep embedding
    "time_embed": 256,
}