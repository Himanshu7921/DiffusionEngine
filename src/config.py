DEBUG = True

config = {
    "T": 500,
    "batch_size": 1,

    # Noise schedule
    "beta_start": 1e-4,
    "beta_end": 2e-2,

    # Dataset
    "test_img_dir": "../data/test_images/pet_img.png",

    # Attention
    "q_dim": 32,
    "v_dim": 32,

    # Timestep embedding
    "time_embed": 256,
}