obs_to_tensor for converting from observations space to tensor
preprocess_obs for normalizing the observations

samples from the buffer are converted to tensors within the buffer
observations from the environment are passed as a list of frames

lazy frames make this a bit more complicated, as they need to be returned as a list