# TODO write pdf report
# TODO condense pdf report here

# Continual-Rainbow
 Rainbow algorithm + Continual Backprop, implemented in Stable Baselines 3 - JKU Bachelor Project

### Usage
The code generally follows the Stable Baselines 3 setup, so the usage is similar.

However, for ease of use the file hydra_launcher.py provides both an example of how to use this project and a way to start experiments using the Hydra configuration management tool.

The following command would execute the algortihm for all algorithm configuration and seed combinations in the folders hydra_config/algorithm and hydra_config/random respectively:

```
python hydra_launcher.py --multirun algorithm=glob(*) random=glob(*)
```

For testing pruposes, a command like the following can be used
```
python hydra_launcher.py --multirun algorithm=cbp random=seed1,seed2 wandb.mode=disabled env.parallel_envs=8 env.decorr=False progress_bar=True
```


# Done files

- vec_envs.py
- env_wrapper.py
- retro_utils.py
- logger.py