# cis-430-project

Human Trajectory Prediction Using Denoising Diffusion Probabilistic Models.

## How to inference?

run `python main.py` to load a test sample, predict a trajectory, and display it (using matplotlib's default interface).  it will also log the ADE and FDE for the provided prediction.

## How to train?

There's no dedicated training script at the moment, but these three lines in `main.py`:

```python
ddpm.load_checkpoint("small_adv_unet_time_checkpoint.pth")
# train(ddpm, dataloader, num_epochs=1000, learning_rate=1e-4)
# ddpm.save_checkpoint("small_adv_unet_time_checkpoint.pth")
```

control the training and loading of checkpoints.  uncomment/comment the appropriate lines to change the checkpoint loaded (if any), train the model, or change the checkpoint saved (if any).

## Other Datasets?

Only a VSP loader configured for the UCY dataset is implemented at the moment.  These two lines in `main.py`:

```python
data_frame_size = (720, 576)
dataset = VSPDatasetLoader(vsp_dir="data_zara", vsp_name="crowds_zara01.vsp", frame_size=data_frame_size).load()
```
control dataset loading.  `vsp_dir` and `vsp_name` can be modified to load a different VSP (it currently looks in `./datasets/crowd-data/crowds/data`, but I'll probably change this to be more general).  You can download the UCY dataset used at [here](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data).  `data_frame_size` controls the size of each frame (used to map positions to [-1, 1] range)/