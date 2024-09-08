<p align="center">
  <h1 align="center"><ins>GeoCalib</ins> 📸<br>Single-image Calibration with Geometric Optimization</h1>
  <p align="center">
    <a href="https://www.linkedin.com/in/alexander-veicht/">Alexander Veicht</a>
    ·
    <a href="https://psarlin.com/">Paul-Edouard&nbsp;Sarlin</a>
    ·
    <a href="https://www.linkedin.com/in/philipplindenberger/">Philipp Lindenberger</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/people/mapoll/">Marc&nbsp;Pollefeys</a>
  </p>
  <h2 align="center">
    <p>ECCV 2024</p>
    <a href="" align="center">Paper</a> | <!--TODO: update link-->
    <a href="https://colab.research.google.com/drive/1oMzgPGppAPAIQxe-s7SRd_q8r7dVfnqo#scrollTo=etdzQZQzoo-K" align="center">Colab</a> | 
    <a href="https://huggingface.co/spaces/veichta/GeoCalib" align="center">Demo 🤗</a>
  </h2>
  
</p>
<p align="center">
    <a href=""><img src="assets/teaser.gif" alt="example" width=80%></a> <!--TODO: update link-->
    <br>
    <em>
      GeoCalib accurately estimates the camera intrinsics and gravity direction from a single image 
      <br>
      by combining geometric optimization with deep learning.
    </em>
</p>

##

GeoCalib is a an algoritm for single-image calibration: it estimates the camera intrinsics and gravity direction from a single image only. By combining geometric optimization with deep learning, GeoCalib provides a more flexible and accurate calibration compared to previous approaches. This repository hosts the [inference](#setup-and-demo), [evaluation](#evaluation), and [training](#training) code for GeoCalib and instructions to download our training set [OpenPano](#openpano-dataset).


## Setup and demo 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oMzgPGppAPAIQxe-s7SRd_q8r7dVfnqo#scrollTo=etdzQZQzoo-K)
[![Hugging Face](https://img.shields.io/badge/Gradio-Demo-blue)](https://huggingface.co/spaces/veichta/GeoCalib)

We provide a small inference package [`geocalib`](geocalib) that requires only minimal dependencies and Python >= 3.9. First clone the repository and install the dependencies:

```bash
git clone https://github.com/cvg/GeoCalib.git && cd GeoCalib
python -m pip install -e .
# OR
python -m pip install -e "git+https://github.com/cvg/GeoCalib#egg=geocalib"
```

Here is a minimal usage example:

```python
from geocalib import GeoCalib

device = "cuda" if torch.cuda.is_available() else "cpu"
model = GeoCalib().to(device)

# load image as tensor in range [0, 1] with shape [C, H, W]
img = model.load_image("path/to/image.jpg").to(device)
result = model.calibrate(img)

print("camera:", result["camera"])
print("gravity:", result["gravity"])
```

When either the intrinsics or the gravity are already know, they can be provided:

```python
# known intrinsics:
result = model.calibrate(img, priors={"focal": focal_length_tensor})

# known gravity:
result = model.calibrate(img, priors={"gravity": gravity_direction_tensor})
```

The default model is optimized for pinhole images. To handle lens distortion, use the following:

```python
model = GeoCalib(weights="distorted")  # default is "pinhole"
result = model.calibrate(img, camera_model="simple_radial")  # or pinhole, simple_divisional
```

Check out our [demo notebook](demo.ipynb) for a full working example.

<details>
<summary><b>[Interactive demo for your webcam - click to expand]</b></summary>
Run the following command:
  
```bash
python -m geocalib.interactive_demo --camera_id 0
```

The demo will open a window showing the camera feed and the calibration results. If `--camera_id` is not provided, the demo will ask for the IP address of a [droidcam](https://droidcam.app) camera.

Controls:

>Toggle the different features using the following keys:
>
>- ```h```: Show the estimated horizon line
>- ```u```: Show the estimated up-vectors
>- ```l```: Show the estimated latitude heatmap
>- ```c```: Show the confidence heatmap for the up-vectors and latitudes
>- ```d```: Show undistorted image, will overwrite the other features
>- ```g```: Shows a virtual grid of points
>- ```b```: Shows a virtual box object
>
>Change the camera model using the following keys:
>
>- ```1```: Pinhole -> Simple and fast
>- ```2```: Simple Radial -> For small distortions
>- ```3```: Simple Divisional -> For large distortions
>
>Press ```q``` to quit the demo.

</details>


<details>
<summary><b>[Load GeoCalib with torch hub - click to expand]</b></summary>

```python
model = torch.hub.load("cvg/GeoCalib", "GeoCalib", trust_repo=True)
```

</details>

## Evaluation

The full evaluation and training code is provided in the single-image calibration library [`siclib`](siclib), which can be installed as:
```bash
python -m pip install -e siclib
```

Running the evaluation commands will write the results to `outputs/results/`.

### LaMAR

Running the evaluation commands will download the dataset to ```data/lamar2k``` which will take around 400 MB of disk space.

<details>
<summary>[Evaluate GeoCalib]</summary>

To evaluate GeoCalib trained on the OpenPano dataset, run:

```bash
python -m siclib.eval.lamar2k --conf geocalib-pinhole --tag geocalib --overwrite
```

</details>

<details>
<summary>[Evaluate DeepCalib]</summary>

To evaluate DeepCalib trained on the OpenPano dataset, run:

```bash
python -m siclib.eval.lamar2k --conf deepcalib --tag deepcalib --overwrite
```

</details>

<details>
<summary>[Evaluate Perspective Fields]</summary>

Coming soon!

</details>

<details>
<summary>[Evaluate UVP]</summary>

To evaluate UVP, install the [VP-Estimation-with-Prior-Gravity](https://github.com/cvg/VP-Estimation-with-Prior-Gravity) under ```third_party/VP-Estimation-with-Prior-Gravity```. Then run:

```bash
python -m siclib.eval.lamar2k --conf uvp --tag uvp --overwrite data.preprocessing.edge_divisible_by=null
```

</details>

<details>
<summary>[Evaluate your own model]</summary>

If you have trained your own model, you can evaluate it by running:

```bash
python -m siclib.eval.lamar2k --checkpoint <experiment name> --tag <eval name> --overwrite
```

</details>


<details>
<summary>[Results]</summary>

Here are the results for the Area Under the Curve (AUC) for the roll, pitch and field of view (FoV) errors at 1/5/10 degrees for the different methods:

| Approach  | Roll               | Pitch              | FoV                |
| --------- | ------------------ | ------------------ | ------------------ |
| DeepCalib | 44.1 / 73.9 / 84.8 | 10.8 / 28.3 / 49.8 | 0.7 / 13.0 / 24.0  |
| ParamNet  | 51.7 / 77.0 / 86.0 | 27.0 / 52.7 / 70.2 | 02.8 / 06.8 / 14.3 |
| UVP       | 72.7 / 81.8 / 85.7 | 42.3 / 59.9 / 69.4 | 15.6 / 30.6 / 43.5 |
| GeoCalib  | 86.4 / 92.5 / 95.0 | 55.0 / 76.9 / 86.2 | 19.1 / 41.5 / 60.0 |
</details>

### MegaDepth

Running the evaluation commands will download the dataset to ```data/megadepth2k``` or ```data/memegadepth2k-radial``` which will take around 2.1 GB and 1.47 GB of disk space respectively.

<details>
<summary>[Evaluate GeoCalib]</summary>

To evaluate GeoCalib trained on the OpenPano dataset, run:

```bash
python -m siclib.eval.megadepth2k --conf geocalib-pinhole --tag geocalib --overwrite
```

To run the eval on the radial distorted images, run:

```bash
python -m siclib.eval.megadepth2k_radial --conf geocalib-pinhole --tag geocalib --overwrite model.camera_model=simple_radial
```

</details>

<details>
<summary>[Evaluate DeepCalib]</summary>

To evaluate DeepCalib trained on the OpenPano dataset, run:

```bash
python -m siclib.eval.megadepth2k --conf deepcalib --tag deepcalib --overwrite
```

</details>

<details>
<summary>[Evaluate Perspective Fields]</summary>

Coming soon!

</details>

<details>
<summary>[Evaluate UVP]</summary>

To evaluate UVP, install the [VP-Estimation-with-Prior-Gravity](https://github.com/cvg/VP-Estimation-with-Prior-Gravity) under ```third_party/VP-Estimation-with-Prior-Gravity```. Then run:

```bash
python -m siclib.eval.megadepth2k --conf uvp --tag uvp --overwrite data.preprocessing.edge_divisible_by=null
```

</details>

<details>
<summary>[Evaluate your own model]</summary>

If you have trained your own model, you can evaluate it by running:

```bash
python -m siclib.eval.megadepth2k --checkpoint <experiment name> --tag <eval name> --overwrite
```

</details>

<details>
<summary>[Results]</summary>

Here are the results for the Area Under the Curve (AUC) for the roll, pitch and field of view (FoV) errors at 1/5/10 degrees for the different methods:

| Approach  | Roll               | Pitch              | FoV                |
| --------- | ------------------ | ------------------ | ------------------ |
| DeepCalib | 34.6 / 65.4 / 79.4 | 11.9 / 27.8 / 44.8 | 5.6 / 12.1 / 22.9  |
| ParamNet  | 43.4 / 70.7 / 82.2 | 15.4 / 34.5 / 53.3 | 3.2 / 10.1 / 21.3  |
| UVP       | 69.2 / 81.6 / 86.9 | 21.6 / 36.2 / 47.4 | 8.2 / 18.7 / 29.8  |
| GeoCalib  | 82.6 / 90.6 / 94.0 | 32.4 / 53.3 / 67.5 | 13.6 / 31.7 / 48.2 |
</details>

### TartanAir

Running the evaluation commands will download the dataset to ```data/tartanair``` which will take around 1.85 GB of disk space.

<details>
<summary>[Evaluate GeoCalib]</summary>

To evaluate GeoCalib trained on the OpenPano dataset, run:

```bash
python -m siclib.eval.tartanair --conf geocalib-pinhole --tag geocalib --overwrite
```

</details>

<details>
<summary>[Evaluate DeepCalib]</summary>

To evaluate DeepCalib trained on the OpenPano dataset, run:

```bash
python -m siclib.eval.tartanair --conf deepcalib --tag deepcalib --overwrite
```

</details>

<details>
<summary>[Evaluate Perspective Fields]</summary>

Coming soon!

</details>

<details>
<summary>[Evaluate UVP]</summary>

To evaluate UVP, install the [VP-Estimation-with-Prior-Gravity](https://github.com/cvg/VP-Estimation-with-Prior-Gravity) under ```third_party/VP-Estimation-with-Prior-Gravity```. Then run:

```bash
python -m siclib.eval.tartanair --conf uvp --tag uvp --overwrite data.preprocessing.edge_divisible_by=null
```

</details>

<details>
<summary>[Evaluate your own model]</summary>

If you have trained your own model, you can evaluate it by running:

```bash
python -m siclib.eval.tartanair --checkpoint <experiment name> --tag <eval name> --overwrite
```

</details>

<details>
<summary>[Results]</summary>

Here are the results for the Area Under the Curve (AUC) for the roll, pitch and field of view (FoV) errors at 1/5/10 degrees for the different methods:

| Approach  | Roll               | Pitch              | FoV                |
| --------- | ------------------ | ------------------ | ------------------ |
| DeepCalib | 24.7 / 55.4 / 71.5 | 16.3 / 38.8 / 58.5 | 1.5 / 8.8 / 27.2   |
| ParamNet  | 34.5 / 59.2 / 73.9 | 19.4 / 42.0 / 60.3 | 6.0 / 16.8 / 31.6  |
| UVP       | 52.1 / 64.8 / 71.9 | 36.2 / 48.8 / 58.6 | 15.8 / 25.8 / 35.7 |
| GeoCalib  | 71.3 / 83.8 / 89.8 | 38.2 / 62.9 / 76.6 | 14.1 / 30.4 / 47.6 |
</details>

### Stanford2D3D

Before downloading and running the evaluation, you will need to agree to the [terms of use](https://docs.google.com/forms/d/e/1FAIpQLScFR0U8WEUtb7tgjOhhnl31OrkEs73-Y8bQwPeXgebqVKNMpQ/viewform?c=0&w=1) for the Stanford2D3D dataset.
Running the evaluation commands will download the dataset to ```data/stanford2d3d``` which will take around 885 MB of disk space.

<details>
<summary>[Evaluate GeoCalib]</summary>

To evaluate GeoCalib trained on the OpenPano dataset, run:

```bash
python -m siclib.eval.stanford2d3d --conf geocalib-pinhole --tag geocalib --overwrite
```

</details>

<details>
<summary>[Evaluate DeepCalib]</summary>

To evaluate DeepCalib trained on the OpenPano dataset, run:

```bash
python -m siclib.eval.stanford2d3d --conf deepcalib --tag deepcalib --overwrite
```

</details>

<details>
<summary>[Evaluate Perspective Fields]</summary>

Coming soon!

</details>

<details>
<summary>[Evaluate UVP]</summary>

To evaluate UVP, install the [VP-Estimation-with-Prior-Gravity](https://github.com/cvg/VP-Estimation-with-Prior-Gravity) under ```third_party/VP-Estimation-with-Prior-Gravity```. Then run:

```bash
python -m siclib.eval.stanford2d3d --conf uvp --tag uvp --overwrite data.preprocessing.edge_divisible_by=null
```

</details>

<details>
<summary>[Evaluate your own model]</summary>

If you have trained your own model, you can evaluate it by running:

```bash
python -m siclib.eval.stanford2d3d --checkpoint <experiment name> --tag <eval name> --overwrite
```

</details>

<details>
<summary>[Results]</summary>

Here are the results for the Area Under the Curve (AUC) for the roll, pitch and field of view (FoV) errors at 1/5/10 degrees for the different methods:

| Approach  | Roll               | Pitch              | FoV                |
| --------- | ------------------ | ------------------ | ------------------ |
| DeepCalib | 33.8 / 63.9 / 79.2 | 21.6 / 46.9 / 65.7 | 8.1 / 20.6 / 37.6  |
| ParamNet  | 44.6 / 73.9 / 84.8 | 29.2 / 56.7 / 73.1 | 5.8 / 14.3 / 27.8  |
| UVP       | 65.3 / 74.6 / 79.1 | 51.2 / 63.0 / 69.2 | 22.2 / 39.5 / 51.3 |
| GeoCalib  | 83.1 / 91.8 / 94.8 | 52.3 / 74.8 / 84.6 | 17.4 / 40.0 / 59.4 |

</details>

### Evaluation options

If you want to provide priors during the evaluation, you can add one or multiple of the following flags:

```bash
python -m siclib.eval.<benchmark> --conf <config> \
    --tag <tag> \
    data.use_prior_focal=true \
    data.use_prior_gravity=true \
    data.use_prior_k1=true
```

<details>
<summary>[Visual inspection]</summary>

To visually inspect the results of the evaluation, you can run the following command:

```bash
python -m siclib.eval.inspect <benchmark> <one or multiple tags>

```
For example, to inspect the results of the evaluation of the GeoCalib model on the LaMAR dataset, you can run:
```bash
python -m siclib.eval.inspect lamar2k geocalib
```
</details>

## OpenPano Dataset

The OpenPano dataset is a new dataset for single-image calibration which contains about 2.8k panoramas from various sources, namely [HDRMAPS](https://hdrmaps.com/hdris/), [PolyHaven](https://polyhaven.com/hdris), and the [Laval Indoor HDR dataset](http://hdrdb.com/indoor/#presentation). While this dataset is smaller than previous ones, it is publicly available and it provides a better balance between indoor and outdoor scenes.

<details>
<summary>[Downloading and preparing the dataset]</summary>

In order to assemble the training set, first download the Laval dataset following the instructions on [the corresponding project page](http://hdrdb.com/indoor/#presentation) and place the panoramas in ```data/indoorDatasetCalibrated```. Then, tonemap the HDR images using the following command:

```bash
python -m siclib.datasets.utils.tonemapping --hdr_dir data/indoorDatasetCalibrated --out_dir data/laval-tonemap
```

We provide a script to download the PolyHaven and HDRMAPS panos. The script will create folders ```data/openpano/panoramas/{split}``` containing the panoramas specified by the ```{split}_panos.txt``` files. To run the script, execute the following commands:

```bash
python -m siclib.datasets.utils.download_openpano --name openpano --laval_dir data/laval-tonemap
```
Alternatively, you can download the PolyHaven and HDRMAPS panos from [here](https://cvg-data.inf.ethz.ch/GeoCalib_ECCV2024/).


After downloading the panoramas, you can create the training set by running the following command:

```bash
python -m siclib.datasets.create_dataset_from_pano --config-name openpano
```

The dataset creation can be sped up by using multiple workers and a GPU. To do so, add the following arguments to the command:

```bash
python -m siclib.datasets.create_dataset_from_pano --config-name openpano n_workers=10 device=cuda
```

This will create the training set in ```data/openpano/openpano``` with about 37k images for training, 2.1k for validation, and 2.1k for testing.

<details>
<summary>[Distorted OpenPano]</summary>

To create the OpenPano dataset with radial distortion, run the following command:

```bash
python -m siclib.datasets.create_dataset_from_pano --config-name openpano_radial
```

</details>

</details>

## Training

As for the evaluation, the training code is provided in the single-image calibration library [`siclib`](siclib), which can be installed by:

```bash
python -m pip install -e siclib
```

Once the [OpenPano Dataset](#openpano-dataset) has been downloaded and prepared, we can train GeoCalib with it:

First download the pre-trained weights for the [MSCAN-B](https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/) backbone:

```bash
mkdir weights
wget "https://cloud.tsinghua.edu.cn/d/c15b25a6745946618462/files/?p=%2Fmscan_b.pth&dl=1" -O weights/mscan_b.pth
```

Then, start the training with the following command:

```bash
python -m siclib.train geocalib-pinhole-openpano --conf geocalib --distributed
```

Feel free to use any other experiment name. By default, the checkpoints will be written to ```outputs/training/```. The default batch size is 24 which requires 2x 4090 GPUs with 24GB of VRAM each. Configurations are managed by [Hydra](https://hydra.cc/) and can be overwritten from the command line.
For example, to train GeoCalib on a single GPU with a batch size of 5, run:

```bash
python -m siclib.train geocalib-pinhole-openpano \
    --conf geocalib \
    data.train_batch_size=5 # for 1x 2080 GPU
```

Be aware that this can impact the overall performance. You might need to adjust the learning rate and number of training steps accordingly.

If you want to log the training progress to [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://wandb.ai/), you can set the ```train.writer``` option:

```bash
python -m siclib.train geocalib-pinhole-openpano \
    --conf geocalib \
    --distributed \
    train.writer=tensorboard
```

The model can then be evaluated using its experiment name:

```bash
python -m siclib.eval.<benchmark> --checkpoint geocalib-pinhole-openpano \
    --tag geocalib-retrained
```

<details>
<summary>[Training DeepCalib]</summary>

To train DeepCalib on the OpenPano dataset, run:

```bash
python -m siclib.train deepcalib-openpano --conf deepcalib --distributed
```

Make sure that you have generated the [OpenPano Dataset](#openpano-dataset) with radial distortion or add
the flag ```data=openpano``` to the command to train on the pinhole images.

</details>

<details>
<summary>[Training Perspective Fields]</summary>

Coming soon!

</details>

## BibTeX citation

If you use any ideas from the paper or code from this repo, please consider citing:

```bibtex
@inproceedings{veicht2024geocalib,
  author    = {Alexander Veicht and
               Paul-Edouard Sarlin and
               Philipp Lindenberger and
               Marc Pollefeys},
  title     = {{GeoCalib: Single-image Calibration with Geometric Optimization}},
  booktitle = {ECCV},
  year      = {2024}
}
```

## License

The code is provided under the [Apache-2.0 License](LICENSE) while the weights of the trained model are provided under the [Creative Commons Attribution 4.0 International Public License](https://creativecommons.org/licenses/by/4.0/legalcode). Thanks to the authors of the [Laval Indoor HDR dataset](http://hdrdb.com/indoor/#presentation) for allowing this.

