# PUMA: Fully Decentralized Uncertainty-aware Multiagent Trajectory Planner with Real-time Image Segmentation-based Frame Alignment #

#### **Submitted to 2024 IEEE International Conference on Robotics and Automation (ICRA)**

<a target="_blank" href=""><img src="./puma/imgs/pads-const-xy-circle-gif.gif" width="400" height="221" alt="Image segmentation-based real-time frame alignment pipeline (pads, circle, constant drifts)"></a>  <a target="_blank" href=""><img src="./puma/imgs/pads-linear-venn-gif.gif" width="400" height="221" alt="Image segmentation-based real-time frame alignment pipeline (pads, partically overlapping circle, linear drfits)"></a>  

<a target="_blank" href=""><img src="./puma/imgs/random-linear-puma-gif.gif" width="400" height="221" alt="Image segmentation-based real-time frame alignment pipeline with PUMA (random objects, linear drifts)"></a>  <a target="_blank" href=""><img src="./puma/imgs/hw-gif.gif" width="400" height="221" style="margin:20px 20px" alt="Hardware experiments: image segmentation-based real-time frame alignment pipeline (pads, circle)"></a>  

## YouTube Video
[https://www.youtube.com/watch?v=W73p42XRcaQ](https://www.youtube.com/watch?v=W73p42XRcaQ)

## Citation

(ICRA24 Paper) [PUMA: Fully Decentralized Uncertainty-aware Multiagent Trajectory Planner with Real-time Image Segmentation-based Frame Alignment](https://arxiv.org/abs/2311.03655):

```bibtex
@article{kondo2023puma,
  title={{PUMA}: Fully Decentralized Uncertainty-aware Multiagent Trajectory Planner with Real-time Image Segmentation-based Frame Alignment},
  author={Kondo, Kota and Tewari T. Claudius and Peterson, B. Mason and Thomas, Annika and Kinnari, Jouko and Tagliabue, Andrea and How, Jonathan P},
  journal={arXiv preprint arXiv:2311.03655},
  year={2023}
}
```

## Using Docker

Our uncertainty-aware planner uses IPOPT and HSL packages.
To use HSL packages, you need to go to http://www.hsl.rl.ac.uk/ipopt/, click on `HSL Academic Licence` (if you are in an academic institute) to install the solver MA27 (free for everyone), and submit the form. Once you receive the corresponding email, download the compressed file, uncompress it (eg. `coinhsl-2015.06.23`), and place it in the folder `puma/puma/docker`.

### PUMA
To run uncertainty-aware planner with one dynamic obstacle, run the following script.
```
roscd puma && cd docker
make build-planner
make run-planner
```

### Image Segmentation-based Real-time Frame Alignment
To run frame alignment pipeline, run the following script.
```
roscd puma && cd docker
make build-frame
make run-frame
```

### Multiagent PUMA on Segmentation-based Real-time Frame Alignment
To run multiagent PUMA on frame alignment pipeline, run the following script.
```
roscd puma && cd docker
make build-multiagent
make run-multiagent
```
### Note
If you see the following error message, you need to run `xhost +local:docker` on your host machine.
```
No protocol specified
qt.qpa.xcb: could not connect to display :1
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
```

<details>
  <summary> <b>NOT USING DOCKER</b></summary>

PUMA has been tested with Ubuntu 20.04/ROS Noetic.

### PUMA
To set up an environment for PUMA, run the following script.
```
./install_puma_deps.bash
```

### Image Segmentation-based Real-time Frame Alignment
To set up an environment for the frame alignment pipeline, run the following script.
```
./install_fastsam_deps.bash
```

## Demos

PUMA has been tested with Ubuntu 20.04/ROS Noetic. Other Ubuntu/ROS version may need some minor modifications, feel free to [create an issue](https://github.com/mit-acl/puma/issues) if you have any problems.

The python scripts described below use `tmux`, and if you want to see what is going on in the background, use `tmux attach`.

### PUMA

```
roscd puma && cd other/demos
python3 uncertainty_aware_planner_demo.py
```
* `uncertainty_aware_planner_demo.py` runs our uncertainty-aware planner with one dynamic obstacle and visualize it in RViz.
* If you want to change parameters of the planner, you can take a look at `puma.yaml` in the `param` folder.
* If you want to change the planner's optimization formulation, you can take a look at `main.m` in the `matlab` folder.
* Note that PUMA is still computationally heavy, and therefore we pause the ROS time while solving for the optimal trajectory -- you can change this in `pause_time_when_replanning` in `puma.yaml`.

### Image Segmentation-based Real-time Frame Alignment

```
roscd puma && cd other/demos
python3 frame_alignment_demo.py
```
* `frame_alignment_demo.py` runs our frame alignment algorithm and visualize it in RViz.
* If you want to record a bag, pass `True` to `--record_bag` and specify where to save a rosbag in `--output_dir`. 
* If you don't have CUDA on your computer, change `self.DEVICE` in `fastsam.py` to `cpu`.

### Multiagent PUMA on Segmentation-based Real-time Frame Alignment

```
roscd puma && cd other/demos
python3 uncertaintyaware_planner_on_frame_alignment_demo.py
```
* Note that PUMA is still computationally heavy, and therefore we pause the ROS time while solving for the optimal trajectory -- you can change this in `pause_time_when_replanning` in `puma.yaml`.
* Currently, `main.m` supports obstacle tracking and uncertainty propagation for one obstacle/agent; however, Check and DelayCheck in [Robust MADER](https://github.com/mit-acl/rmader)'s trajectory deconfliction checks potential for all the received trajectories so PUMA guarantees safety.

</details>

## Important files

If you want to...

* **Tune PUMA's cost functions:** `main.m`
  * Required matlab add-ons: Phased Array System Toolbox, Statistics and Machine Learning Toolbox, Symbolic Math Toolbox
  * PUMA is develped on MATLAB R2022b -- symvar related error on MATLAB R2023b. 
* **Take a look at how we implemented FastSAM:** `fastsam.py`.
* **Modify the optimization problem:** You will need to have MATLAB installed (especifically, you will need the `Symbolic Math Toolbox` and the `Phased Array System Toolbox` installed), and follow the steps detailed in the MATLAB section below. You can then make any modification in the optimization problem by modifying the file `main.m`, and then running it. This will generate all the necessary `.casadi` files in the `casadi_generated_files` folder, which will be read by the C++ code.
