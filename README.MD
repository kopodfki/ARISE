Based on the [CVPR2024] ChatScene: Knowledge-Enabled Safety-Critical Scenario Generation for Autonomous Vehicles [ChatScene paper](https://arxiv.org/abs/2405.14062); [ChatScene github](https://javyduck.github.io/chatscene/)

## Installation

This repository utilizes [Scenic](https://github.com/BerkeleyLearnVerify/Scenic).

**Recommended system: Ubuntu 20.04 or 22.04**

### 1. Local Installation

Step 1: Setup conda environment

```
conda create -n arise
conda activate arise
```

Step 2: Clone this git repo in an appropriate folder

```
git clone git@github.com:kopodfki/ARISE.git
```

Step 3: Enter the repo root folder and install the packages:

```
cd ARISE
pip install -r requirements.txt
```

Step 4: Install the Scenic package:

```
cd Scenic
python -m pip install -e .
```

Step 5: Download and install CARLA.

Step 7: Add the python API of CARLA to the ```PYTHONPATH``` environment variable. You can add the following commands to your `~/.bashrc`:

```
export CARLA_ROOT={path/to/your/carla}
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.1x-py3.x-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/agents
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
```

Then, do `source ~/.bashrc` to update the environment variable.

## CARLA Setup

### 1. Desktop Users

Enter the CARLA root folder, launch the CARLA server and run our platform with

```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -windowed -carla-port=2000
```

### 2. Remote Server Users

Enter the CARLA root folder, launch the CARLA server with headless mode, and run our platform with

```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000
```

(Optional) You can also visualize the pygame window using [TurboVNC](https://sourceforge.net/projects/turbovnc/files/).
First, launch CARLA with headless mode, and run our platform on a virtual display.

```
# Launch CARLA
./CarlaUE4.sh -prefernvidia -RenderOffScreen -carla-port=2000

# Run a remote VNC-Xserver. This will create a virtual display "8".
/opt/TurboVNC/bin/vncserver :8 -noxstartup
```

You can use the TurboVNC client on your local machine to connect to the virtual display.

```
# Use the built-in SSH client of TurboVNC Viewer
/opt/TurboVNC/bin/vncviewer -via user@host localhost:n

# Or you can manually forward connections to the remote server by
ssh -L fp:localhost:5900+n user@host
# Open another terminal on local machine
/opt/TurboVNC/bin/vncviewer localhost::fp
```

where `user@host` is your remote server, `fp` is a free TCP port on the local machine, and `n` is the display port specified when you started the VNC server on the remote server ("8" in our example).

BibTeX formatted citation:
```
@misc{poddubnyy2026ariseadaptiverefinement,
      title={ARISE -- Adaptive Refinement and Iterative Scenario Engineering}, 
      author={Konstantin Poddubnyy and Igor Vozniak and Nils Lipp and Ivan Burmistrov and Davit Hovhannisyan and Christian Mueller and Philipp Slusallek},
      year={2026},
      eprint={2601.14743},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2601.14743}, 
}
```



