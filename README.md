# DepthAI Calibration

This repository contains the calibration scripts for device calibration, which are used in many calibration programs, such as user calibration in [DepthAI](https://github.com/luxonis/depthai) repository, [Factory-caibration-DepthAI](https://github.com/luxonis/Factory-calibration-DepthAI) with RobotArm and others.

## Instructions
### Fetching it as submodule from an exsisting project
In case your repository should have this submodule and it is not detected in files, add it with:
```
git submodule update --init
```
If you just want to update your submodule up to date, use:
```
git pull --recurse-submodules
```

### Instructions on how to integrate this into your calibration routine

You can use this library to easly calibrate the cameras. Firstly you need to clone the submodule in your calibration script as:
```
git submodule add https://github.com/luxonis/depthai-calibration.git
```
Please add this to your `README.md`, to let users of your project know that they need to clone this repository as well:
```
git submodule update --init --recursive
```
Then when submodule is installed, you can just call it as a function as shown in example below:
 ```python
 import depthai_calibration.calibration_utils
 ```
Example of integration of depthai-calibration submodule can be found in our user calibration script [calibrate.py](https://github.com/luxonis/depthai/blob/main/calibrate.py).

