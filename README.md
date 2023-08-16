# DepthAI Calibration

This repository contains the calibration scripts for device calibration, which are used in many calibration programs, such as calibrate.py [calibrate.py](https://github.com/luxonis/depthai), [Factory-caibration-DepthAI](https://github.com/luxonis/Factory-calibration-DepthAI) and others.

### Instructions
Add this repository with
```
git submodule add https://github.com/luxonis/depthai-calibration.git
```
Add this to your `README.md`, to let users of your project know that they need to clone this repository as well:
```
git submodule update --init --recursive
```
### DepthaAI as submodule
In case your repository should have this submodule and it is not detected in files, add it with:
```
git submodule update --init
```
If you just want to update your submodule up to date, use:
```
git pull --recurse-submodules
```

