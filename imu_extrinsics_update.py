import depthai as dai
import numpy as np

import json
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import argparse

stringToCam = {
                'RGB'   : dai.CameraBoardSocket.CAM_A,
                'LEFT'  : dai.CameraBoardSocket.CAM_B,
                'RIGHT' : dai.CameraBoardSocket.CAM_C,
                'CAM_A' : dai.CameraBoardSocket.CAM_A,
                'CAM_B' : dai.CameraBoardSocket.CAM_B,
                'CAM_C' : dai.CameraBoardSocket.CAM_C,
                'CAM_D' : dai.CameraBoardSocket.CAM_D,
                'CAM_E' : dai.CameraBoardSocket.CAM_E,
                'CAM_F' : dai.CameraBoardSocket.CAM_F,
                'CAM_G' : dai.CameraBoardSocket.CAM_G,
                'CAM_H' : dai.CameraBoardSocket.CAM_H
                }

epilog_text="IMU Extrinsics update"
parser = argparse.ArgumentParser(
    epilog=epilog_text, description="~~", formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-bp', '--boardPath', default="", type=str, required=False,
                        help="Board file path")

options = parser.parse_args()

device = dai.Device()
calibHandler = device.readCalibration()

if options.boardPath == "":
    device_name = device.getDeviceName()
    path = (Path(__file__).parent.parent / 'resources/depthai_boards/boards' / f"{device_name}.json")
else:
    device_name = options.boardPath
    path = Path(options.boardPath)
print(f"Opening file {path}.")
with open(path, 'r') as f:
    board = json.load(f)['board_config']
if 'imuExtrinsics' in board:
    extrinsics = board['imuExtrinsics']['sensors']['BNO']['extrinsics']
    translation = extrinsics['specTranslation']
    translation_vec = [translation['x'], translation['y'], translation['z']]
    rot = extrinsics['rotation']
    rot_mat = R.from_euler('zyx', [rot['y'], rot['p'], rot['r']], degrees=True).as_matrix()

    calibHandler.setImuExtrinsics(stringToCam[extrinsics['to_cam']], rot_mat, translation_vec, translation_vec)
    device.flashCalibration2(calibHandler)
    calibHandler.eepromToJsonFile('./calib.json')
    print("Succesfuly uploaded IMU data")
else:
    print(f"Data for {device_name} in {path} does not exsist.")