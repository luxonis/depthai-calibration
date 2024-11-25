from typing import TypedDict, List, Tuple, Iterable
import cv2.aruco as aruco
from enum import Enum
import numpy as np
import cv2

class StrEnum(Enum): # Doesn't exist in python 3.10
  def __eq__(self, other):
    if isinstance(other, str):
      print('compare string')
      return self.value == other
    return super().__eq__(other)

class CalibrationModel(StrEnum):
  Perspective = 'perspective'
  Fisheye = 'fisheye'

class DistortionModel(StrEnum):
  Normal = 'NORMAL'
  Tilted = 'TILTED'
  Prism = 'PRISM'
  Thermal = 'THERMAL' # TODO : Is this even a distortion model

class CharucoBoard:
  def __init__(self, squaresX = 16, squaresY = 9, squareSize = 1.0, markerSize = 0.8, dictSize = cv2.aruco.DICT_4X4_1000):
    """Charuco board configuration used in a captured dataset

    Args:
        squaresX (int, optional): Number of squares horizontally. Defaults to 16.
        squaresY (int, optional): Number of squares vertically. Defaults to 9.
        squareSize (float, optional): Length of the side of one square (cm). Defaults to 1.0.
        markerSize (float, optional): Length of the side of one marker (cm). Defaults to 0.8.
        dictSize (_type_, optional): cv2 aruco dictionary size. Defaults to cv2.aruco.DICT_4X4_1000.
    """
    self.squaresX = squaresX
    self.squaresY = squaresY
    self.squareSize = squareSize
    self.markerSize = markerSize
    self.dictSize = dictSize

  def __getstate__(self): # Magic to allow pickling the instance without pickling the cv2 dictionary
    state = self.__dict__.copy()
    for hidden in ['_board', '_dictionary']:
      if hidden in state:
        del state[hidden]
    return state

  def __build(self):
    self._dictionary = aruco.Dictionary_get(self.dictSize)
    self._board = aruco.CharucoBoard_create(self.squaresX, self.squaresY, self.squareSize, self.markerSize, self._dictionary)

  @property
  def dictionary(self):
    """cv2 aruco dictionary"""
    if not hasattr(self, '_dictionary'):
      self.__build()
    return self._dictionary

  @property
  def board(self):
    """cv2 aruco board"""
    if not hasattr(self, '_board'):
      self.__build()
    return self._board

class Dataset:
  class Images:
    def __init__(self, images: Iterable[np.ndarray | str]):
      self._images = list(images)

    def at(self, key):
      if isinstance(self._images[key], str):
        self._images[key] = cv2.imread(self._images[key], cv2.IMREAD_UNCHANGED)
      return self._images[key]

    def __len__(self):
      return len(self._images)

    def __iter__(self):
      for i, _ in enumerate(self._images):
        yield self.at(i)

    def __getitem__(self, key):
      return self.at(key)

    def __repr__(self):
      return self._images.__repr__()

  def __init__(self, name: str, board: CharucoBoard, images: List[np.ndarray | str] = [], allCorners: List[np.ndarray] = [], allIds: List[np.ndarray] = [], imageSize: Tuple[float, float] = (), enableFiltering: bool = True):
    """Create a dataset for camera calibration

    Args:
        name (str): Name of the camera for and the key for output data
        board (CharucoBoard): Charuco board configuration used in the dataset
        images (List[np.ndarray  |  str], optional): Set of images for calibration, can be left empty if `allCorners`, `allIds` and `imageSize` is provided. Defaults to [].
        allCorners (List[np.ndarray], optional): Set of corners used for intrinsic calibration. Defaults to [].
        allIds (List[np.ndarray], optional): Set of ids used for intrinsic calibration. Defaults to [].
        imageSize (Tuple[float, float], optional): Size of the images captured during calibration, must be provided if `images` is empty. Defaults to ().
        enableFiltering (bool, optional): Whether to filter provided corners, or use all of them as is. Defaults to True.
    """
    self.name = name
    self.images = images if isinstance(images, Dataset.Images) else Dataset.Images(images)
    self.allCorners = allCorners
    self.allIds = allIds
    self.imageSize = imageSize
    self.board = board
    self.enableFiltering = enableFiltering

class CalibrationConfig:
  def __init__(self, initialMaxThreshold = 0, initialMinFiltered = 0,
               cameraModel = 0, stereoCalibCriteria = 0):
    """Calibration configuration options

    Args:
        initialMaxThreshold (int, optional): _description_. Defaults to 0.
        initialMinFiltered (int, optional): _description_. Defaults to 0.
        cameraModel (int, optional): _description_. Defaults to 0.
        stereoCalibCriteria (int, optional): _description_. Defaults to 0.
    """
    self.initialMaxThreshold = initialMaxThreshold
    self.initialMinFiltered = initialMinFiltered
    self.cameraModel = cameraModel
    self.stereoCalibCriteria = stereoCalibCriteria

class CameraData(TypedDict):
  calib_model: CalibrationModel
  dist_coeff: str
  distortion_model: DistortionModel
  extrinsics: str
  to_cam: str
  filtered_corners: str
  type: str
  filtered_ids: str
  height: str
  width: str
  hfov: str
  socket: str
  images_path: str
  imsize: str
  intrinsics: str
  sensorName: str
  hasAutofocus: str
  model: str
  max_threshold: str
  min_inliers: str
  name: str
  reprojection_error: str
  size: str
  threshold_stepper: str
  ids: str