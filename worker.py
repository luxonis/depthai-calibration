from typing import Dict, Any, Callable, Iterable, TypeVar, Generic
from collections.abc import Iterable
import multiprocessing
import random
import queue

R = TypeVar('R')

class ParallelTask(Generic[R]):
  def __init__(self, fun: Callable[..., R], args: Iterable[Any] = (), kwargs: Dict[str, Any] = {}):
    self._fun = fun
    self._args = args
    self._kwargs = kwargs
    self._id = random.getrandbits(64)
    self._retvals = None

  def __repr__(self):
    return f'<Task {self._fun} {self._id}>'

  # For accessing task return values
  def __iter__(self):
    yield Retvals[R](task=self)

  def __getitem__(self, key):
    return Retvals[R](task=self, key=key)

  @property
  def retvals(self) -> R:
    return self._retvals

  def _canExecute(self) -> bool:
    for arg in self._args:
      if isinstance(arg, Retvals) and not arg.resolved():
        return False
    for kwargs in self._kwargs.values():
      if isinstance(kwargs, Retvals) and not kwargs.resolved():
        return False
    return True

  def _substituteArgs(self):
    newargs = []
    for arg in self._args:
      if isinstance(arg, Retvals):
        rev = arg.get()
        newargs.extend(rev)
      else:
        newargs.append(arg)
    self._args = newargs

    for key, value in self._kwargs.items():
      if isinstance(value, Retvals):
        self._kwargs[key] = value.get()

class ParallelTaskGroup:
  def __init__(self, fun: Callable[[Any], Any], args: Iterable[Iterable[Any]] = [], kwargs: Iterable[Dict[str, Any]] = []):
    if len(args) != 0 and len(kwargs) != 0 and len(args) != len(kwargs):
      raise RuntimeError('The length of args and kwargs must match, or one must not be provided')
    if len(args) == 0:
      args = [()] * len(kwargs)
    if len(kwargs) == 0:
      kwargs = [{}] * len(args)
    self._fun = fun
    self._args = args
    self._kwargs = kwargs
    def _mapTask(args_kwargs):
      args, kwargs = args_kwargs
      if not isinstance(args, (tuple, list)):
        args = [args]
      return ParallelTask(self._fun, args, kwargs)
    self._tasks = list(map(_mapTask, zip(self._args, self._kwargs)))

  # For accessing task return values
  def __iter__(self):
    yield Retvals(taskGroup=self)

  def __getitem__(self, key):
    return Retvals(taskGroup=self, key=key)

class Retvals(Generic[R]):
  def __init__(self, task: ParallelTask = None, taskGroup: ParallelTaskGroup = None, key: slice | tuple | int = None):
    self._task = task
    self._taskGroup = taskGroup
    self._key = key

  def resolved(self) -> bool:
    if self._task:
      return self._task._retvals is not None
    else:
      for task in self._taskGroup._tasks:
        if task._retvals is None:
          return False
      return True

  def get(self) -> R:
    if self._task:
      if self._key is None:
        if isinstance(self._task._retvals, tuple):
          return self._task._retvals
        else:
          return [self._task._retvals]
      elif isinstance(self._key, tuple):
        return [self._task._retvals[i] for i in self._key]
      elif isinstance(self._key, slice):
        return self._task._retvals[self._key]
      return [self._task._retvals[self._key]]
    else:
      def lmap(*args):
        return [list(map(*args))]
      if self._key is None:
        return lmap(lambda task: task._retvals, self._taskGroup._tasks)
      elif isinstance(self._key, tuple):
        return lmap(lambda task: [task._retvals[i] for i in self._key], self._taskGroup._tasks)
      elif isinstance(self._key, slice):
        return lmap(lambda task: task._retvals[self._key], self._taskGroup._tasks)
      return lmap(lambda task: [task._retvals[self._key]], self._taskGroup._tasks)

  def __repr__(self):
    if self._task:
      return f'<Retvals <key:{self._key}> of {self._task}>'
    else:
      return f'<Retvals <key:{self._key}> of {self._taskGroup}>'

def worker_controller(_stop: multiprocessing.Event, _in: multiprocessing.Queue, _out: multiprocessing.Queue, _wId: int) -> None:
  while not _stop.is_set():
    try:
      task: ParallelTask | None = _in.get(timeout=0.1)
    except queue.Empty:
      continue
    retvals = task._fun(*task._args, **task._kwargs)
    _out.put((task._id, retvals))

class ParallelWorker:
  def __init__(self, workers: int = 16):
    self._workerIn = multiprocessing.Queue()
    self._workerOut = multiprocessing.Queue()
    self._stop = multiprocessing.Event()
    self._workers = []

    for i in range(workers):
      p = multiprocessing.Process(target=worker_controller, args=(self._stop, self._workerIn, self._workerOut, i))
      self._workers.append(p)
      p.start()

  def _iterTasks(self, tasks: Iterable[ParallelTask]):
    for task in tasks:
      if isinstance(task, ParallelTaskGroup):
        for task in task._tasks:
          yield task
      else:
        yield task

  def execute(self, tasks: Iterable[ParallelTask]) -> None:
    tasks = list(self._iterTasks(tasks))
    doneTasks = []
    remaining = len(tasks)
    while remaining != 0:
      for task in tasks:
        if task._canExecute():
          task._substituteArgs()
          self._workerIn.put(task)
          doneTasks.append(task)
          tasks.remove(task)

      if not self._workerOut.empty():
        tId, retvals = self._workerOut.get()
        remaining -= 1
        for task in doneTasks:
          if task._id == tId:
            task._retvals = retvals

    self._stop.set()
    for worker in self._workers:
      worker.join()