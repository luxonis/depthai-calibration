from typing import Dict, Any, Callable, Iterable, Tuple, TypeVar, Generic, Sequence, List
from collections import abc
import multiprocessing
import random
import queue

T = TypeVar('T')

def allArgs(args, kwargs):
  for arg in args:
    yield arg
  for kwarg in kwargs.values():
    yield kwarg

class Retvals(Generic[T]):
  def __init__(self, taskOrGroup: 'ParallelTask', key: slice | tuple | int):
    self._taskOrGroup = taskOrGroup
    self._key = key

  def __iter__(self) -> T: # Allow iterating over slice or list retvals
    if isinstance(self._key, slice):
      if not self._key.stop:
        raise RuntimeError('Cannot iterate over an unknown length Retvals')
      for i in range(self._key.start or 0, self._key.stop, self._key.step or 1):
        yield Retvals(self._taskOrGroup, i)
    elif isinstance(self._key, list | tuple):
      for i in self._key:
        yield Retvals(self._taskOrGroup, i)
    else:
      yield Retvals(self._taskOrGroup, self._key)

  def __repr__(self):
    return f'<Retvals of {self._taskOrGroup}>'

  def ret(self) -> T:
    if isinstance(self._key, list | tuple):
      ret = self._taskOrGroup.ret()
      return [ret[i] for i in self._key]
    return self._taskOrGroup.ret()[self._key]

  def finished(self) -> bool:
    return self._taskOrGroup.finished()

class ParallelTask(Generic[T]):
  def __init__(self, worker: 'ParallelWorker', fun, args, kwargs):
    self._worker = worker
    self._fun = fun
    self._args = args
    self._kwargs = kwargs
    self._ret = None
    self._exc = None
    self._id = random.getrandbits(64)
    self._finished = False

  def __getitem__(self, key) -> T:
    return Retvals(self, key)

  def __repr__(self):
    return f'<Task {self._fun} {self._id}>'

  def resolveArguments(self) -> None:
    def _replace(args):
      for arg in args:
        if isinstance(arg, ParallelTask | ParallelTaskGroup | Retvals):
          yield arg.ret()
        else:
          yield arg
    self._args = list(_replace(self._args))
    for key, value in self._kwargs:
      if isinstance(value, ParallelTask | ParallelTaskGroup | Retvals):
        self._kwargs[key] = value.ret()

  def isExecutable(self) -> bool:
    for arg in allArgs(self._args, self._kwargs):
      if isinstance(arg, ParallelTask | ParallelTaskGroup | Retvals) and not arg.finished():
        return False
    return True

  def finished(self) -> bool:
    return self._finished

  def finish(self, ret, exc) -> None:
    self._ret = ret
    self._exc = exc
    self._finished = True

  def exc(self) -> BaseException | None:
    return self._exc

  def ret(self) -> T | None:
    return self._ret

class ParallelTaskGroup(Generic[T]):
  def __init__(self, fun, args, kwargs):
    self._fun = fun
    self._args = args
    self._kwargs = kwargs

  def __getitem__(self, key):
    return Retvals(self, key)

  def __repr__(self):
    return f'<TaskGroup {self._fun}>'

  def finished(self) -> bool:
    for task in self._tasks:
      if not task.finished():
        return False
    return True

  def exc(self) -> List[BaseException | None]:
    return list(map(lambda t: t.exc(), self._tasks))

  def tasks(self) -> List[ParallelTask]:
    nTasks = 1
    for arg in allArgs(self._args, self._kwargs):
      if isinstance(arg, ParallelTask | Retvals):
        arg = arg.ret()
      if isinstance(arg, abc.Sized) and not isinstance(arg, str | bytes | dict):
        nTasks = max(nTasks, len(arg))
    self._tasks = []
    for arg in allArgs(self._args, self._kwargs):
      if isinstance(arg, abc.Sized) and len(arg) != nTasks:
        raise RuntimeError('All sized arguments must have the same length or 1')
    
    def argsAtI(i: int):
      for arg in self._args:
        if isinstance(arg, list):
          yield arg[i]
        elif isinstance(arg, ParallelTask | Retvals) and isinstance(arg.ret(), abc.Iterable) and not isinstance(arg.ret(), (str, bytes, dict)):
          yield arg.ret()[i]
        else:
          yield arg
    def kwargsAtI(i: int):
      newKwargs = {}
      for key, value in self._kwargs:
        if isinstance(arg, abc.Sized):
          newKwargs[key] = value[i]
        else:
          newKwargs[key] = value
      return newKwargs

    for i in range(nTasks):
      task = ParallelTask(self, self._fun, list(argsAtI(i)), kwargsAtI(i))
      self._tasks.append(task)
    return self._tasks

  def ret(self) -> List[T | None]:
    def zip_retvals(tasks):
      def _iRet(i):
        for task in tasks:
          yield task.ret()[i]

      l = len(tasks[0].ret()) if isinstance(tasks[0].ret(), abc.Sized) else 1
      if l > 1:
        for i in range(len(tasks[0].ret())):
          yield list(_iRet(i))
      else:
        for task in tasks:
          yield task.ret()

    return list(zip_retvals(self._tasks))

  def isExecutable(self) -> bool:
    for arg in allArgs(self._args, self._kwargs):
      if isinstance(arg, ParallelTask | ParallelTaskGroup | Retvals) and not arg.finished():
        return False
    return True

def worker_controller(_stop: multiprocessing.Event, _in: multiprocessing.Queue, _out: multiprocessing.Queue, _wId: int) -> None:
  while not _stop.is_set():
    try:
      task: ParallelTask | None = _in.get(timeout=0.01)
    except queue.Empty:
      continue
    #try:
    ret, exc = None, None
    ret = task._fun(*task._args, **task._kwargs)
    #except BaseException as e:
      #exc = e
    #finally:
    _out.put((task._id, ret, exc))

class ParallelWorker:
  def __init__(self, workers: int = 16):
    self._workers = workers
    self._tasks: List[ParallelTask] = []

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.execute()

  def run(self, fun: Callable[[Any], T], *args: Tuple[Any], **kwargs: Dict[str, Any]) -> ParallelTask[T]:
    task = ParallelTask(self, fun, args, kwargs)
    self._tasks.append(task)
    return task

  def map(self, fun: Callable[[Any], T], *args: Tuple[Iterable[Any]], **kwargs: Dict[str, Iterable[Any]]) -> ParallelTaskGroup[T]:
    taskGroup = ParallelTaskGroup(fun, args, kwargs)
    self._tasks.append(taskGroup)
    return taskGroup

  def execute(self):
    workerIn = multiprocessing.Manager().Queue()
    workerOut = multiprocessing.Manager().Queue()
    stop = multiprocessing.Event()
    processes = []
    for i in range(self._workers):
      p = multiprocessing.Process(target=worker_controller, args=(stop, workerIn, workerOut, i))
      processes.append(p)
      p.start()

    doneTasks = []
    remaining = len(self._tasks)

    while remaining > 0:
      for task in self._tasks:
        if task.isExecutable():
          if isinstance(task, ParallelTask):
            task.resolveArguments()
            workerIn.put(task)
            doneTasks.append(task)
            self._tasks.remove(task)
          else:
            newTasks = task.tasks()
            remaining += len(newTasks) - 1
            self._tasks.extend(newTasks)
            self._tasks.remove(task)

      if not workerOut.empty():
        tId, ret, exc = workerOut.get()
        for task in doneTasks:
          if task._id == tId:
            task.finish(ret, exc)
            remaining -= 1
    stop.set()
    for p in processes:
      p.join()