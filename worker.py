from typing import Dict, Any, Callable, Iterable, Tuple, TypeVar, Generic, List
from collections import abc
import multiprocess
import inspect
import random
import queue
import copy
import ast

T = TypeVar('T')


def allArgs(args, kwargs):
  for arg in args:
    yield arg
  for kwarg in kwargs.values():
    yield kwarg


class Retvals(Generic[T]):
  """Representation of a value yet to be returned by a worker"""

  def __init__(self, taskOrGroup: 'ParallelTask', key: slice | tuple | int):
    self._taskOrGroup = taskOrGroup
    self._key = key

  def __iter__(self) -> T:  # Allow iterating over slice or list retvals
    if isinstance(self._key, slice):
      if not self._key.stop:
        raise RuntimeError('Cannot iterate over an unknown length Retvals')
      for i in range(self._key.start or 0, self._key.stop, self._key.step
                     or 1):
        yield Retvals(self._taskOrGroup, i)
    elif isinstance(self._key, list | tuple):
      for i in self._key:
        yield Retvals(self._taskOrGroup, i)
    else:
      yield Retvals(self._taskOrGroup, self._key)

  def __repr__(self):
    return f'<Retvals of {self._taskOrGroup}>'

  def ret(self) -> T:
    """Retrieve the value returned by the worker"""
    if isinstance(self._key, list | tuple):
      ret = self._taskOrGroup.ret()
      return [ret[i] for i in self._key]
    return self._taskOrGroup.ret()[self._key]

  def finished(self) -> bool:
    """Check if the worker has finished"""
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
    """Convert all Retvals arguments into their underlying values"""

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
    """Check if all tasks on which this task dependes have finished"""

    for arg in allArgs(self._args, self._kwargs):
      if isinstance(arg, ParallelTask | ParallelTaskGroup
                    | Retvals) and not arg.finished():
        return False
    return True

  def finished(self) -> bool:
    """Check if this task has finished"""

    return self._finished

  def finish(self, ret, exc) -> None:
    self._ret = ret
    self._exc = exc
    self._finished = True

  def exc(self) -> BaseException | None:
    """Retrieve an exception, if it occured, otherwise None"""
    return self._exc

  def ret(self) -> T | None:
    """Retrieve the return value"""
    return self._ret


class ParallelTaskGroup(Generic[T]):

  def __init__(self, fun, args, kwargs):
    self._fun = fun
    self._args = args
    self._kwargs = kwargs
    self._tasks = None

  def __getitem__(self, key):
    return Retvals(self, key)

  def __repr__(self):
    return f'<TaskGroup {self._fun}>'

  def finished(self) -> bool:
    """Check if all tasks in the group have finished"""

    if not self._tasks:
      return False
    for task in self._tasks:
      if not task.finished():
        return False
    return True

  def exc(self) -> List[BaseException | None]:
    """Retrieve any exceptions from the tasks of this group"""

    return list(map(lambda t: t.exc(), self._tasks))

  def tasks(self) -> List[ParallelTask]:
    """Retrieve the tasks which make up this task group"""

    nTasks = 1
    for arg in allArgs(self._args, self._kwargs):
      if isinstance(arg, ParallelTask | Retvals):
        arg = arg.ret()
      if isinstance(arg,
                    abc.Sized) and not isinstance(arg, str | bytes | dict):
        nTasks = max(nTasks, len(arg))
    self._tasks = []
    for arg in allArgs(self._args, self._kwargs):
      if isinstance(arg, abc.Sized) and len(arg) != nTasks:
        raise RuntimeError(
            'All sized arguments must have the same length or 1')

    def argsAtI(i: int):
      for arg in self._args:
        if isinstance(arg, list):
          yield arg[i]
        elif isinstance(arg, ParallelTask | Retvals) and isinstance(
            arg.ret(),
            abc.Iterable) and not isinstance(arg.ret(), (str, bytes, dict)):
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
    """Retrieve the return value of all tasks in the group as a single list"""

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
    """Check if all dependency tasks have finished"""
    for arg in allArgs(self._args, self._kwargs):
      if isinstance(arg, ParallelTask | ParallelTaskGroup
                    | Retvals) and not arg.finished():
        return False
    return True


def worker_controller(_stop: multiprocess.Event, _in: multiprocess.Queue,
                      _out: multiprocess.Queue, _wId: int) -> None:
  while not _stop.is_set():
    try:
      task: ParallelTask | None = _in.get(timeout=0.01)

      ret, exc = None, None
      try:
        ret = task._fun(*task._args, **task._kwargs)
      except BaseException as e:
        exc = e
      _out.put((task._id, ret, exc))
    except queue.Empty:
      continue
    except:
      break


class ParallelWorker:

  def __init__(self, workers: int = 16):
    self._workers = workers
    self._tasks: List[ParallelTask] = []

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.execute()

  def run(self, fun: Callable[[Any], T], *args: Tuple[Any],
          **kwargs: Dict[str, Any]) -> ParallelTask[T]:
    task = ParallelTask(self, fun, args, kwargs)
    self._tasks.append(task)
    return task

  def map(self, fun: Callable[[Any], T], *args: Tuple[Iterable[Any]],
          **kwargs: Dict[str, Iterable[Any]]) -> ParallelTaskGroup[T]:
    taskGroup = ParallelTaskGroup(fun, args, kwargs)
    self._tasks.append(taskGroup)
    return taskGroup

  def execute(self):
    workerIn = multiprocess.Manager().Queue()
    workerOut = multiprocess.Manager().Queue()
    stop = multiprocess.Event()
    processes = []
    for i in range(self._workers):
      p = multiprocess.Process(target=worker_controller,
                               args=(stop, workerIn, workerOut, i))
      processes.append(p)
      p.start()

    doneTasks = []
    remaining = len(self._tasks)

    try:
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
              if exc:
                raise Exception(
                    f'Calibration pipeline failed during \'{task}\'') from exc
              task.finish(ret, exc)
              remaining -= 1
    finally:
      stop.set()
      for p in processes:
        p.join()


class ParallelFunctionTransformer(ast.NodeTransformer):
  """AST Transformer for parsing ParallelFunctions and replacing calls to other parallel functions with pw.run(fun)"""

  def __init__(self, pw: ParallelWorker):
    self._pw = pw

  def visit_Expr(self, node):
    """Finds all standalone expressions and converts them to __ParallelWorker_run__"""
    if isinstance(node.value, ast.Call):
      call_node = node.value
      new_node = ast.Expr(value=ast.Call(
          func=ast.Name(id='__ParallelWorker_run__', ctx=ast.Load()),
          args=[ast.Constant(0), call_node.func, *call_node.args],
          keywords=call_node.keywords))
      ast.fix_missing_locations(new_node)
      return ast.copy_location(new_node, node)
    return node

  def visit_Assign(self, node):
    """
    Finds all assignments and converts them to __ParallelWorker_run__
    """
    targets = node.targets
    target_names = []
    for target in targets:
      if isinstance(target, ast.Tuple):
        target_names.extend(
            [elt.id for elt in target.elts if isinstance(elt, ast.Name)])
      elif isinstance(target, ast.Name):
        target_names.append(target.id)

    # Modify the assignment to include print calls
    if isinstance(node.value, ast.Call):
      assign_call = node.value
      new_assign = ast.Assign(
          targets=node.targets,
          value=ast.Call(func=ast.Name(id='__ParallelWorker_run__',
                                       ctx=ast.Load()),
                         args=[
                             ast.Constant(len(target_names)), assign_call.func,
                             *assign_call.args
                         ],
                         keywords=assign_call.keywords))
      ast.fix_missing_locations(new_assign)
      return new_assign
    return node

  def visit_For(self, node: ast.For):
    """
    Finds all for loops that append to arrays, converting them to __ParallelWorker_map__
    """
    if isinstance(node.iter, ast.Name):
      being_iterated = copy.deepcopy([node.iter])
    else:
      being_iterated = copy.deepcopy(node.iter.args)

    if isinstance(node.target, ast.Name):
      itElNames = [node.target.id]
    else:
      itElNames = [n.id for n in node.target.elts]

    appending = {}
    funName = None

    for el in node.body:
      match el:
        case ast.Expr():
          # Filter for only '.append' functions
          if not isinstance(el.value,
                            ast.Call) or el.value.func.attr != 'append':
            continue
          appending[el.value.args[0].id] = copy.deepcopy(el.value.func.value)
          appending[el.value.args[0].id].ctx = ast.Load()
        case ast.Assign():
          if funName:  # If there already was a function, then it can't be valid
            funName = None
            break
          if not isinstance(el.value, ast.Call):
            continue
          funName = copy.deepcopy(el.value.func)
          args = copy.deepcopy(el.value.args)
          if isinstance(el.targets[0], ast.Name):
            assigns = el.targets[0]
          else:
            assigns = el.targets[0].elts

    # If the signature doesn't match at all then skip
    if not appending or not funName or len(node.body) > 10:
      return self.generic_visit(node)

    argMap = {a: b for a, b in zip(itElNames, being_iterated)}
    assigns = [appending[a.id] for a in assigns if a.id in appending]

    args = [ast.Constant(len(assigns))] + [funName] + [
        argMap.get(a.id if isinstance(a, ast.Name) else 0, a) for a in args
    ]

    for arg in args:
      arg.ctx = ast.Load()

    for assign in assigns:
      assign.ctx = ast.Store()

    funName.ctx = ast.Load()

    # Create the map call node and return it
    if len(assigns) != 1:
      assigns = [ast.Tuple(elts=assigns, ctx=ast.Store())]

    workerMap = ast.Assign(targets=assigns,
                           value=ast.Call(func=ast.Name(
                               id="__ParallelWorker_map__", ctx=ast.Load()),
                                          args=args,
                                          keywords=[]))

    if_condition = ast.Call(
        func=ast.Name(id="isinstance", ctx=ast.Load()),
        args=[funName,
              ast.Name(id="ParallelFunction", ctx=ast.Load())],
        keywords=[])
    if_node = ast.If(test=if_condition, body=[workerMap], orelse=[node])

    ast.fix_missing_locations(if_node)
    return if_node


class ParallelFunction:
  """Decorator for a parallel function"""

  def __init__(self, fun):
    self._fun = fun

  def __call__(self, *args, **kwargs):
    return self._fun(*args, **kwargs)

  @staticmethod
  def _run_function(fun, *args, **kwargs):
    return fun(*args, **kwargs)

  def run_parallel(self, workers: int = 8, *args, **kwargs):
    """Run the function in parallel

    Args:
        workers (int, optional): Number of worker processes to use. Defaults to 8.
    """
    pw = ParallelWorker(workers)

    # Recurse through the function and find all parallelizable functions
    func_globals = self._convert_to_pw(pw)
    ret = func_globals[self._fun.__name__](*args, **kwargs)

    # Execute all parallel functions
    pw.execute()

    # Unwind and replace retvals
    def replace_retvals(el):
      if isinstance(el, Retvals):
        return el.ret()
      elif isinstance(el, dict):
        return {k: replace_retvals(v) for k, v in el.items()}
      elif isinstance(el, list):
        return list(map(replace_retvals, el))
      elif isinstance(el, tuple):
        return tuple(map(replace_retvals, el))
      return el

    return replace_retvals(ret)

  def _convert_to_pw(self, pw: ParallelWorker):
    source_lines = inspect.getsourcelines(self._fun)[0]
    source = ''.join(
        [line for line in source_lines if not line.startswith('@')])

    tree = ast.parse(source)

    # Walk the parsed AST and add calls to ParallelWorkers
    transformer = ParallelFunctionTransformer(pw)
    transformed_tree = transformer.visit(tree)

    ast.fix_missing_locations(transformed_tree)

    # Recompile to bytecode
    compiled_code = compile(transformed_tree, filename="<ast>", mode="exec")
    func_globals = self._fun.__globals__.copy()

    def worker_run(nret, fun, *args):
      # Check whether it's a parallel function, otherwise treat it normally
      if isinstance(fun, ParallelFunction):
        retvals = pw.run(fun, *args)
        if nret > 1:
          return retvals[:nret]
        return retvals
      return fun(*args)

    def worker_map(nret, fun, *args):
      # Check whether it's a parallel function, otherwise treat it normally
      if isinstance(fun, ParallelFunction):
        retvals = pw.map(fun, *args)
        if nret > 1:
          return retvals[:nret]
        return retvals
      return fun(*args)

    # Inject __ParallelWorker... definitions
    func_globals = {
        **func_globals, '__ParallelWorker_run__': worker_run,
        '__ParallelWorker_map__': worker_map
    }
    exec(compiled_code, func_globals)

    return func_globals


T = TypeVar("T")


def parallel_function(fun: T) -> T:
  return ParallelFunction(fun)
