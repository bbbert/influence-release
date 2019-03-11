from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import os
import time
import numpy as np
import pickle
from collections import namedtuple

"""
The default output directory (../../output/)
"""
DEFAULT_OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..', 'output'))

"""
Metadata class for a phase of an experiment.
"""
ExperimentPhase = namedtuple('Phase', ['name', 'idx', 'priority', 'instancemethod'])

def phase(priority=0):
  """
  Decorates an instancemethod of an Experiment class to denote
  that it is an experiment phase. The tags will be used by the class
  decorator `collect_phases' to register phases as a class variable.

  :param priority: A priority number for phase.idxing. Phases will be ordered
                   in order of high to low priority, and then by order of
                   declaration in the class.
  """
  def decorator(func):
    phase_name = func.__name__

    def wrapper(self, *args, **kwargs):
      return func(self, *args, **kwargs)

    wrapper.is_phase = True
    wrapper.priority = priority
    return wrapper
  return decorator

def collect_phases(cls):
  """
  Decorates a class inheriting Experiment. Compiles all instancemethods
  decorated by `phase' into a class variable PHASES containing
  `ExperimentPhase's in the correct order.
  """
  phase_attrs = []
  for attrname in dir(cls):
    attr = getattr(cls, attrname)
    if hasattr(attr, 'is_phase') and attr.is_phase:
      idx = 0
      while idx < len(phase_attrs) and phase_attrs[idx][0] >= attr.priority:
        idx += 1
      phase_attrs.insert(idx, (attr.priority, attrname))

  cls.PHASES = [ExperimentPhase(phase_attr[1], i, phase_attr[0],
                                getattr(cls, phase_attr[1]))
                for i, phase_attr in enumerate(phase_attrs)]
  return cls

@collect_phases
class Experiment(object):
  """
  Base class to be inherited by any experiment.
  """
  def __init__(self, config, out_dir=None):
    """
    :param config: A dictionary containing the experiment's configuration.
    :param out_dir: The directory to save an experiment's results in. If None,
                    defaults to DEFAULT_OUT_DIR.
    """
    self.config = config
    self.out_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR

  """
  A string uniquely identifying the series of experiments.
  """
  experiment_id = "exp"

  @property
  def run_id(self):
    """
    A string that uniquely identifies this particular run of the experiment
    among other runs, depending on the configuration.
    """
    raise NotImplementedError('Each experiment must generate a run_id that is unique across runs')

  @staticmethod
  def get_base_dir(out_dir, experiment_id, run_id):
    """
    Return the experiment base directory given the output directory,
    experiment id and run id. Also ensures that the directory exists.
    """
    base_dir = os.path.join(out_dir, experiment_id, run_id)
    if not os.path.exists(base_dir):
      os.makedirs(base_dir)
    if not os.path.isdir(base_dir):
      raise Exception('{} already exists but is not a directory.'.format(base_id))
    return base_dir

  @property
  def base_dir(self):
    """
    The path to the base directory of this run of the experiment.
    """
    return self.get_base_dir(self.out_dir, self.experiment_id, self.run_id)

  def get_result_path(self, phase_index):
    """
    Returns the path to save the result of a phase in.

    :param phase_index: The index of the phase in PHASES.
    """
    phase = self.PHASES[phase_index]
    result_name = "result_{}-{}.npz".format(phase.idx, phase.name)
    result_path = os.path.join(self.base_dir, result_name)
    return result_path

  @staticmethod
  def get_config_path(base_dir):
    """
    Returns the path to the saved config, given the base dir.

    :param base_dir: The base directory
    """
    return os.path.join(base_dir, 'config.pickle')

  @property
  def config_path(self):
    """
    The path to the config of this run of the experiment.
    """
    return self.get_config_path(self.base_dir)

  @staticmethod
  def load_phase_result(result_path):
    """
    Loads the result of a phase from the give result path. This is independent
    of the phase itself because results are constrained to be homogenous.

    :param result_path: The path to the result.
    """
    data = np.load(result_path)
    result = dict(data)
    for key, value in result.items():
      if value.shape == tuple():
        # This value was previously a scalar, extract it
        result[key] = value.reshape(1)[0]
    return result

  @staticmethod
  def save_phase_result(result_path, result):
    """
    Saves the result of a phase into the given result path.

    :param result_path: The path to the result.
    :param result: A dictionary representing the result of a phase.
    """
    np.savez(result_path, **result)

  @staticmethod
  def load_config(config_path):
    """
    Loads a config from the given path.

    :param config_path: The path to the config.
    """
    with open(config_path, 'rb') as f:
      config = pickle.load(f)
    return config
  
  @staticmethod
  def save_config(config_path, config):
    """
    Saves a config into the given path. Configs are saved using pickle.

    :param config_path: The path to the config.
    :param config: A dictionary representing all configuration for this experiment.
    """
    with open(config_path, 'wb') as f:
      pickle.dump(config, f)

  def run(self,
          force_refresh=False,
          stop_after_phase=None):
    """
    Runs all phases of the experiment, skipping phases that have already
    been run if possible, and if desired. Previous results will be overwritten.

    :param force_refresh: If False, phases with previously saved results will be loaded
                          and skipped. Otherwise, every phase will be re-run.
    :param stop_after_phase: If not None, the experiment will be truncated after this
                             phase index for development convenience.
    """
    print("Experiment {}: running {}".format(self.experiment_id, self.run_id))
    print("Results will be stored in {}".format(self.base_dir))
    print()

    self.save_config(self.config_path, self.config)

    exp_start = time.time()

    self.results = dict()
    for phase in self.PHASES:
      result_path = self.get_result_path(phase.idx)

      if not force_refresh and os.path.exists(result_path):
        print("Loading phase {}-{} from previous run:".format(phase.idx, phase.name))
        print(result_path)
        result = self.load_phase_result(result_path)
      else:
        print("Running phase {}-{}...".format(phase.idx, phase.name))

        phase_start = time.time()
        result = phase.instancemethod(self)
        phase_time = time.time() - phase_start
        print("Phase {}-{} took {} seconds".format(phase.idx, phase.name, phase_time))

        if not isinstance(result, dict):
          raise ValueError('Experiment phases should return dictionaries.')
        self.save_phase_result(result_path, result)

      self.results[phase.name] = result
      print()

      if stop_after_phase is not None and phase.idx == stop_after_phase:
        break

    exp_time = time.time() - exp_start

    print("Experiment {}: run {} completed in {} seconds.".format(
      self.experiment_id, self.run_id, exp_time))
    print()

  def load_results(self):
    """
    Loads the results of a previously run experiment. Fails if the results of
    any phase are missing. Previously loaded/run results will be overwritten.
    """
    result_paths = [self.get_result_path(phase.idx) for phase in self.PHASES]
    all_done = all(os.path.exists(result_path) for result_path in result_paths)
    if not all_done:
      raise ValueError('Unable to load results. Experiment has not been completely run.')

    load_start = time.time()

    self.results = dict()
    for phase, result_path in zip(self.PHASES, result_paths):
      print("Loading phase {}-{} from previous run:".format(phase.idx, phase.name))
      print(result_path)
      self.results[phase.name] = self.load_phase_result(result_path)

    load_time = time.time() - load_start
  
    print("Experiment {}: run {} results loaded in {} seconds.".format(
      self.experiment_id, self.run_id, load_time))
    print()

  @classmethod
  def load_run(cls, run_id, out_dir=None):
    """
    Loads a previously run experiment.
    Fails if the saved config or the results of any phase are missing.

    :param run_id: The unique run_id to load
    :param out_dir: The directory that the experiment's results are saved in.
                    If None, defaults to DEFAULT_OUT_DIR.
    """
    out_dir = out_dir if out_dir is not None else DEFAULT_OUT_DIR
    config_path = cls.get_config_path(cls.get_base_dir(out_dir, cls.experiment_id, run_id))
    config = cls.load_config(config_path)
    exp = cls(config, out_dir=out_dir)
    exp.load_results()
    return exp

@collect_phases
class TestExperiment(Experiment):
  """
  Example experiment class demonstrating how to write experiment phases.
  """
  def __init__(self, config, out_dir=None):
    super(TestExperiment, self).__init__(config, out_dir)

  experiment_id = "test_exp"

  @property
  def run_id(self):
    return "some_meaningful_id"

  # The final phase order is bar, then foo, then foo_bar.

  @phase()
  def foo(self):
    print("foo")
    return { 'foo_result': 3, 'foo_dict': { 'key': 'value' } }

  @phase()
  def foo_bar(self):
    print("foo_bar")
    # Phases can depend on the results of previous phases.
    return { 'foo_bar': self.results['foo']['foo_result'] * self.results['bar']['bar'] }

  @phase(priority=1)
  def bar(self):
    # This phase is run before foo because of its priority
    print("bar")
    return { 'bar': np.random.normal((5, 5)) }
