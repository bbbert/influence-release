from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import time
import pickle

class TaskQueue(object):
    def __init__(self, task_dir):
        self.task_dir = task_dir
        if not os.path.exists(task_dir):
            os.makedirs(self.task_dir)
        self.task_by_id = dict()

        self.tasks = []
        self.num_tasks_by_id = dict()

        self.uuid = str(time.time()) # Probably not a good idea, but should work

    def define_task(self, task_id, task_func):
        if task_id in self.task_by_id:
            raise ValueError('Task {} has already been defined'.format(task_id))
        self.task_by_id[task_id] = task_func
        self.num_tasks_by_id[task_id] = 0

    @property
    def tasks_path(self):
        return os.path.join(self.task_dir, 'all_tasks.pickle')

    def save_all_tasks(self):
        data = { 'tasks': self.tasks, 'num_tasks_by_id': self.num_tasks_by_id }
        with open(self.tasks_path, 'wb') as f:
            pickle.dump(data, f)

    def load_all_tasks(self):
        if not os.path.exists(self.tasks_path):
            self.tasks = []
            self.num_tasks_by_id = dict()
            return

        with open(self.tasks_path, 'rb') as f:
            data = pickle.load(f)
        self.tasks = data['tasks']
        self.num_tasks_by_id = data['num_tasks_by_id']

    def execute(self, task_id, task_args):
        current_tasks = []
        for args in task_args:
            index = self.num_tasks_by_id[task_id]
            self.tasks.append((task_id, index, args))
            current_tasks.append((task_id, index, args))
            self.num_tasks_by_id[task_id] += 1
        self.save_all_tasks()

        # Purge all claims with no results
        for task_id, index, args in current_tasks:
            claim_path = self.task_claim_path(task_id, index)
            result_path = self.task_result_path(task_id, index)
            if os.path.exists(claim_path) and not os.path.exists(result_path):
                os.remove(claim_path)

        # Do some work as the master too
        self.work()

        # Wait for all tasks to be done
        while True:
            all_done = all(self.task_is_complete(task_id, index)
                           for task_id, index, args in current_tasks)
            if all_done:
                break
            time.sleep(5)

        results = [self.get_task_result(task_id, index)
                   for task_id, index, args in current_tasks]
        return results

    def work(self):
        self.load_all_tasks()

        while True:
            unclaimed_task = self.find_unclaimed_task()
            if unclaimed_task is None:
                break
            task_id, index, args = unclaimed_task
            if not self.claim_task(task_id, index):
                continue

            print("Claimed task {}_{}.".format(task_id, index))
            task_func = self.task_by_id[task_id]
            result = task_func(*args)
            self.complete_task(task_id, index, result)
            print("Completed task {}_{}.".format(task_id, index))

    def task_claim_path(self, task_id, index):
        return os.path.join(self.task_dir, '{}_{}_claim'.format(task_id, index))

    def task_result_path(self, task_id, index):
        return os.path.join(self.task_dir, '{}_{}_result'.format(task_id, index))

    def find_unclaimed_task(self):
        for task_id, index, args in self.tasks:
            path = self.task_claim_path(task_id, index)
            if not os.path.exists(path):
                return task_id, index, args
        return None

    def claim_task(self, task_id, index):
        path = self.task_claim_path(task_id, index)
        if os.path.exists(path):
            return False
        with open(path, 'w') as f:
            f.write(self.uuid)
        time.sleep(0.1)
        with open(path, 'r') as f:
            data = f.read()
        return data == self.uuid

    def complete_task(self, task_id, index, result):
        path = self.task_result_path(task_id, index)
        with open(path, 'wb') as f:
            pickle.dump(result, f)

    def task_is_complete(self, task_id, index):
        path = self.task_result_path(task_id, index)
        return os.path.exists(path)

    def get_task_result(self, task_id, index):
        path = self.task_result_path(task_id, index)
        if not os.path.exists(path):
            raise ValueError('Task {}_{} is not complete yet'.format(task_id, index))
        with open(path, 'rb') as f:
            result = pickle.load(f)
        return result

    def run_worker(self):
        while True:
            time.sleep(1)
            self.work()
    
    def collate_results(self, results):
        keys = set(results[0].keys())
        all_keys_same = all([set(result.keys()) == keys for result in results])
        if not all_keys_same:
            raise ValueError("Could not collate results, not all keys are equal.")

        collated_results = dict()
        for key in keys:
            shape = results[0][key].shape
            if len(shape) == 1:
                collated_results[key] = np.hstack([result[key] for reult in results])
            else:
                collated_results[key] = np.vstack([result[key] for reult in results])

        return collated_results
