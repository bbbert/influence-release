from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import argparse

from experiments.test_distribute import TestDistribute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test the task queue')
    parser.add_argument('--out-dir', default=None, type=str,
                        help="The experiment output directory")

    parser.add_argument('--force-refresh', dest='force_refresh', action='store_true',
                        help="Ignore previously saved results")
    parser.set_defaults(force_refresh=False)
    parser.add_argument('--worker', dest='worker', action='store_true',
                        help="Behave only as a worker")
    parser.set_defaults(worker=False)
    parser.add_argument('--invalidate', default=None, type=int,
                        help="Invalidate phases starting from this phase index")
    args = parser.parse_args()

    config = dict()

    exp = TestDistribute(config, out_dir=args.out_dir)
    if args.worker:
        exp.task_queue.run_worker()
    else:
        exp.run(force_refresh=args.force_refresh,
                invalidate_phase=args.invalidate)
