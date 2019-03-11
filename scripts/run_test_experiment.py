from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import argparse

from experiments.common import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a test experiment.')
    parser.add_argument('--out_dir', default=None, type=str,
                        help="The experiment output directory")
    args = parser.parse_args()

    config = {'fake_config': 'dict'}

    exp = TestExperiment(config, out_dir=args.out_dir)
    exp.run()

    exp = TestExperiment(config, out_dir=args.out_dir)
    exp.load_results()

    exp = TestExperiment.load_run('some_meaningful_id')
    print("Results:", exp.results)
