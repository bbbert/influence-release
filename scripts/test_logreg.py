from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import argparse

from experiments.test_logreg import TestLogreg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a test experiment.')
    parser.add_argument('--out_dir', default=None, type=str,
                        help="The experiment output directory")
    parser.add_argument('--force-refresh', dest='force_refresh', action='store_true',
                        help="Ignore previously saved results")
    parser.set_defaults(force_refresh=False)
    parser.add_argument('--dataset-id', default="hospital", type=str,
                        help="The dataset to try training on.")
    args = parser.parse_args()

    dataset_config = {
        'dataset_id': args.dataset_id,
        'center_data': False,
        'append_bias': False,
    }
    config = {
        'dataset_config': dataset_config,
    }
    force_refresh = args.force_refresh

    exp = TestLogreg(config, out_dir=args.out_dir)
    exp.run(force_refresh=force_refresh)
