from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse

from experiments.subset_influence import SubsetInfluenceLogreg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute subset self and test influences')
    parser.add_argument('--out_dir', default=None, type=str,
                        help="The experiment output directory")

    parser.add_argument('--force-refresh', dest='force_refresh', action='store_true',
                        help="Ignore previously saved results")
    parser.add_argument('--invalidate', default=None, type=int,
                        help="Invalidate phases starting from this phase index")
    parser.set_defaults(force_refresh=False)

    parser.add_argument('--dataset-id', default="hospital", type=str,
                        help="The dataset to use")
    parser.add_argument('--subset-seed', default=0, type=int,
                        help="The seed to use for subset selection")
    parser.add_argument('--subset-rel-size', default=0.1, type=float,
                        help="The size of the subset relative to the dataset")
    parser.add_argument('--num-subsets', default=5, type=int,
                        help="The number of subsets per random choice type")
    args = parser.parse_args()

    dataset_config = {
        'dataset_id': args.dataset_id,
        'center_data': False,
        'append_bias': False,
    }
    config = {
        'dataset_config': dataset_config,
        'subset_seed': args.subset_seed,
        'subset_rel_size': args.subset_rel_size,
        'num_subsets': args.num_subsets,
        'cross_validation_folds': 5,
        'normalized_cross_validation_range': {
            'hospital': (1e-4, 1e-1, 10),
            'mnist_small': (1e-3, 1, 4),
        }[args.dataset_id],
    }

    exp = SubsetInfluenceLogreg(config, out_dir=args.out_dir)
    exp.run(force_refresh=args.force_refresh,
            invalidate_phase=args.invalidate)

