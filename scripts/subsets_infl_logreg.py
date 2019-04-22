from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse

from experiments.subset_influence import SubsetInfluenceLogreg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute subset self and test influences')
    # Environment args
    parser.add_argument('--data-dir', default=None, type=str,
                        help="The base dataset directory")
    parser.add_argument('--out-dir', default=None, type=str,
                        help="The experiment output directory")
    parser.add_argument('--max-memory', default=1e9, type=int,
                        help="A rule-of-thumb estimate of the GPU memory capacity")

    # Execution args
    parser.add_argument('--force-refresh', dest='force_refresh', action='store_true',
                        help="Ignore previously saved results")
    parser.add_argument('--invalidate', default=None, type=int,
                        help="Invalidate phases starting from this phase index")
    parser.set_defaults(force_refresh=False)

    # Experiment args
    parser.add_argument('--dataset-id', default="hospital", type=str,
                        help="The dataset to use")
    parser.add_argument('--subset-seed', default=0, type=int,
                        help="The seed to use for subset selection")
    parser.add_argument('--subset-rel-size', default=0.1, type=float,
                        help="The size of the subset relative to the dataset")
    parser.add_argument('--num-subsets', default=5, type=int,
                        help="The number of subsets per random choice type")
    parser.add_argument('--inverse-hvp-method', default=None, type=str,
                        help="The dataset to use")
    args = parser.parse_args()

    dataset_config = {
        'dataset_id': args.dataset_id,
        'center_data': False,
        'append_bias': False,
        'data_dir': args.data_dir,
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
            'mnist': (1e-3, 1, 4),
            'spam': (1e-4, 1e-1, 4),
            'cifar10_small': (1e-3, 1, 4),
            'cifar10': (1e-3, 1, 4),
        }[args.dataset_id],
        'inverse_hvp_method': {
            'hospital': 'explicit',
            'mnist_small': 'explicit',
            'mnist': 'cg',
            'spam': 'explicit',
            'cifar10_small': 'cg',
            'cifar10': 'cg',
        }[args.dataset_id],
        'max_memory': int(args.max_memory),
    }

    if args.inverse_hvp_method is not None:
        config['inverse_hvp_method'] = args.inverse_hvp_method

    config['skip_newton'] = config['inverse_hvp_method'] != 'explicit'
    config['skip_z_norms'] = config['inverse_hvp_method'] != 'explicit'

    exp = SubsetInfluenceLogreg(config, out_dir=args.out_dir)
    exp.run(force_refresh=args.force_refresh,
            invalidate_phase=args.invalidate)
