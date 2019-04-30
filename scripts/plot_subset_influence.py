from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import argparse

import matplotlib as mpl
mpl.use('Agg')
from experiments.subset_influence import SubsetInfluenceLogreg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate plots for subset influence experiments')

    # Environment args
    parser.add_argument('--out-dir', default=None, type=str,
                        help="The experiment output directory")

    # Experiment args
    parser.add_argument('--run-id', type=str,
                        help="The run id of the experiment")
    args = parser.parse_args()

    E = SubsetInfluenceLogreg.load_run(args.run_id, out_dir=args.out_dir)
    E.plot_all()
