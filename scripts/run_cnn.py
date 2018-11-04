from influence.experiment import CNNExperiment
import influence.experiments

experiment = CNNExperiment("config/cnn_unstable_over_seeds.yaml")
experiment.run()
