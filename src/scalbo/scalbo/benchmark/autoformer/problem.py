from deephyper.problem import HpProblem


hp_problem = HpProblem()

hp_problem.add_hyperparameter([16, 32, 64, 128, 256], "encoder_width", default_value=64)
hp_problem.add_hyperparameter([1, 2, 3, 4], "encoder_depth", default_value=3)
hp_problem.add_hyperparameter([16, 32, 64, 128, 256], "decoder_width", default_value=64)
hp_problem.add_hyperparameter([1, 2, 3, 4], "decoder_depth", default_value=3)
hp_problem.add_hyperparameter([16, 32, 64, 128, 256, 512], "CAM_width", default_value=128)
hp_problem.add_hyperparameter([1, 2, 3, 4], "CAM_depth", default_value=3)
hp_problem.add_hyperparameter([0.01, 0.005, 0.002, 0.001], "lr", default_value=0.01)
hp_problem.add_hyperparameter((5, 20), "epochs", default_value=5)
hp_problem.add_hyperparameter([8, 16, 32, 64], "train_batch", default_value=8)
hp_problem.add_hyperparameter([8, 16, 32, 64], "test_batch", default_value=8)