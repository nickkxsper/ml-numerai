from StackedModel import NumerAIStack

def main():
    # Driver function to check the above function
    numerai = NumerAIStack(model_type = 'Classification',
                      val_size=1,
                      num_windows=1,
                      rf_n_estimators=1000,
                      rf_max_features=1.,
                      rf_min_samples_split = 2,
                      gb_n_estimators=1000,
                      gb_max_features=1.,
                      cv_split=3,
                      l2_reg = 0,
                      target_days=40,
                      embargo=0.2,
                      one_hot_encode = True
                      )

    models = numerai.window_train(windows=numerai.training_windows, pct_train=0.9)
    tournamentPrediction = numerai.tournament_predict(models)
    numerai.submit_predictions(file = 'predictions.csv', model_id = 'EPOCHTREE')


if __name__ == '__main__': main()
