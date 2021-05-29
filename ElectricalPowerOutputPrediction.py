import h2o
from h2o.automl import H2OAutoML as AutoML
from h2o.estimators.glm import H2OGeneralizedLinearEstimator as GLM
from h2o.estimators.gbm import H2OGradientBoostingEstimator as GBM
from h2o.estimators.random_forest import H2ORandomForestEstimator as RF
from h2o.grid.grid_search import H2OGridSearch as Grid

h2o.init()

df = h2o.import_file("ElectricalPowerOutputPrediction.csv", destination_frame="df")

print(df.head)

train_df, valid_df, test_df = df.split_frame(ratios=[0.6, 0.2], seed=123)
features = df.columns[:-1]

# General Linear
model_glm = GLM(model_id = 'electric_glm')
model_glm.train(training_frame=train_df, validation_frame=valid_df, y='PE', x=features)

test_glm = model_glm.model_performance(test_df)
print(test_glm)

# Gradient Boosting
model_gbm = GBM(model_id = 'electric_gbm')
model_gbm.train(training_frame=train_df, validation_frame=valid_df, y='PE', x=features)

test_gbm = model_gbm.model_performance(test_df)
print(test_gbm)

# Random Forest
model_rf = RF(model_id = 'electric_rf')
model_rf.train(training_frame=train_df, validation_frame=valid_df, y='PE', x=features)

test_rf = model_rf.model_performance(test_df)
print(test_rf)

# Grid Search
hyper_params = {'max_depth':[2,3,5,7,10,12,14,16]}
grid = Grid(model_gbm, hyper_params, grid_id='depth_grid')
grid.train(training_frame=train_df, validation_frame=valid_df, y='PE', x=features)

# AutoML Leader Board
aml = AutoML(max_models=10, max_runtime_secs=100, seed=123)
aml.train(training_frame=train_df, validation_frame=valid_df, y='PE', x=features)
print(aml.leaderboard)