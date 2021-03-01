# SFO : Impact of COVID19 on Passenger Air Traffic

In this project, we will try to highligts the impact of Covid19 on San Francisco International Airport (SFO) Air Traffic Passengers counts. We will compare actual data with a forecast between March 2020 and September 2020.

## Project Set Up and Installation

This project will use Azure AutoML algorithms and Tensorflow for a Deep Learning approach.

All environment files are available in the outputs folder.

## Dataset

### Overview
The dataset used for this experiment will be an opensource univariate dataset provided by the San Francisco International Airport (SFO). The dataset contains all informations on Monthly Passenger Traffic at the airport since July 2005.

More informations about the dataset can be found here :  [https://data.sfgov.org/Transportation/Air-Traffic-Passenger-Statistics/rkru-6vcg](https://data.sfgov.org/Transportation/Air-Traffic-Passenger-Statistics/rkru-6vcg)

The main goal of the experiment will be to see the impact of the COVID19 on passenger traffic at SFO. We are going to compare forecast data train on the dataset before Covid and compare it to actual data from march 2020 to September 2020.

### Task
AutoML forecasting task will be used to try to find the best model to predict the number of passenger.

Deep Learning forecasting will use a simple DNN and a LSTM implementation.

### Access

The DataSet is a single csv file.

    # import the dataset into a DataFrame
    df = pd.read_csv('Air_Traffic_Passenger_Statistics.csv')

![5 first row of the dataframe](/img/raw_dataset.png)

### Tranformation

The dataset need to be manipulated to extract only the column we need. For a forecasting task, we will have only a column with the passenger count and a DateTime index (Monthly).

    # Add a date column in a datetime format in order to use it as an date index
	df['date'] = pd.to_datetime(df['Activity Period'], format = '%Y%m')
	
	# Select only two columns Date and Passenger Count to prepare the forecasting task
	# We want to group all passenger by date regardless of any other features
	df_clean = df[['date','Passenger Count']].groupby('date').sum()

![Transformed dataframe](/img/transformed_dataset.png)

Once done with the transformation we can have a look at the trend by plotting it.

![Monthly Passenger count](/img/plot_dataset.png)

As we can imagine with a dataset of Air Traffic Passenger, we can see that their is a strong seasonality element. Models will have to capture that to make a useful forecast.

Last transformation is a split between before February 2020 and after.

	# Split the dataframe 
	split_date ='2020-02-01'
	X_precovid = X.loc[X['date'] <= split_date]
	X_covid = X.loc[X['date'] > split_date]

### Register the Dataset

To facilitate reusability and avoid being dependant of the csv, we have to upload the datasets into an azure datastore and register them.

	local_path = './data/SFO_prepared.csv'
    local_path_covid = './data/SFO_covid.csv'

    # Save the two dataset to csv
    X_precovid.to_csv(local_path)
    X_covid.to_csv(local_path_covid)

    # upload the local file to a datastore on the cloud

    # get the datastore to upload prepared data
    datastore = ws.get_default_datastore()

    # upload the local file from src_dir to the target_path in datastore
    datastore.upload(src_dir='data', target_path='data', overwrite=True, show_progress=True)

    # create a dataset referencing the cloud location
    dataset_prepared = Dataset.Tabular.from_delimited_files(path = [(datastore, (local_path))])
    dataset_covid = Dataset.Tabular.from_delimited_files(path = [(datastore, (local_path_covid))])

    # Register the Dataset

    dataset_prepared = dataset_prepared.register(workspace=ws,
                                     name='SFO Air Traffic cleaned',
                                     description='SFO Air Traffic predictions cleaned',
                                     create_new_version=True)

    dataset_covid = dataset_covid.register(workspace=ws,
                                     name='SFO Air Traffic covid',
                                     description='SFO Air Traffic predictions covid',
                                     create_new_version=True)

Once this is done, we can easily access the datasets with *Dataset.get_by_name()* method.

	dataset_prepared = Dataset.get_by_name(ws, name='SFO Air Traffic cleaned')

And create a dataframe from it with *to_pandas_dataframe()* method.

	df_prepared = dataset_prepared.to_pandas_dataframe()


## Automated ML

The task we are going to use is a forecasting task. We have to specify the time column and the y column. The other parameter are standard, we will use normalized RMSE as a primary metric to optimize and a cross validation of 5. For forecasting, we use Rolling Origin Cross Validation.

More information on ROCV can be found here :  [https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-forecast#training-and-validation-data](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-auto-train-forecast#training-and-validation-data)

- enable_dnn is set to false (by default) because I want to test dnn with Hyperdrive instead of autoML.

- target_lags, feature_lags and target_rolling_window_size are set to None.

- forecast_horizon is set to 1.

These are the settings for the automl_config.
		
	automl_settings = {
			    "experiment_timeout_minutes":60,
			    "max_concurrent_iterations":4,
			    "primary_metric": 'normalized_root_mean_squared_error',
			    'time_column_name': 'date'
			}
			
	automl_config = AutoMLConfig(compute_target=compute_target,
			                             task = "forecasting",
			                             training_data=dataset_prepared,
			                             label_column_name="Passenger Count",
			                             enable_early_stopping= True,
			                             n_cross_validations=5,
			                             featurization= 'auto',
			                             debug_log = "automl_errors.log",
			                             **automl_settings
			                            )

### Results

A VotingEnsemble with a MAE of 91.9K was the best run for this experiment followed by a DecisionTree and a GradientBoosting model. Prophet and AutoArima performed significaly worst probably because of my autoML setting. I will have to have a better look at the way to use these model more efficiently with autoML.

![AutoML Run Details](/img/autoML_runDetails.png)

Let's have a look at the pipeline steps of this VotingEnsemble:
	('prefittedsoftvotingregressor',

	PreFittedSoftVotingRegressor(estimators=[('10',
	Pipeline(memory=None,
	steps=[('standardscalerwrapper',
	<azureml.automl.runtime.shared.model_wrappers.StandardScalerWrapper object at 0x7f190e9d5d30>),
	('decisiontreeregressor',
	DecisionTreeRegressor(ccp_alpha=0.0,
	criterion='friedman_mse',
	max_depth=None,
	max_features=0.8,
	max_leaf_nodes=None,
	min_impurity_decreas...
	criterion='mse',
	max_depth=None,
	max_features=0.5,
	max_leaf_nodes=None,
	min_impurity_decrease=0.0,
	min_impurity_split=None,
	min_samples_leaf=0.014975785745950434,
	min_samples_split=0.02180025323490051,
	min_weight_fraction_leaf=0.0,
	presort='deprecated',
	random_state=None,
	splitter='best'))],
	verbose=False))],
	weights=[0.8, 0.06666666666666667,
	0.13333333333333333]))

## Hyperparameter Tuning

During the hyperdrive run, we will test 2 different algoritm and 6 hyperparameters.

Models:
-   Simple DNN
-   LSTM

Hyperparameters:
-   epochs
-   layers
-   neurons
-   look back window
-   dropout
-   batch size

Early termination policy:
-   Bandit with a slack_factor of 0.1

The dataset will also be a script parameters.

We will use the TensorFlow estimator object with the 'train.py' script and the pip_packages dependencies.

Finally, the Hyperdrive config will try to minimize the MAE during a max of 40 runs.

    param_sampling = RandomParameterSampling(
    {
        '--n_epochs': choice(100,200,500),
        '--model_type': choice('DNN','LSTM'),
        '--n_layers': choice(0,1,2),
        '--n_neurons': choice(16,64,128),
        '--look_back': choice(6,12,15),
        '--dropout': choice(0.0,0.2,0.3),
        '--batch_size': choice(16,64,128)
    }

)

### Training

#### DNN

For the DNN, we have to create a 2D array where each row contains n months of data and one target data. n = look_back period)

	def dnn_2d(df, look_back):
	    X,Y =[], []
	    for i in range(len(df)-look_back):
	        d=i+look_back
	        X.append(df[i:d,0])
	        Y.append(df[d,0])
	    return np.array(X),np.array(Y)

Example with a look back of 5:
|Training feature | Target |
|--|--|
Passenger Month 1, Passenger Month 2, ..., Passenger Month 5 | Passenger Month 6
Passenger Month 2, Passenger Month 3, ..., Passenger Month 6 | Passenger Month 7
Passenger Month 3, Passenger Month 4, ..., Passenger Month 7 | Passenger Month 8

The model is define by the look back window, the number of layers and the number of neurons for each layer.

	def model_dnn(look_back, n_layers, n_neurons):
	    model = Sequential()
	    model.add(Dense(units=n_neurons, input_dim=look_back, activation='relu'))
	    for i in range(n_layers):
	        model.add(Dense(n_neurons, activation='relu'))
	    model.add(Dense(1))
	    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse','mae'])
	    return model

#### Long Short-Term Memory (LSTM)

The implementation of the Long Short-Term Memory (LSTM) architecture is a bit different.

We need a 3D array instead of the 2D we used for the DNN above because the LSTM layer must be 3D. 

The 3D input layer is defined by:
-   Samples
-   Time steps
-   Features

We have to modify the function above a bit to make it into a 3d shape.

	ddef lstm_3d(df, look_back):
    X, Y =[], []
    for i in range(len(df)-look_back):
        d=i+look_back
        X.append(df[i:d,])
        Y.append(df[d,])
    return np.array(X), np.array(Y)


And the model is defined this way:

	def model_lstm(look_back, n_layers, n_neurons, dropout):
    model=Sequential()    
    model.add(LSTM(n_neurons, activation='relu', input_shape=(1,look_back), dropout=dropout))
    for i in range(n_layers):
        model.add(Dense(n_neurons, activation='relu'))
    model.add(Dense(1))    
    model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])    
    return model


### Results

The best Model was a DNN with a MAE of 106K. I had to set the max run to 40 because of the time constraint of this project so I may have put too many combinations in my Random parameter sampling but I know that with more time, the LSTM should take the lead over the simple DNN.

I also wanted to change the way I output the prediction to predict more than one month because I think it hurt my prediction to use predicted value instead of actual data. I will work on that later to see if the forecast is improved this way.

MAE of the best hyperdrive run : 106673.0 

Choice of HyperParameters:
--dataset : SFO Air Traffic cleaned
--batch_size : 16
--dropout : 0.2
--look_back : 15
--model_type : DNN
--n_epochs : 100
--n_layers : 2
--n_neurons : 128

![HyperDrive Run Details](/img/HDR_runDetails1.png)
![HyperDrive Run Details](/img/HDR_runDetails2.png)

Finally, let's have a look at the Actual vs Predicted plot to see if the model did a good job with the seasonality.

![Actual vs Predicted](/img/Actual_Predicted_bestHDR.png)

And the prediction for the months between March 2020 to September 2020:

![DNN Prediction](/img/hyperdrive_prediction.png)


## Model Deployment

The autoML model performed better than the one I got with the hyperdrive DNN notebook so far so I'm going to deploy this version.

We will use an Azure Container Instance  with authentication enabled. With an ACI we can't use token so we will use key based authentication.

    service_name = 'sfo-prediction-automl'
    
	# Create an inference config from the autoML output files
	inference_config = InferenceConfig(entry_script='scoring_file_v_1_0_0.py',
                                  conda_file = 'conda_env_v_1_0_0.yml',
                                  source_directory='./outputs/AutoML/Outputs',
                                  description='ACI for SFO passengers prediction',
                                  runtime='python'
                                  )

	# Create a deployment config with an Azure Container Instance
	aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1, auth_enabled=True)

	# Deploy the model and check the deployment status
	service = Model.deploy(workspace=ws,
                      name=service_name,
                      models=[model],
                      inference_config=inference_config,
                      deployment_config=aci_config
                      )

	service.wait_for_deployment(show_output=True)

Then we can test our endpoint with the data we need to compare the impact of COVID19.

	data = {
	    "data":
	    [
	        {
	            'date': "2020-03-01 00:00:00,000000",
	        },
	        {
	            'date': "2020-04-01 00:00:00,000000",
	        },
	        {
	            'date': "2020-05-01 00:00:00,000000",
	        },
	        {
	            'date': "2020-06-01 00:00:00,000000",
	        },
	        {
	            'date': "2020-07-01 00:00:00,000000",
	        },
	        {
	            'date': "2020-08-01 00:00:00,000000",
	        },
	        {
	            'date': "2020-09-01 00:00:00,000000",
	        }
	    ],
	}

Finally, we can parse the response sent by the webservice and create a dataframe to plot it with actual data.

	# Parse the result from the webservice request and create a DataFrame to plot it later
	for i in range(len(data['data'][:])):
	    date = pd.to_datetime(data['data'][i]['date'][:10], format = '%Y-%m-%d')
	    df_prediction = df_prediction.append(pd.DataFrame({'Passenger Count': y['forecast'][i]},
	     index=[date]))
	    
	df_prediction.sort_index(inplace=True)


And the prediction:
![Impact of COVID19 on SFO Air Traffic passenger](/img/autoML_prediction.png)


## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
