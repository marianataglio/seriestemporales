from imports import *

def arima_forecast(train, test, order=(7, 1, 7)):
    # Fit ARIMA model
    arima_model = sm.tsa.ARIMA(train, order=order).fit()

    # Forecast
    predictions = arima_model.forecast(steps=len(test))
    predictions_values = predictions.values
    test_values = test.values
    
    # Calculate RMSE
    rmse = sqrt(mean_squared_error(test_values, predictions_values))
    print('Test RMSE: %.3f' % rmse)
    
    # Plot predictions vs actual values
    plt.figure(figsize=(8, 2.5))
    plt.plot(test_values, label='Test')
    plt.plot(predictions_values, color='red', label='Predictions')
    plt.title("ARIMA Forecast")
    plt.legend()
    plt.show()

    return arima_model

def arima_rolling_forecast(train, test, order=(7, 1, 7)):
    # Guardamos training history
    history = [x for x in train]
    predictions = []

    for t in range(len(test)):
        # fiteamos arima
        arima_rolling_forecast = ARIMA(history, order=order).fit()
        
        # Forecast one step ahead
        output = arima_rolling_forecast.forecast()
        yhat = output[0]
        predictions.append(yhat)
        
        # actualizamos history con los valores observados del test set
        obs = test[t]
        history.append(obs)
    
    # RMSE
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)

    # Plot predictions vs actual values
    plt.figure(figsize=(8,2.5))
    plt.plot(test.values, label='Test')
    plt.plot(predictions, color='red', label='Predictions')
    plt.legend()
    plt.show()
    
    return arima_rolling_forecast


def sarima_forecast(train, test, order=(7, 1, 7), seasonal_order=(4,1,4,28)):
    sarima_forecast = sm.tsa.SARIMAX(train, order=order, seasonal_order=seasonal_order, missing='raise').fit()
    
    # Forecast
    predictions = sarima_forecast.forecast(steps=len(test))
    predictions_values = predictions.values
    test_values = test.values
    
    # Calculamos rmse
    rmse = sqrt(mean_squared_error(test_values, predictions_values))
    print('Test RMSE: %.3f' % rmse)
    
    # Plot predictions vs actual values
    plt.figure(figsize=(8,2.5))
    plt.plot(test_values, label='Test')
    plt.plot(predictions_values, color='red', label='Predictions')
    plt.title("Sarima Forecast")
    plt.legend()
    plt.show()

    return sarima_forecast