#### Bitacora LSTM

16 abril
Deje corriendo un grid search con 162 experimentos y con toda la data.

Los resultados muestran que la mejor configuracion es la siguiente

hps = {'batch_size': 14,
 'hidden_size': 25,
 'lookback': 28,
 'lr': 0.0003,
 'n_epochs': 300,
 'num_layers': 1,
 'output_size': 1,
 'final_train_loss': 0.013162605464458466,
 'final_test_loss': 0.024842604994773865,
 'train_time': 15.378505945205688}}

**Notas**:

* Creo que puede ser una buena decision probar con 500 epochs.
* Hace falta revisar lo del input_size = lookback. Por qu'e funciona? Probar con input_size = 1



15 abril
Entrenar muchas epochs > 1000 con lr as bajo 0.001 para que overfitee

##### Configuraciones que parecen estar funcionando sobre un dataset peque;o

lookback=1
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)

print(X_train.shape,  y_train.shape)
print(X_test.shape, y_test.shape)

1. Min max scaler, primero le hago el reshape a X_train, X_test, y_train, y_test:
X_train = X_train.squeeze().numpy().reshape(-1, 1)

2. nn.LSTM(input_size=1, hidden_size=50, num_layers=2, batch_first=True)
3. optimizer = optim.Adam(model.parameters(), lr=0.1)
4. loss_fn = nn.MSELoss()
5. loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=14, drop_last=True)
6. n_epochs = 100
7. X_batch.shape, torch.Size([14, 1])


#### Voy a probar con num_layers = 3
num_layers tiene que estar entre 1 y 2 para esta cantidad de datos

#### Voy a probar con 500 epochs
Empeora

#### Voy a probar con loockback 7, red 7,50,1, lr=0.1








##### **1. Min Max Scaler fit_transform en x_train y x_test**

1. hice min max scaler sobre train, o sea hice algo que no esta muy bien que es fit transform sobre xtrain y xtest
2. setee un lr mas alto 0.1 y n_layers = 1
3. entrene con 3000 epochs
4. resultados hasta ahi: parece ya no convergir a la media, la train loss baja de manera exponencial, la test loss tmb aunque igual siguen altisimas.
5. lookback = 14
6. batch_size = 64

##### **2. Pruebo ahora seteando el n_layers = 3**
1. El resultado empeora.

##### **3. Pruebo ahora haciendo dos scaling sobre x e y**
Aca hice un fit transfrom sobre xtrain e ytrain pero trasnform sobre xtest e ytest
1. hice min max scaler sobre train y test
2. setee un lr  0.1 y n_layers = 1
3. entrene con 3000 epochs
4. mse loss
5. lookback = 14
6. batch_size = 64
 
**Resultados**: malos, la loss de train empeora. o sea solo sube
Las predicciones son un desastre igual, no aprende ninugn patron

#### Conclusiones so far:
Al parecer hacer el fit_transform sobre xtrain y el transform sobre x_test manteniendo todo lod emas igual es peor que haciendo el fit_transform sobre los dos, aunque es teoricamnete correcto. Si le seteo un lr de 0.1 mejoran las predicciones un cachin. 


##### **4. Probar una loss mas robusta como huber loss**
el mse es muy sensible a outliers, puebo probar con otras losses mas robustas como huberloss, tomar el log (fit en logatistmo y exponenciar el resultado de la lstm

1. hice min max scaler sobre train (fit_trasnform xtrain, transform x_test)
2. setee un lr  0.01 y n_layers = 1
3. entrene con 3000 epochs
4. **huber loss**
5. lookback = 14
6. batch_size = 64

**Resultados**: la loss de test diverge mycho a partir de la epoch 250 aprox. 
en las preds quedan dos linear horizontales en el eje cero, no aprende nada. un desastre


##### **5. Probar una loss mas robusta como huber loss con minmax scaler en train y test**
el mse es muy sensible a outliers, puebo probar con otras losses mas robustas como huberloss, tomar el log (fit en logatistmo y exponenciar el resultado de la lstm

1. hice min max scaler sobre train y test
2. setee un lr  0.01 y n_layers = 1
3. entrene con 3000 epochs
4. **huber loss**
5. lookback = 14
6. batch_size = 64

**Resultados**: la loss de test diverge mycho a partir de la epoch 250 aprox. 
en las preds quedan dos linear horizontales en el eje cero, no aprende nada. un desastre

##### **6. Pruebo con dataloder shuffle=False, lr=0.01**
Tegno que revisar como hacer el hidden state.reset

1. hice min max scaler sobre train y test
2. setee un lr  0.01 y n_layers = 1
3. entrene con 3000 epochs
4. mse loss
5. lookback = 14
6. Dataloader.shuffle=False
7. batch_size = 64

**Resultados**: la loss de test tiene muchas oscilaciones, nunca converge.
en las preds quedan dos linear horizontales en el eje cero, no aprende nada. un desastre


##### **7. Pruebo con dataloder shuffle=False, lr=0.1**
Tegno que revisar como hacer el hidden state.reset

1. hice min max scaler sobre train y test
2. setee un lr  0.1 y n_layers = 1
3. entrene con 3000 epochs
4. mse loss
5. lookback = 14
6. Dataloader.shuffle=False
7. batch_size = 64


**Resultados**: la loss de test mejora un poco, sigue sin converger, pero esta mas cerca de la de trrain y no tiene tantas oscilaciones. 

en las preds quedan dos linear horizontales en el eje cero, no aprende nada. un desastre


##### **8. Pruebo con dataloder shuffle=True, lr=0.1, batch_size=14**
Tegno que revisar como hacer el hidden state.reset

1. hice min max scaler sobre train y test
2. setee un lr  0.1 y n_layers = 1
3. entrene con 3000 epochs
4. mse loss
5. lookback = 14
6. Dataloader.shuffle=True
7. batch_size = 14

**Resultados**: sobre pred es lo mismo, lal oss de test presenta muchas oscilaciones.

##### **9. Pruebo con el mismo scaler para train y test, batch_size = 14**
Tegno que revisar como hacer el hidden state.reset

1. hice min max scaler sobre train y test
2. setee un lr  0.1 y n_layers = 1
3. entrene con 3000 epochs
4. mse loss
5. lookback = 14
6. Dataloader.shuffle=True
7. batch_size = 14

**Resultados**: las losses suben, predicciones un desastre. 

##### **10. Pruebo con el mismo scaler para train y test, batch_size = 64**
Tegno que revisar como hacer el hidden state.reset

1. hice min max scaler sobre train y test
2. setee un lr  0.1 y n_layers = 1
3. entrene con 3000 epochs
4. mse loss
5. lookback = 14
6. Dataloader.shuffle=True
7. batch_size = 64

**Resultados**: la loss mejora, empieza a converger la de test. Pero las predicciones siguen en el eje 0...

##### **11. Pruebo haciendo inverse trasnform de y_pred_test **
Tegno que revisar como hacer el hidden state.reset

1. hice min max scaler sobre train y test
2. setee un lr  0.1 y n_layers = 1
3. entrene con 3000 epochs
4. mse loss
5. lookback = 14
6. Dataloader.shuffle=True
7. batch_size = 64

**Resultados**: la loss mejora, empieza a converger la de test.un poco mejoran las preds, super levemente

##### **12. Pruebo haciendo inverse trasnform de y_pred_test pero con 2 scalers **
Tegno que revisar como hacer el hidden state.reset

1. hice min max scaler sobre train y test
2. setee un lr  0.1 y n_layers = 1
3. entrene con 3000 epochs
4. mse loss
5. lookback = 14
6. Dataloader.shuffle=True
7. batch_size = 64



#### Pareceria que el scaling sobre train y test no esta bueno, vamos a probar otra cosa usar stndard scalar




##### Probar diferenciar la serie, fitear LSTM. el rdo me va a dar las derivadas, tengo que sumarlas, para integrarlo. 



Graficar la norma del gradiente, ver si tengo exploding gradients 
https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/y

