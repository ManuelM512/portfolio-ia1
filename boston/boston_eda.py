# %%
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split

# %% [markdown]
# Primero importaremos el dataset y visualizaremos las primeras filas para saber con qué estamos trabajando.
# %%
df = pd.read_csv("portfolio-ia1/boston/dataset.csv")
df.head(10)

# %%
df.dtypes

# %% [markdown]
# Revisando los tipos de las columnas, vemos que son todas de tipo float.
# Estas columnas significan:
# - **CRIM** - tasa de criminalidad per cápita por ciudad
# - **ZN** - proporción de terrenos residenciales zonificados para lotes de más de 25,000 pies cuadrados
# - **INDUS** - proporción de acres de negocios no minoristas por ciudad
# - **CHAS** - variable ficticia del río Charles (1 si el sector limita con el río; 0 de lo contrario)
# - **NOX** - concentración de óxidos de nitrógeno (partes por 10 millones)
# - **RM** - número promedio de habitaciones por vivienda
# - **AGE** - proporción de unidades ocupadas por propietarios construidas antes de 1940
# - **DIS** - distancias ponderadas a cinco centros de empleo en Boston
# - **RAD** - índice de accesibilidad a las autopistas radiales
# - **TAX** - tasa de impuesto a la propiedad de valor total por cada $10,000
# - **PTRATIO** - relación alumno-profesor por ciudad
# - **B** - 1000(Bk - 0.63)² donde Bk es la proporción de negros por ciudad
# - **LSTAT** - % de estatus inferior de la población
# - **MEDV** - valor medio de las viviendas ocupadas por propietarios en miles de dólares
#
# Lo que salta a la vista es la variable B, la cual como bien podemos ver, es sesgada étnicamente. Esto no debería haber sido un dato tomado en cuenta, así que la quitaremos del dataset.

# %%
df = df.drop(columns=["B"])
df.columns

# %% [markdown]
# Por otro lado, nuestra variable objetivo sería `MEDV`, así que la usaremos como referencia en futuros ploteos.
# Continuando, veamos si hay `missing values` en el dataset.
# %%
df.isna().sum()
# %% [markdown]
# Tal parece que no hay ninguno. Continuemos con la descripción de las variables.

# %%
df.describe()

# %% [markdown]
# Lo que podemos ver de forma rápida es los cuartiles de `ZN` y `CHAS`, que en 25 y 50% tienen ambas valores 0, incluso `CHAS` también lo tiene para el cuartil 75%.
# Por otro lado, el máximo valor de `MEDV`, que haciendo un poco de investigación, parece fue limitado a ese 50.
#
# Ahora veamos la cantidad de outliers para cada columna.

# %%
for k, v in df.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(df)[0]
    print("%s outliers = %.2f%%" % (k, perc))

# %% [markdown]
# De nuevo, `CHAS` se muestra particular, ahora con un 100% de outliers, que se debe a como se comporta la variable. Esto hace que pueda ser difícil de usarla en un modelo.
# `ZN` y `CRIM` también tienen porcentajes altos de outliers, pudiendo significar esto que hay zonas con muy altos / bajos porcentaje de crimen o una cantidad desproporcionada de áreas residenciales.
# El resto de las features predictivas no tienen una cantidad significativamente infueyente de outliers.
# `MEDV` tiene un porcentaje de outliers el cual deberíamos tener en cuenta, siendo que esto representa casas que valen mucho más/menos, y puede alterar a la predicción del modelo.
# Aún siendo outliers, estos datos pueden representar partes de la realidad necesarias. Por eso, en un principio no voy a tocar a las features predictivas.
# Podría ser bueno quitar los outliers de `MEDV`, existiendo la posibilidad de que una casa sea más valorizada por algo externo a ella
# como podría ser que un famoso vivió ahí.

# %%
df = df[~(df["MEDV"] >= 50.0)]
# %% [markdown]
# Siguiendo, veamos la correlacción entre features.

# %%
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr().abs(), annot=True)
# %% [markdown]
# Buscando por celdas claras en la grilla, encontramos algunas variables muy correlacionadas entre sí.
# El correlacionamiento de variables es algo que puede empeorar el procesado del modelo, al ser que se "muestra" 2 o más veces el mismo dato, y de forma artificial, dandole más importancia.
# Así, las variables muy correlacionadas (|0.6| o más) las eliminaremos.
# Por otro lado, hay variables muy correlacionadas con el objetivo, como es `LSTAT`, `INDUS`, `RM`, `TAX`, `NOX`, `PTRAIO`
# lo cual es muy bueno, indicando la posibilidad de predecir el precio a través de ellas.
#
# Vamos a borrar las columnas correlacionadas y volver a graficar.

# %%
df = df.drop(columns=["DIS", "NOX", "RAD", "TAX", "AGE"])
plt.figure(figsize=(20, 10))
sns.heatmap(df.corr().abs(), annot=True)
# %% [markdown]
# Se dejan estas columnas, las cuales muestran buenos valores de correlación con el objetivo, y no grandes valores entre sí.
# Ahora, vamos a plotear la relación entre `LSTAT`, `INDUS`, `RM` respecto a `MEDV` con un regplot para ver como se comportan.
# Las voy a escalar antes de plotearlas.

# %%
min_max_scaler = preprocessing.MinMaxScaler()
column_sels = ["LSTAT", "INDUS", "RM"]
x = df.loc[:, column_sels]
y = df["MEDV"]
x = pd.DataFrame(data=min_max_scaler.fit_transform(x), columns=column_sels)
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 5))
index = 0
axs = axs.flatten()
for i, k in enumerate(column_sels):
    sns.regplot(y=y, x=x[k], ax=axs[i])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

# %% [markdown]
# Con esto podemos ver las tendencias de `MEDV` respecto a las features elegidas. Por ejemplo,
# el como disminuye respecto a `INDUS` y `LSTAT` cuando estas aumentan, pero por otro lado aumenta a la par que `RM`
#
# Sigamos con un pairplot, para tener un vistazo general de lo que tenemos hasta ahora.

# %%
sns.pairplot(df)

# %% [markdown]
# Algunas columnas (`RM`, `MEDV`) presentan una distribución normal, lo cual puede ayudarnos en la utilización de nuestro modelo.
# Por otro lado, hay pocas tendencias visibles facilmente. Una podría ser el comportamiento de `LSTAT` hacia `MEDV`
# mostrando que a medida que desciende el valor de la casa, es mayor el porcentaje de personas de un contexto económico menor en ese barrio.
# Otra cosa a ver es como aumenta significativamente `CRIM` a medida que `MEDV` disminuye. Esto nos dice que, en barrios con mayor crimen, las casas valen menos.
# La última tendencia que puedo notar es el aumento del valor de la casa (`MEDV`) a medida que aumentan la cantidad de cuartos `RM`, lo cual tiene sentido y creo no amerita mayor explicación.
#
# Con todo esto, parece que tenemos bastante insight para poder empezar a modelar.
# Empecemos con una regresión lineal, siendo que las variables parecen ser bastante lineales y esto un problema de regresión,
# usando **MAE** como métrica. Luego, probaremos con otro modelo, para ver que podemos lograr.
# %%

# Split data into features (X) and target (y)
X = df.drop("MEDV", axis=1)  # All features except 'MEDV'
y = df["MEDV"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create a Linear Regression model
model = LinearRegression()

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)

# Print the results
print(f"Mean Absolute Error: {mae:.2f}")

# Plot predictions vs actual values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# %%
# Podemos ver que el modelo performó bastante bien, con 3 mil dolares de error promedio, lo cual puede ser bastante
# más o menos significativo, dependiendo de la casa. Siendo que la mayoria de los precios se ubican en el entorno de los 20 y 25 mil dolares,
# hablamos de poco más de un 10% de error en el valor del hogar. Así, podemos ver que haciendo un correcto
# tratado de los datos y análisi de estos podemos conseguir un modelo bastante robusto para la predicción,
# aun siendo a partir de un modelo simple como es la regresión lineal.
#
# Para terminar, probemos un modelo más complejo y potente, como lo es `XGBoost`, y comparar resultados.
# Además, se le añadió optimización de parametros a través de optuna, para ver qué tanto puede, o no, mejorar.


# %%
def objective(trial):
    # Define search spaces for hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "alpha": trial.suggest_float("alpha", 0, 10),
        "lambda": trial.suggest_float("lambda", 0, 10),
    }

    xgbr = xgb.XGBRegressor(
        random_state=42,
        objective="reg:absoluteerror",
        **params,
    )

    # Cross-validation score
    score = cross_val_score(
        xgbr, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
    )

    # Minimize the negative MAE
    return -np.mean(score)


# %%
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
best_params = study.best_params


# %%
# Create and train the XGBoost model
xgboost_model = xgb.XGBRegressor(
    objective="reg:absoluteerror", random_state=42, **best_params
)
xgboost_model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred_xgb = xgboost_model.predict(X_test)

# Evaluate the XGBoost model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print(f"MAE XGBoost: {mae_xgb:.2f}")
# Plot predictions vs actual values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

# %% [markdown]
# Era esperable que mejorase, tanto por ser un modelo más potente, como utilizar la optimización de parámetros.
# Hablamos de en promedio, un aproximado de 20% de mejora, lo cual es significativo, y valió la pena la prueba.
