import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Carregar o conjunto de dados
# Substitua este caminho pelo caminho para o seu conjunto de dados
df=pd.read_csv("C:/Users/Yosdan/OneDrive/Glendo/Valeska/Optisystem/data_valeska.csv")


# 2. Dividir o conjunto de dados em treino e teste
features=["Power_WDM","EDFA1","EDFA2","EDFA3"]
X = df[features]
y = df["Max_Q_Factor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Padronizar os dados (opcional, mas recomendado)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Definir o modelo e o Grid Search
model = GradientBoostingRegressor(random_state=42)

# Definindo o grid de hiperparâmetros para busca
param_grid = {
    'n_estimators': [100, 300, 600],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3,  5,  8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurando o GridSearchCV com validação cruzada
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

# 5. Executar a busca por hiperparâmetros
grid_search.fit(X_train, y_train)

# 6. Obter os melhores hiperparâmetros
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# 7. Treinar o modelo com os melhores hiperparâmetros
best_model = grid_search.best_estimator_

# 8. Avaliar o modelo com validação cruzada
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f'Cross-Validation RMSE scores: {cv_rmse}')
print(f'Mean Cross-Validation RMSE: {cv_rmse.mean()}')

# 9. Fazer previsões no conjunto de teste
y_test_pred = best_model.predict(X_test)

# 10. Avaliar o desempenho do modelo final no conjunto de teste
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test Mean Squared Error: {mse_test}')
print(f'Test R^2: {r2_test}')
