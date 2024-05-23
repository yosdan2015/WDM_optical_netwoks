import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

# 1. Carregar o conjunto de dados
# Substitua este caminho pelo caminho para o seu conjunto de dados
df=pd.read_csv("C:/Users/Yosdan/OneDrive/Glendo/Valeska/Optisystem/data_valeska.csv")


# 2. Dividir o conjunto de dados em treino e teste
features=["Power_WDM","EDFA1","EDFA2","EDFA3"]
X = df[features]
y = df["Max_Q_Factor"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# 4. Definir as colunas numéricas e categóricas (se houver)
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# 5. Construir o pipeline de pré-processamento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),        # Imputação de valores faltantes
    ('scaler', StandardScaler()),                      # Padronização das features
    ('poly', PolynomialFeatures(degree=2, include_bias=False))  # Adição de features polinomiais
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# 6. Adicionar o modelo ao pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Definir o grid de hiperparâmetros para busca
param_grid = {
    'regressor__n_estimators': [100, 300, 600],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Configurar o GridSearchCV com validação cruzada
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

# 7. Executar a busca por hiperparâmetros
grid_search.fit(X_train, y_train)

# 8. Obter os melhores hiperparâmetros
best_params = grid_search.best_params_
print(f"Best parameters found: {best_params}")

# 9. Treinar o modelo com os melhores hiperparâmetros
best_model = grid_search.best_estimator_

# 10. Avaliar o modelo com validação cruzada
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores)
print(f'Cross-Validation RMSE scores: {cv_rmse}')
print(f'Mean Cross-Validation RMSE: {cv_rmse.mean()}')

# 11. Fazer previsões no conjunto de teste
y_test_pred = best_model.predict(X_test)

# 12. Avaliar o desempenho do modelo final no conjunto de teste
mse_test = mean_squared_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f'Test Mean Squared Error: {mse_test}')
print(f'Test R^2: {r2_test}')
