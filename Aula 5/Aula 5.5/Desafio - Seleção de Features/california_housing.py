from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score
from functools import partial

X, y = fetch_california_housing(return_X_y=True, as_frame=True)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_completo = HistGradientBoostingRegressor(random_state=42)

modelo_completo.fit(X_tr, y_tr)

pred_completo = modelo_completo.predict(X_te)

r2_completo = r2_score(y_te, pred_completo)

mi = partial(mutual_info_regression, random_state=42)

#seleção de features
seletor = SelectKBest(score_func=mi, k=4)

X_tr_sel = seletor.fit_transform(X_tr, y_tr)
X_te_sel = seletor.transform(X_te)

#features mantidas e removidas
features_mantidas = X.columns[seletor.get_support()].tolist()
features_removidas = X.columns[~seletor.get_support()].tolist()

#modelo com as features selecionadas
modelo_selecionado = HistGradientBoostingRegressor(random_state=42)

modelo_selecionado.fit(X_tr_sel, y_tr)

pred_selecionado = modelo_selecionado.predict(X_te_sel)

r2_selecionado = r2_score(y_te, pred_selecionado)

#resultados
print("Método de seleção: SelectKBest com Mutual Information")

print("\nNúmero de features antes:", X.shape[1])
print("Número de features depois:", len(features_mantidas))

print("\nFeatures mantidas:")
print(features_mantidas)

print("\nFeatures removidas:")
print(features_removidas)

print("\nR² antes da seleção:", round(r2_completo, 4))
print("R² depois da seleção:", round(r2_selecionado, 4))

#Insights
#As features MedInc, AveRooms, Latitude e Longitude foram mantidas porque apresentaram maior relevância segundo o método SelectKBest com Mutual Information.
#As features HouseAge, AveBedrms, Population e AveOccup foram removidas porque apresentaram menor relevância em comparação com as features selecionadas.
#O modelo com todas as 8 features obteve R² = 0.8347.
#O modelo com 4 features selecionadas obteve R² = 0.8218.
#Houve uma pequena redução no desempenho, porém o número de atributos foi reduzido pela metade (de 8 para 4).
#Portanto, a seleção de features foi vantajosa, pois simplificou o modelo mantendo um desempenho próximo ao modelo original.
#Para o modelo final eu manteria MedInc, AveRooms, Latitude e Longitude, pois foram as features consideradas mais relevantes pelo método utilizado.