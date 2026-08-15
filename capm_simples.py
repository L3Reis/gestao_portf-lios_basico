#* CAPM SIMPLES

# Pacotes
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

# Especifique os ativos/portfólios, benchmark e período avaliado:

ativos = ["PETR4.SA", 'ABEV3.SA', 'MGLU3.SA', 'ITSA4.SA']
benchmark = "^BVSP"
inicio = "2020-12-01"
final = "2025-12-31"

tickers = ativos + [benchmark]

df = yf.download(tickers,
                 start= inicio,
                 end= final,
                 progress= False)


precos = df['Close']
retornos = precos.resample("ME").last().pct_change().dropna()
retornos.head()

# Portfólio com pesos iguais

pesos = pd.Series( 1 / len(ativos), index= ativos)
retornos['portfolio'] = retornos[ativos].dot(pesos) # obs: função dot() calcula o produto escalar

# Separação estrita
mercado = retornos[benchmark]
portfolio = retornos["portfolio"]

# Calculando o Beta utilizando a abordagem da covariância

covariancia = portfolio.cov(mercado)
benchmark_variancia = mercado.var()
beta_cov = covariancia / benchmark_variancia
beta_cov


# Através da regressão

y = portfolio
x = sm.add_constant(mercado) # importante adicionar o intercepto

# Modelo
capm = sm.OLS(y,x).fit()

# Resultados
print(capm.summary())


alpha = capm.params["const"]
beta_ols = capm.params[benchmark]

print(f"Alpha mensal: {alpha:.4%}")
print(f"Beta do portfólio: {beta_ols:.4f}")
