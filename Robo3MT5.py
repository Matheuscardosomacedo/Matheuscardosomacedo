#pip install MetaTrader5
#pip install pandas
#pip install pytz
#pip install yfinance requests
#pip install beautifulsoup4
#pip install scikit-learn
#pip install numpy

import MetaTrader5 as mt5
import pandas as pd
import time
from datetime import datetime, timedelta
import pytz # Para fusos horários
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib # Para salvar o modelo treinado
# Usando apenas pandas e numpy para evitar dependências externas
PANDAS_TA_AVAILABLE = False

import numpy as np

def coletar_variaveis_mt5():
    """
    Solicita as credenciais e o símbolo de negociação ao usuário
    e retorna as variáveis coletadas.
    """
    print("--- Configuração de Conexão MT5 ---")
    
    # 1. MT5 Login (Número da conta, será lido como string)
   
    mt5_login = input("Por favor, digite o seu LOGIN da conta MT5 (ex: 5043420764): ").strip()
    
    # MT5 exige o login como número inteiro. Vamos validar e converter.
    if not mt5_login.isdigit():
        print("❌ O login precisa conter apenas números. Tente novamente.")
        raise SystemExit(1)
    
    MT5_LOGIN = int(mt5_login)

    # 2. MT5 Senha
    # Usar input() - note que a senha será visível no console.
    MT5_PASSWORD = input("Por favor, digite a SENHA da sua conta MT5 (ex: K@2rCuTe): ").strip()

    # 3. MT5 Servidor
    # O MetaTrader 5 precisa do nome exato do servidor.
    MT5_SERVER = input("Por favor, digite o NOME DO SERVIDOR MT5 (ex: MetaQuotes-Demo): ").strip()
    
    # 4. Símbolo de Negociação
    # O ativo que será negociado (ex: XAUUSD).
    SYMBOL = input("Por favor, digite o SÍMBOLO de negociação (ex: XAUUSD): ").strip().upper()
    
    print("\n--- Variáveis Coletadas ---")
    print(f"MT5_LOGIN:    {MT5_LOGIN}")
    print(f"MT5_PASSWORD: {MT5_PASSWORD} (Não exibida por segurança)")
    print(f"MT5_SERVER:   {MT5_SERVER}")
    print(f"SYMBOL:       {SYMBOL}")
    print("----------------------------\n")
    
    # Retornamos as variáveis coletadas para uso posterior no seu script de trading
    return MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOL

# --- Configurações do Robô (Serão otimizadas) ---
# As variáveis principais agora são coletadas interativamente
MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, SYMBOL = coletar_variaveis_mt5()

# Função para permitir usuário escolher o timeframe
def escolher_timeframe():
    """Permite ao usuário escolher o timeframe para análise."""
    print("\n📊 --- SELEÇÃO DE TIMEFRAME ---")
    print("Escolha o timeframe (período gráfico) para análise:")
    print()
    
    timeframes = {
        1: ('M1', mt5.TIMEFRAME_M1, 1, '1 minuto'),
        2: ('M5', mt5.TIMEFRAME_M5, 5, '5 minutos'),
        3: ('M15', mt5.TIMEFRAME_M15, 15, '15 minutos'),
        4: ('M30', mt5.TIMEFRAME_M30, 30, '30 minutos'),
        5: ('H1', mt5.TIMEFRAME_H1, 60, '1 hora'),
        6: ('H4', mt5.TIMEFRAME_H4, 240, '4 horas'),
        7: ('D1', mt5.TIMEFRAME_D1, 1440, '1 dia')
    }
    
    for num, (name, tf, minutes, desc) in timeframes.items():
        print(f"{num}. {name} - {desc}")
    
    while True:
        try:
            choice = input("\nEscolha o timeframe (1-7): ").strip()
            choice_num = int(choice)
            
            if choice_num in timeframes:
                name, tf, minutes, desc = timeframes[choice_num]
                print(f"\n✅ Timeframe selecionado: {name} ({desc})")
                return tf, minutes, name
            else:
                print(f"❌ Por favor, escolha um número entre 1 e 7.")
        except ValueError:
            print("❌ Por favor, digite um número válido.")
        except KeyboardInterrupt:
            print("\n⚠️  Usando timeframe padrão M1")
            return mt5.TIMEFRAME_M1, 1, 'M1'

# Coletar timeframe do usuário
TIMEFRAME, TIMEFRAME_MINUTES, TIMEFRAME_NAME = escolher_timeframe()

# LOT_SIZE será calculado dinamicamente baseado na gestão de risco

# Janela máxima para manter posição aberta e alvo mínimo de lucro em USD
MAX_HOLD_MINUTES = 30
MIN_PROFIT_USD = 0.0  # Fechar assim que lucro > 0 dentro da janela
# Trailing Stop: protege lucros movendo SL para break-even ou acima quando em lucro
ENABLE_TRAILING_STOP = True  # Habilitar trailing stop para proteger lucros
TRAILING_STOP_ACTIVATION = 0.5  # Ativar trailing stop quando lucro >= 50% do TP
TRAILING_STOP_DISTANCE = 0.3  # Manter SL a 30% do lucro máximo alcançado

# Períodos das Médias Móveis (Valores iniciais, serão otimizados)
MA_SHORT_PERIOD = 3  
MA_LONG_PERIOD = 5  

# Período do RSI (Valores iniciais, serão otimizados)
RSI_PERIOD = 5      
RSI_OVERBOUGHT = 70  
RSI_OVERSOLD = 30   

# --- Parâmetros de Gestão de Risco ---
ATR_PERIOD = 14       # Período usado para calcular o ATR (padrão é 14)
SL_MULTIPLIER = 2.0   # SL será 2.0 * ATR
SL_MULTIPLIER_MIN = 1.0  # Mínimo: 1.0 * ATR (evita SL muito apertado)
SL_MULTIPLIER_MAX = 3.0  # Máximo: 3.0 * ATR (evita SL muito grande)
RR_RATIO = 2.5        # Relação Risco:Recompensa (TP será 2.5 * SL)
RISK_PERCENT = 0.01   # Risco máximo por trade: 1% do capital

# --- Parâmetros de Contrato XAUUSD ---
TICK_SIZE = 0.01           # O menor movimento de preço (XAUUSD é geralmente 0.01)
TICK_VALUE = 0.01          # Valor do TICK (depende da corretora e do ativo)

# --- Configurações de Indicadores Selecionáveis ---
# Indicadores disponíveis para o usuário escolher
AVAILABLE_INDICATORS = {
    'MA_CROSSOVER': 'Cruzamento de Médias Móveis',
    'RSI': 'RSI (Relative Strength Index)',
    'ML_PREDICTION': 'Predição de Machine Learning',
    'STRONG_SIGNALS': 'Sinais Fortes (MA + RSI)'
}

# Configuração inicial dos indicadores (será definida pelo usuário)
SELECTED_INDICATORS = {}

# Sistema de tracking de resultados
TRADING_RESULTS = {
    'total_trades': 0,
    'profitable_trades': 0,
    'losing_trades': 0,
    'total_profit': 0.0,
    'win_rate': 0.0,
    'trades_history': []
}

# --- Configurações de Notícias ---
# Moedas a serem monitoradas para notícias de alto impacto
NEWS_CURRENCIES = ['USD']
# Período de segurança em minutos antes de um evento de alto impacto
SAFETY_PERIOD_MINUTES = 30
# URL do Calendário Econômico do Investing.com
INVESTING_CALENDAR_URL = "https://www.investing.com/economic-calendar/"

# Variável global para armazenar o modelo de ML treinado
ML_MODEL = None
# Arquivo para salvar o modelo treinado para não precisar treinar novamente
ML_MODEL_FILE = "trading_model.pkl"

# --- SISTEMA DE CACHE PARA OTIMIZAÇÃO ---
# Cache de informações do MT5 para reduzir chamadas repetidas
CACHE_SYMBOL_INFO = None
CACHE_ACCOUNT_INFO = None
CACHE_TIMESTAMP = None
CACHE_TTL = 5  # Time-to-live do cache em segundos (5 segundos)

# Cache de dados históricos para evitar buscas repetidas
CACHED_DF = None
CACHED_DF_LAST_UPDATE = None
CACHED_DF_TIMEFRAME = None
CACHED_DF_SYMBOL = None

# Cache de verificação de notícias
CACHE_NEWS_CHECK = None
CACHE_NEWS_TIMESTAMP = None
CACHE_NEWS_TTL = 300  # 5 minutos em segundos

# Cache de posições
CACHE_POSITIONS = None
CACHE_POSITIONS_TIMESTAMP = None
CACHE_POSITIONS_TTL = 2  # 2 segundos

# --- FUNÇÕES DE CACHE OTIMIZADAS ---
def get_cached_symbol_info(symbol, force_refresh=False):
    """Obtém informações do símbolo com cache para reduzir chamadas ao MT5."""
    global CACHE_SYMBOL_INFO, CACHE_TIMESTAMP
    
    now = time.time()
    
    # Se não há cache, está expirado, ou forçado a atualizar
    if (CACHE_SYMBOL_INFO is None or 
        CACHE_TIMESTAMP is None or 
        (now - CACHE_TIMESTAMP) > CACHE_TTL or 
        force_refresh):
        CACHE_SYMBOL_INFO = mt5.symbol_info(symbol)
        CACHE_TIMESTAMP = now
    
    return CACHE_SYMBOL_INFO

def get_cached_account_info(force_refresh=False):
    """Obtém informações da conta com cache para reduzir chamadas ao MT5."""
    global CACHE_ACCOUNT_INFO, CACHE_TIMESTAMP
    
    now = time.time()
    
    # Se não há cache, está expirado, ou forçado a atualizar
    if (CACHE_ACCOUNT_INFO is None or 
        CACHE_TIMESTAMP is None or 
        (now - CACHE_TIMESTAMP) > CACHE_TTL or 
        force_refresh):
        CACHE_ACCOUNT_INFO = mt5.account_info()
        CACHE_TIMESTAMP = now
    
    return CACHE_ACCOUNT_INFO

def get_cached_positions(symbol, force_refresh=False):
    """Obtém posições com cache para reduzir chamadas ao MT5."""
    global CACHE_POSITIONS, CACHE_POSITIONS_TIMESTAMP
    
    now = time.time()
    
    # Se não há cache, está expirado, ou forçado a atualizar
    if (CACHE_POSITIONS is None or 
        CACHE_POSITIONS_TIMESTAMP is None or 
        (now - CACHE_POSITIONS_TIMESTAMP) > CACHE_POSITIONS_TTL or 
        force_refresh):
        CACHE_POSITIONS = mt5.positions_get(symbol=symbol)
        CACHE_POSITIONS_TIMESTAMP = now
    
    return CACHE_POSITIONS if CACHE_POSITIONS else []

def invalidate_cache():
    """Invalida todos os caches (útil após operações de trading)."""
    global CACHE_SYMBOL_INFO, CACHE_ACCOUNT_INFO, CACHE_TIMESTAMP
    global CACHE_POSITIONS, CACHE_POSITIONS_TIMESTAMP
    global CACHED_DF, CACHED_DF_LAST_UPDATE
    
    CACHE_SYMBOL_INFO = None
    CACHE_ACCOUNT_INFO = None
    CACHE_TIMESTAMP = None
    CACHE_POSITIONS = None
    CACHE_POSITIONS_TIMESTAMP = None
    CACHED_DF = None
    CACHED_DF_LAST_UPDATE = None

# --- Funções de Gestão de Risco ---
def calcular_atr_manual(df, period=14):
    """
    Calcula o ATR manualmente caso pandas_ta não esteja disponível.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # ATR = média móvel exponencial do True Range
    atr = true_range.ewm(span=period).mean()
    
    return atr

def calcular_niveis_dinamicos(df, atr_period, sl_mult, rr_ratio, sl_mult_min=None, sl_mult_max=None):
    """
    Calcula o ATR, define o SL e o TP de forma dinâmica.
    Com limites mínimo e máximo para evitar SL muito apertado ou muito grande.
    """
    
    # 1. Calcula o ATR usando método manual
    atr_series = calcular_atr_manual(df, atr_period)
    atr_value = atr_series.iloc[-1]
    
    # 2. Define o SL (em termos de variação de preço)
    # SL = Multiplicador * ATR, mas com limites
    sl_distance_base = atr_value * sl_mult
    
    # Aplicar limites se fornecidos
    if sl_mult_min is not None:
        sl_distance_min = atr_value * sl_mult_min
        sl_distance_base = max(sl_distance_base, sl_distance_min)
    
    if sl_mult_max is not None:
        sl_distance_max = atr_value * sl_mult_max
        sl_distance_base = min(sl_distance_base, sl_distance_max)
    
    sl_distance = sl_distance_base
    
    # 3. Define o TP (em termos de variação de preço)
    # TP = Relação R:R * SL
    tp_distance = sl_distance * rr_ratio
    
    return sl_distance, tp_distance, atr_value

def calcular_posicao_e_risco(sl_distance, account_balance, risk_percent, tick_size, tick_value, symbol=SYMBOL):
    """
    Calcula o tamanho do lote com base no risco percentual da conta e na distância do SL.
    Usa informações reais do MT5 para cálculo preciso.
    Também verifica margem disponível para garantir que a ordem possa ser executada.
    """
    
    # 1. Define o Risco em Moeda
    risco_por_trade = account_balance * risk_percent
    
    # 2. Obter informações reais do símbolo do MT5 (com cache)
    symbol_info = get_cached_symbol_info(symbol)
    if symbol_info is None:
        print(f"⚠️  Não foi possível obter informações do símbolo {symbol}, usando valores padrão")
        # Fallback: usar fórmula simplificada
        lotagem = risco_por_trade / (sl_distance * 100)
        lotagem_final = np.floor(lotagem * 100) / 100
        return max(0.01, lotagem_final)
    
    # 3. Obter informações da conta para verificar margem (com cache)
    account_info = get_cached_account_info()
    if account_info is None:
        print(f"⚠️  Não foi possível obter informações da conta, continuando sem verificação de margem...")
        margin_free = float('inf')  # Não limitar se não conseguir obter
    else:
        margin_free = account_info.margin_free
    
    # 4. Obter preço atual para cálculo de margem
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"⚠️  Não foi possível obter tick atual, usando cálculo sem verificação de margem...")
        current_price = None
    else:
        # Usar ask como referência (será ajustado na ordem real)
        current_price = tick.ask
    
    # 3. Calcular o valor do ponto (tick value) real
    # Para XAUUSD: 1 lote padrão = 100 onças
    # Cada movimento de 0.01 (1 pip) = $1 por lote padrão
    # Mas vamos usar o valor real do contrato do MT5
    contract_size = symbol_info.trade_contract_size  # Geralmente 100 para XAUUSD
    tick_value_real = symbol_info.trade_tick_value  # Valor do tick em moeda da conta
    
    # 4. Calcular número de pips até o SL
    sl_ticks = sl_distance / tick_size
    
    # 5. Calcular perda por lote padrão até o SL
    # Perda por lote = (distância em ticks) * (valor do tick por lote)
    # tick_value_real já está por lote padrão, então:
    perda_por_lote = sl_ticks * tick_value_real
    
    # 6. Calcular tamanho do lote necessário
    # Lotagem = Risco Máximo / Perda por Lote
    if perda_por_lote > 0:
        lotagem = risco_por_trade / perda_por_lote
    else:
        # Fallback se perda_por_lote for zero ou negativo
        print(f"⚠️  Perda por lote inválida ({perda_por_lote}), usando fórmula simplificada")
        lotagem = risco_por_trade / (sl_distance * 100)
    
    # 7. Ajustar para múltiplo do lote mínimo
    lot_step = symbol_info.volume_step  # Passo mínimo do lote (geralmente 0.01)
    lot_min = symbol_info.volume_min    # Lote mínimo permitido
    
    # Arredondar para baixo ao múltiplo mais próximo do passo
    lotagem_final = np.floor(lotagem / lot_step) * lot_step
    
    # Garantir que seja pelo menos o mínimo
    lotagem_final = max(lot_min, lotagem_final)
    
    # 8. Validação: recalcular risco real para garantir que não exceda o desejado
    risco_real = lotagem_final * perda_por_lote
    if risco_real > risco_por_trade * 1.1:  # Se exceder 10% do risco desejado, reduzir lote
        lotagem_final = np.floor((risco_por_trade * 0.95) / perda_por_lote / lot_step) * lot_step
        lotagem_final = max(lot_min, lotagem_final)
        risco_real = lotagem_final * perda_por_lote
    
    # 9. VERIFICAR MARGEM DISPONÍVEL (se possível)
    if current_price is not None and margin_free != float('inf'):
        # Calcular margem necessária para BUY (usar como referência)
        margin_required = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, lotagem_final, current_price)
        if margin_required is not None:
            if margin_required > margin_free:
                # Reduzir lote para caber na margem disponível
                lotagem_max_por_margem = (margin_free * 0.95) / (margin_required / lotagem_final) if margin_required > 0 else lotagem_final
                lotagem_max_por_margem = np.floor(lotagem_max_por_margem / lot_step) * lot_step
                lotagem_max_por_margem = max(lot_min, lotagem_max_por_margem)
                
                if lotagem_max_por_margem < lot_min:
                    print(f"⚠️  ATENÇÃO: Margem insuficiente! Lote calculado: {lotagem_final:.2f}, mas margem só permite: {lotagem_max_por_margem:.2f}")
                    print(f"   Margem livre: ${margin_free:.2f}, Necessária: ${margin_required:.2f}")
                    # Retornar o máximo possível por margem, mesmo que seja menor que o desejado
                    return lotagem_max_por_margem
                elif lotagem_max_por_margem < lotagem_final:
                    print(f"⚠️  Lote reduzido por margem: {lotagem_final:.2f} → {lotagem_max_por_margem:.2f}")
                    lotagem_final = lotagem_max_por_margem
    
    return lotagem_final

def gerenciar_risco(df, preco_entrada, side="BUY"):
    """
    Calcula os níveis de SL e TP para uma nova entrada.
    :param df: DataFrame com dados OHLC para calcular ATR
    :param preco_entrada: O preço atual para a ordem de entrada.
    :param side: "BUY" ou "SELL".
    """
    
    # 1. Obter saldo da conta MT5 (com cache)
    account_info = get_cached_account_info()
    if account_info is None:
        print("❌ Erro ao obter informações da conta")
        return None
    
    account_balance = account_info.balance
    
    # 2. Calcular Níveis Dinâmicos (com limites para evitar SL muito grande ou pequeno)
    sl_distancia, tp_distancia, atr_valor = calcular_niveis_dinamicos(
        df, ATR_PERIOD, SL_MULTIPLIER, RR_RATIO, SL_MULTIPLIER_MIN, SL_MULTIPLIER_MAX
    )
    
    # 3. Calcular Tamanho da Posição (Lotagem) usando informações reais do MT5
    lote_calculado = calcular_posicao_e_risco(
        sl_distancia, account_balance, RISK_PERCENT, TICK_SIZE, TICK_VALUE, SYMBOL
    )
    
    # 4. Definir Preços Reais de SL e TP
    
    # Calcula o preço do SL
    if side == "BUY":
        # SL para Compra = Preço de Entrada - Distância SL
        preco_sl = preco_entrada - sl_distancia
        # TP para Compra = Preço de Entrada + Distância TP
        preco_tp = preco_entrada + tp_distancia
    else: # SELL
        # SL para Venda = Preço de Entrada + Distância SL
        preco_sl = preco_entrada + sl_distancia
        # TP para Venda = Preço de Entrada - Distância TP
        preco_tp = preco_entrada - tp_distancia

    # Arredondar os preços para a precisão do XAUUSD (geralmente 2 casas decimais)
    preco_sl = round(preco_sl, 2)
    preco_tp = round(preco_tp, 2)
    
    # Verificar se os preços são válidos e logicamente corretos
    if preco_sl <= 0 or preco_tp <= 0:
        print(f"⚠️  Preços SL/TP inválidos calculados. Usando valores mínimos seguros.")
        preco_sl = max(preco_sl, 1000.0)  # Preço mínimo seguro para XAUUSD
        preco_tp = max(preco_tp, 1000.0)  # Preço mínimo seguro para XAUUSD
    
    # Verificar lógica dos preços SL/TP
    if side == "BUY":
        # Para compra: TP deve ser > preço atual > SL
        if preco_tp <= preco_entrada:
            print(f"⚠️  TP inválido para BUY. Ajustando...")
            preco_tp = preco_entrada + (preco_entrada - preco_sl) * 2  # TP = entrada + 2x distância do SL
        if preco_sl >= preco_entrada:
            print(f"⚠️  SL inválido para BUY. Ajustando...")
            preco_sl = preco_entrada - sl_distancia
    else: # SELL
        # Para venda: SL deve ser > preço atual > TP
        if preco_sl <= preco_entrada:
            print(f"⚠️  SL inválido para SELL. Ajustando...")
            preco_sl = preco_entrada + sl_distancia
        if preco_tp >= preco_entrada:
            print(f"⚠️  TP inválido para SELL. Ajustando...")
            preco_tp = preco_entrada - (preco_sl - preco_entrada) * 2  # TP = entrada - 2x distância do SL
    
    # Arredondar novamente após ajustes
    preco_sl = round(preco_sl, 2)
    preco_tp = round(preco_tp, 2)
    
    # 5. Calcular risco real para validação (com cache)
    symbol_info = get_cached_symbol_info(SYMBOL)
    risco_real_usd = 0.0
    if symbol_info:
        sl_ticks_real = abs(preco_entrada - preco_sl) / TICK_SIZE
        tick_value_real = symbol_info.trade_tick_value
        perda_por_lote = sl_ticks_real * tick_value_real
        risco_real_usd = lote_calculado * perda_por_lote
        risco_real_percent = (risco_real_usd / account_balance) * 100 if account_balance > 0 else 0
    else:
        # Fallback
        risco_real_usd = account_balance * RISK_PERCENT
        risco_real_percent = RISK_PERCENT * 100
    
    # 6. Verificar se SL está dentro dos limites aceitáveis
    sl_mult_efetivo = sl_distancia / atr_valor if atr_valor > 0 else SL_MULTIPLIER
    sl_aviso = ""
    if sl_mult_efetivo > SL_MULTIPLIER_MAX * 0.9:
        sl_aviso = "⚠️  SL próximo do limite máximo!"
    elif sl_mult_efetivo < SL_MULTIPLIER_MIN * 1.1:
        sl_aviso = "⚠️  SL próximo do limite mínimo!"
    
    # --- RELATÓRIO DE RISCO MELHORADO ---
    print("--- RELATÓRIO DE GERENCIAMENTO DE RISCO DINÂMICO ---")
    print(f"Sinal de Entrada: {side} @ {preco_entrada:.2f}")
    print(f"Saldo da Conta: ${account_balance:.2f}")
    print(f"---------------------------------------------------")
    print(f"Volatilidade ATR ({ATR_PERIOD}): {atr_valor:.4f}")
    print(f"SL Multiplicador: {sl_mult_efetivo:.2f}x ATR (Limites: {SL_MULTIPLIER_MIN:.1f} - {SL_MULTIPLIER_MAX:.1f}) {sl_aviso}")
    print(f"SL Distância: {sl_distancia:.4f} ({sl_distancia/TICK_SIZE:.0f} pips)")
    print(f"TP Distância: {tp_distancia:.4f} (R:R {RR_RATIO}:1)")
    print(f"---------------------------------------------------")
    print(f"Risco Desejado: {RISK_PERCENT*100:.2f}% (${account_balance * RISK_PERCENT:.2f})")
    print(f"Risco Real Calculado: {risco_real_percent:.2f}% (${risco_real_usd:.2f})")
    if abs(risco_real_percent - RISK_PERCENT * 100) > 0.5:
        print(f"⚠️  ATENÇÃO: Risco real difere do desejado em {abs(risco_real_percent - RISK_PERCENT * 100):.2f}%")
    print(f"Lote calculado: {lote_calculado:.2f}")
    print(f"Preço SL: {preco_sl:.2f} (Perda máxima: ${risco_real_usd:.2f})")
    print(f"Preço TP: {preco_tp:.2f} (Ganho potencial: ${risco_real_usd * RR_RATIO:.2f})")
    print("---------------------------------------------------")

    return {
        "lote": lote_calculado,
        "sl": preco_sl,
        "tp": preco_tp,
        "atr": atr_valor
    }

# --- Funções para Seleção de Indicadores e Relatórios ---
def select_indicators():
    """Permite ao usuário selecionar quais indicadores usar para as entradas."""
    print("\n🎯 --- SELEÇÃO DE INDICADORES DE TRADING ---")
    print("Escolha quais indicadores deseja utilizar para gerar sinais de entrada:")
    print()
    
    for i, (key, description) in enumerate(AVAILABLE_INDICATORS.items(), 1):
        print(f"{i}. {description}")
    
    print("\nDigite os números dos indicadores que deseja usar (separados por vírgula):")
    print("Exemplo: 1,2,4 (para usar MA Crossover, RSI e Sinais Fortes)")
    print("Ou digite 'all' para usar todos os indicadores")
    
    while True:
        try:
            choice = input("\nSua escolha: ").strip().lower()
            
            if choice == 'all':
                SELECTED_INDICATORS.update(AVAILABLE_INDICATORS)
                print("\n✅ Todos os indicadores selecionados!")
                break
            
            # Parse da entrada do usuário
            selected_numbers = [int(x.strip()) for x in choice.split(',')]
            
            if not selected_numbers:
                print("❌ Por favor, digite pelo menos um número.")
                continue
            
            if any(num < 1 or num > len(AVAILABLE_INDICATORS) for num in selected_numbers):
                print(f"❌ Por favor, digite números entre 1 e {len(AVAILABLE_INDICATORS)}.")
                continue
            
            # Adicionar indicadores selecionados
            indicator_keys = list(AVAILABLE_INDICATORS.keys())
            for num in selected_numbers:
                key = indicator_keys[num - 1]
                SELECTED_INDICATORS[key] = AVAILABLE_INDICATORS[key]
            
            print(f"\n✅ Indicadores selecionados:")
            for key, description in SELECTED_INDICATORS.items():
                print(f"   - {description}")
            
            if not SELECTED_INDICATORS:
                print("❌ Nenhum indicador selecionado. Por favor, selecione pelo menos um.")
                continue
                
            break
            
        except ValueError:
            print("❌ Por favor, digite apenas números separados por vírgula.")
        except KeyboardInterrupt:
            print("\n❌ Seleção cancelada pelo usuário.")
            return False
    
    return True

def update_trading_results(trade_type, profit_loss, entry_price, exit_price, indicators_used):
    """Atualiza o sistema de tracking de resultados."""
    TRADING_RESULTS['total_trades'] += 1
    TRADING_RESULTS['total_profit'] += profit_loss
    
    trade_record = {
        'trade_number': TRADING_RESULTS['total_trades'],
        'type': trade_type,
        'profit_loss': profit_loss,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'indicators_used': indicators_used,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    TRADING_RESULTS['trades_history'].append(trade_record)
    
    if profit_loss > 0:
        TRADING_RESULTS['profitable_trades'] += 1
    else:
        TRADING_RESULTS['losing_trades'] += 1
    
    # Calcular win rate
    if TRADING_RESULTS['total_trades'] > 0:
        TRADING_RESULTS['win_rate'] = (TRADING_RESULTS['profitable_trades'] / TRADING_RESULTS['total_trades']) * 100

def print_trading_summary():
    """Imprime um resumo dos resultados de trading."""
    print("\n📊 --- RESUMO DOS RESULTADOS DE TRADING ---")
    print(f"Total de Operações: {TRADING_RESULTS['total_trades']}")
    print(f"Operações Lucrativas: {TRADING_RESULTS['profitable_trades']}")
    print(f"Operações Perdedoras: {TRADING_RESULTS['losing_trades']}")
    print(f"Taxa de Acerto: {TRADING_RESULTS['win_rate']:.1f}%")
    print(f"Lucro Total: ${TRADING_RESULTS['total_profit']:.2f}")
    
    if TRADING_RESULTS['total_trades'] > 0:
        avg_profit = TRADING_RESULTS['total_profit'] / TRADING_RESULTS['total_trades']
        print(f"Lucro Médio por Operação: ${avg_profit:.2f}")
    
    # Indicadores mais utilizados
    if TRADING_RESULTS['trades_history']:
        indicator_usage = {}
        for trade in TRADING_RESULTS['trades_history']:
            for indicator in trade['indicators_used']:
                indicator_usage[indicator] = indicator_usage.get(indicator, 0) + 1
        
        print(f"\nIndicadores mais utilizados:")
        for indicator, count in sorted(indicator_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {indicator}: {count} operações")


def calcular_tempo_proxima_barra(timeframe_minutes):
    """
    Calcula quantos segundos faltam até a próxima barra fechar.
    
    Args:
        timeframe_minutes: Número de minutos do timeframe (ex: 1 para M1, 5 para M5)
    
    Returns:
        Número de segundos até a próxima barra fechar
    """
    now = datetime.now()
    
    # Para timeframes menores que 1 hora
    if timeframe_minutes < 60:
        # Calcular quantos segundos se passaram no minuto atual
        seconds_passed = now.second + (now.microsecond / 1000000)
        # Calcular quantos segundos faltam para completar o minuto atual
        seconds_remaining_in_minute = 60 - seconds_passed
        
        # Se o timeframe é múltiplo de minutos (M5, M15, M30)
        if timeframe_minutes > 1:
            # Calcular em qual "slot" de minutos estamos
            current_minute = now.minute
            # Calcular quantos minutos faltam para o próximo slot
            minutes_until_next = timeframe_minutes - (current_minute % timeframe_minutes)
            
            if minutes_until_next == timeframe_minutes:
                # Já estamos no início de um slot, próxima barra será no próximo slot
                return seconds_remaining_in_minute + (timeframe_minutes - 1) * 60
            else:
                # Próxima barra será no próximo slot
                return seconds_remaining_in_minute + (minutes_until_next - 1) * 60
        else:
            # M1: próxima barra fecha no próximo minuto
            return seconds_remaining_in_minute
    else:
        # Para timeframes de horas ou mais
        hours = timeframe_minutes // 60
        current_hour = now.hour
        current_minute = now.minute
        
        # Calcular quantas horas faltam para o próximo slot
        hours_until_next = hours - (current_hour % hours)
        
        if hours_until_next == hours and current_minute == 0:
            # Já estamos no início de um slot
            return 60 - (now.second + now.microsecond / 1000000) + (hours - 1) * 3600
        else:
            # Calcular minutos e segundos restantes
            minutes_remaining = 60 - current_minute
            seconds_remaining = 60 - (now.second + now.microsecond / 1000000)
            return seconds_remaining + (minutes_remaining - 1) * 60 + (hours_until_next - 1) * 3600

def evaluate_signals(df, last_ma_short, last_ma_long, prev_ma_short, prev_ma_long, last_rsi, ml_signal):
    """Avalia os sinais baseado nos indicadores selecionados pelo usuário."""
    signals = {
        'buy': False,
        'sell': False,
        'indicators_triggered': []
    }
    
    # MA Crossover
    if 'MA_CROSSOVER' in SELECTED_INDICATORS:
        buy_signal_ma = last_ma_short > last_ma_long and prev_ma_short <= prev_ma_long
        sell_signal_ma = last_ma_short < last_ma_long and prev_ma_short >= prev_ma_long
        
        if buy_signal_ma:
            signals['buy'] = True
            signals['indicators_triggered'].append('MA_CROSSOVER')
        elif sell_signal_ma:
            signals['sell'] = True
            signals['indicators_triggered'].append('MA_CROSSOVER')
    
    # RSI
    if 'RSI' in SELECTED_INDICATORS:
        rsi_ok = RSI_OVERSOLD < last_rsi < RSI_OVERBOUGHT
        if rsi_ok:
            if signals['buy'] or signals['sell']:
                signals['indicators_triggered'].append('RSI')
    
    # ML Prediction
    if 'ML_PREDICTION' in SELECTED_INDICATORS and ML_MODEL is not None:
        if ml_signal == "COMPRA (UP)":
            if not signals['sell']:  # Só adiciona se não há sinal contrário
                signals['buy'] = True
                signals['indicators_triggered'].append('ML_PREDICTION')
        elif ml_signal == "VENDA (DOWN)":
            if not signals['buy']:  # Só adiciona se não há sinal contrário
                signals['sell'] = True
                signals['indicators_triggered'].append('ML_PREDICTION')
    
    # Strong Signals
    if 'STRONG_SIGNALS' in SELECTED_INDICATORS:
        strong_buy = (last_ma_short > last_ma_long * 1.001) and (last_rsi < 40)
        strong_sell = (last_ma_short < last_ma_long * 0.999) and (last_rsi > 60)
        
        if strong_buy:
            signals['buy'] = True
            signals['indicators_triggered'].append('STRONG_SIGNALS')
        elif strong_sell:
            signals['sell'] = True
            signals['indicators_triggered'].append('STRONG_SIGNALS')
    
    return signals

# Função para testar conexão e extração de dados
def test_mt5_connection():
    """Testa a conexão com MT5 e a extração de dados."""
    print("\n🧪 --- TESTE DE CONEXÃO E EXTRAÇÃO DE DADOS ---")
    
    # Teste 1: Conexão
    print("\n1️⃣ Testando conexão com MT5...")
    if not connect_mt5():
        print("❌ Falha na conexão com MT5")
        return False
    
    print("✅ Conexão com MT5 estabelecida")
    
    # Teste 2: Informações do símbolo
    print("\n2️⃣ Verificando informações do símbolo...")
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"❌ Símbolo {SYMBOL} não encontrado")
        return False
    
    print(f"✅ Símbolo {SYMBOL} encontrado:")
    print(f"   - Nome: {symbol_info.name}")
    print(f"   - Bid: {symbol_info.bid}")
    print(f"   - Ask: {symbol_info.ask}")
    print(f"   - Spread: {symbol_info.ask - symbol_info.bid}")
    
    # Teste 3: Extração de dados
    print("\n3️⃣ Testando extração de dados...")
    test_df = get_ohlc_data(SYMBOL, TIMEFRAME, 100)
    
    if test_df.empty:
        print("❌ Método original falhou, tentando método simples...")
        test_df = get_ohlc_data_simple(SYMBOL, TIMEFRAME, 100)
        
        if test_df.empty:
            print("❌ Falha na extração de dados")
            return False
        else:
            print("✅ Dados obtidos com método simples")
    else:
        print("✅ Dados obtidos com método original")
    
    print(f"✅ Dados extraídos com sucesso:")
    print(f"   - Barras obtidas: {len(test_df)}")
    print(f"   - Período: {test_df.index.min().strftime('%Y-%m-%d %H:%M')} até {test_df.index.max().strftime('%Y-%m-%d %H:%M')}")
    print(f"   - Colunas: {list(test_df.columns)}")
    
    # Teste 4: Cálculo de indicadores
    print("\n4️⃣ Testando cálculo de indicadores...")
    test_df_with_indicators = calculate_indicators(test_df.copy(), MA_SHORT_PERIOD, MA_LONG_PERIOD, RSI_PERIOD)
    
    if test_df_with_indicators.empty:
        print("❌ Falha no cálculo de indicadores")
        return False
    
    print(f"✅ Indicadores calculados com sucesso:")
    print(f"   - MA Short: {test_df_with_indicators['MA_Short'].iloc[-1]:.5f}")
    print(f"   - MA Long: {test_df_with_indicators['MA_Long'].iloc[-1]:.5f}")
    print(f"   - RSI: {test_df_with_indicators['RSI'].iloc[-1]:.2f}")
    
    print("\n🎉 Todos os testes passaram! MT5 está funcionando corretamente.")
    return True

# Função para verificar consistência dos dados
def verify_data_consistency(df, source_name):
    """Verifica se os dados são consistentes e adequados para análise."""
    if df.empty:
        print(f"❌ {source_name}: DataFrame vazio")
        return False
    
    # Verificar colunas essenciais
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ {source_name}: Colunas faltando: {missing_columns}")
        return False
    
    # Verificar dados válidos
    invalid_data = df[required_columns].isnull().sum().sum()
    if invalid_data > 0:
        print(f"⚠️  {source_name}: {invalid_data} valores nulos encontrados")
    
    # Verificar se há dados suficientes
    if len(df) < 50:
        print(f"❌ {source_name}: Dados insuficientes ({len(df)} barras)")
        return False
    
    print(f"✅ {source_name}: {len(df)} barras válidas")
    return True

# --- Funções de Conexão e Análise Técnica ---
def connect_mt5():
    """Tenta inicializar e conectar-se ao MetaTrader 5."""
    print("🔌 Tentando conectar ao MetaTrader 5...")
    
    # Inicializar MT5
    if not mt5.initialize():
        print(f"❌ Erro ao inicializar MT5: {mt5.last_error()}")
        print("💡 Verifique se o MetaTrader 5 está instalado e aberto")
        return False
    
    print("✅ MT5 inicializado com sucesso")
    
    # Tentar login
    try:
        authorized = mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if not authorized:
            print(f"❌ Erro ao conectar na conta MT5 #{MT5_LOGIN}: {mt5.last_error()}")
            print("💡 Verifique suas credenciais de login")
            mt5.shutdown()
            return False
    except Exception as e:
        print(f"❌ Erro durante login: {e}")
        mt5.shutdown()
        return False
    
    print(f"✅ Login realizado com sucesso na conta: {MT5_LOGIN}")
    
    # Verificar se o símbolo está disponível
    symbol_info = mt5.symbol_info(SYMBOL)
    if symbol_info is None:
        print(f"❌ Símbolo {SYMBOL} não encontrado no MT5")
        print("💡 Verifique se o símbolo está disponível na sua conta")
        mt5.shutdown()
        return False
    
    if not symbol_info.visible:
        print(f"⚠️  Símbolo {SYMBOL} não está visível, tentando adicionar...")
        if not mt5.symbol_select(SYMBOL, True):
            print(f"❌ Não foi possível adicionar o símbolo {SYMBOL}")
            mt5.shutdown()
            return False
    
    print(f"✅ Símbolo {SYMBOL} disponível para negociação")
    
    # Verificar permissões de trading
    terminal_info = mt5.terminal_info()
    if terminal_info is None:
        print("⚠️  Não foi possível obter informações do terminal")
    elif not terminal_info.trade_allowed:
        print("⚠️  ATENÇÃO: AutoTrading está DESABILITADO no MetaTrader 5!")
        print("💡 Habilite o AutoTrading no MT5 para executar trades")
    else:
        print("✅ AutoTrading está habilitado e funcionando!")
    
    print(f"🎯 Conectado ao MetaTrader 5 com sucesso! Conta: {MT5_LOGIN}")
    return True

def get_web_data(symbol, days=30):
    """Busca dados históricos da web para cálculos imediatos dos indicadores."""
    try:
        symbol_mapping = {
            "XAUUSD": "GC=F",
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            "USDJPY": "USDJPY=X",
        }
        
        yf_symbol = symbol_mapping.get(symbol, symbol)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(yf_symbol)
        
        # Estratégia inteligente para obter dados suficientes
        df = pd.DataFrame()
        
        # 1. Tenta dados de 5 minutos (ideal para análise técnica)
        try:
            df = ticker.history(start=start_date, end=end_date, interval="5m")
            if len(df) >= 100:  # Verifica se tem dados suficientes
                print(f"✅ Dados de 5 minutos obtidos: {len(df)} barras")
                interval_used = "5m"
            else:
                df = pd.DataFrame()  # Reset se dados insuficientes
                print(f"⚠️  Dados de 5 minutos insuficientes: {len(df)} barras")
        except Exception as e:
            print(f"⚠️  Erro ao buscar dados de 5 minutos: {e}")
        
        # 2. Se 5m falhar ou for insuficiente, tenta 1 hora
        if df.empty or len(df) < 100:
            try:
                df = ticker.history(start=start_date, end=end_date, interval="1h")
                if len(df) >= 50:  # Mínimo para análise
                    print(f"✅ Dados de 1 hora obtidos: {len(df)} barras")
                    interval_used = "1h"
                else:
                    df = pd.DataFrame()
                    print(f"⚠️  Dados de 1 hora insuficientes: {len(df)} barras")
            except Exception as e:
                print(f"⚠️  Erro ao buscar dados de 1 hora: {e}")
        
        # 3. Último recurso: dados diários
        if df.empty or len(df) < 50:
            try:
                df = ticker.history(start=start_date, end=end_date, interval="1d")
                if len(df) >= 20:  # Mínimo absoluto
                    print(f"✅ Dados diários obtidos: {len(df)} barras")
                    interval_used = "1d"
                else:
                    print(f"❌ Dados insuficientes mesmo com timeframe diário: {len(df)} barras")
                    return pd.DataFrame()
            except Exception as e:
                print(f"❌ Erro ao buscar dados diários: {e}")
                return pd.DataFrame()
        
        if df.empty:
            print(f"❌ Erro: Nenhum dado encontrado para {yf_symbol}")
            return pd.DataFrame()
        
        # Processa os dados
        if 'Open' in df.columns and 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            df_selected = df[['Open', 'High', 'Low', 'Close']].copy()
            df_selected.columns = ['open', 'high', 'low', 'close']
        else:
            print("❌ Colunas OHLC não encontradas nos dados")
            return pd.DataFrame()
        
        if 'Volume' in df.columns:
            df_selected['volume'] = df['Volume']
        else:
            df_selected['volume'] = 0
        
        df_selected = df_selected.reset_index()
        df_selected['time'] = df_selected['Datetime']
        df_selected = df_selected.set_index('time')
        
        print(f"🎯 Dados finais: {len(df_selected)} barras de {interval_used}")
        return df_selected
        
    except Exception as e:
        print(f"❌ Erro geral ao buscar dados da web: {e}")
        return pd.DataFrame()

def get_ohlc_data(symbol, timeframe, count):
    """Obtém dados históricos de OHLC do MT5 com melhor tratamento de erros."""
    try:
        print(f"📊 Obtendo {count} barras de {symbol} do MT5...")
        
        # Verificar se o MT5 está conectado
        if not mt5.terminal_info():
            print("❌ MT5 não está conectado")
            return pd.DataFrame()
        
        # Verificar se o símbolo está disponível
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Símbolo {symbol} não encontrado no MT5")
            return pd.DataFrame()
        
        # Calcular período de tempo baseado no timeframe
        timeframe_minutes = {
            mt5.TIMEFRAME_M1: 1,
            mt5.TIMEFRAME_M5: 5,
            mt5.TIMEFRAME_M15: 15,
            mt5.TIMEFRAME_M30: 30,
            mt5.TIMEFRAME_H1: 60,
            mt5.TIMEFRAME_H4: 240,
            mt5.TIMEFRAME_D1: 1440
        }
        
        minutes_per_bar = timeframe_minutes.get(timeframe, 5)  # Default 5 minutos
        total_minutes = count * minutes_per_bar
        
        # Calcular data de início - CORREÇÃO DO TIMEZONE
        utc_now = datetime.now()  # Sem timezone primeiro
        utc_from = utc_now - timedelta(minutes=total_minutes * 2)  # Duplicar para garantir dados suficientes
        
        print(f"🔍 Buscando dados de {utc_from.strftime('%Y-%m-%d %H:%M')} até {utc_now.strftime('%Y-%m-%d %H:%M')}")
        
        # Obter dados do MT5
        rates = mt5.copy_rates_from(symbol, timeframe, utc_from, count)
        
        if rates is None or len(rates) == 0:
            print(f"❌ Erro ao buscar dados do {symbol}: {mt5.last_error()}")
            return pd.DataFrame()

        # Converter para DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Verificar se os dados são válidos
        if len(df) < count * 0.3:  # Se menos de 30% dos dados solicitados
            print(f"⚠️  Aviso: Apenas {len(df)} barras obtidas de {count} solicitadas")
            if len(df) < 50:  # Mínimo absoluto
                print("❌ Dados insuficientes para análise")
                return pd.DataFrame()
        else:
            print(f"✅ {len(df)} barras obtidas com sucesso")
        
        # Verificar se há dados muito antigos (mais de 30 dias) - CORREÇÃO DO TIMEZONE
        oldest_date = df.index.min()
        if oldest_date is not None:
            # Converter para datetime sem timezone para comparação
            oldest_date_naive = oldest_date.replace(tzinfo=None)
            if (utc_now - oldest_date_naive).days > 30:
                print(f"⚠️  Aviso: Dados muito antigos encontrados (mais antigo: {oldest_date})")
        
        return df
        
    except Exception as e:
        print(f"❌ Erro inesperado ao obter dados do MT5: {e}")
        print(f"💡 Detalhes do erro: {type(e).__name__}")
        return pd.DataFrame()

def get_ohlc_data_simple(symbol, timeframe, count, use_cache=True):
    """
    Versão simplificada para obter dados do MT5 sem problemas de timezone.
    Otimizada com cache para adicionar apenas novas barras.
    """
    global CACHED_DF, CACHED_DF_LAST_UPDATE, CACHED_DF_TIMEFRAME, CACHED_DF_SYMBOL
    
    try:
        # Verificar se podemos usar cache
        if (use_cache and 
            CACHED_DF is not None and 
            CACHED_DF_SYMBOL == symbol and 
            CACHED_DF_TIMEFRAME == timeframe and
            CACHED_DF_LAST_UPDATE is not None):
            
            # Verificar se cache é recente (menos de 10 segundos)
            time_since_update = (datetime.now() - CACHED_DF_LAST_UPDATE).total_seconds()
            if time_since_update < 10:
                # Tentar adicionar apenas novas barras
                try:
                    # Buscar apenas as últimas barras para verificar se há novas
                    new_rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, 5)
                    if new_rates is not None and len(new_rates) > 0:
                        new_df = pd.DataFrame(new_rates)
                        new_df['time'] = pd.to_datetime(new_df['time'], unit='s')
                        new_df.set_index('time', inplace=True)
                        
                        # Verificar se há novas barras
                        last_cached_time = CACHED_DF.index[-1] if len(CACHED_DF) > 0 else None
                        new_bars = new_df[new_df.index > last_cached_time] if last_cached_time else new_df
                        
                        if len(new_bars) > 0:
                            # Adicionar apenas novas barras ao cache
                            CACHED_DF = pd.concat([CACHED_DF, new_bars]).drop_duplicates(keep='last').sort_index()
                            # Manter apenas as últimas 'count' barras
                            if len(CACHED_DF) > count:
                                CACHED_DF = CACHED_DF.tail(count)
                            CACHED_DF_LAST_UPDATE = datetime.now()
                            print(f"✅ Cache atualizado: {len(new_bars)} novas barras adicionadas (Total: {len(CACHED_DF)})")
                            return CACHED_DF.tail(count).copy()
                        else:
                            # Sem novas barras, retornar cache
                            print(f"✅ Usando cache: {len(CACHED_DF)} barras (sem novas barras)")
                            return CACHED_DF.tail(count).copy()
                except Exception as e:
                    # Se falhar ao atualizar cache, buscar dados completos
                    print(f"⚠️  Erro ao atualizar cache, buscando dados completos: {e}")
        
        # Buscar dados completos (cache não disponível ou expirado)
        print(f"📊 Obtendo {count} barras de {symbol} do MT5 (método simples)...")
        
        # Verificar se o MT5 está conectado
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print("❌ MT5 não está conectado")
            return pd.DataFrame()
        
        # Verificar se o símbolo está disponível (com cache)
        symbol_info = get_cached_symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Símbolo {symbol} não encontrado no MT5")
            return pd.DataFrame()
        
        # Usar copy_rates_from_pos que não precisa de data específica
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            print(f"❌ Erro ao buscar dados do {symbol}: {mt5.last_error()}")
            return pd.DataFrame()

        # Converter para DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Verificar se os dados são válidos
        if len(df) < count * 0.3:  # Se menos de 30% dos dados solicitados
            print(f"⚠️  Aviso: Apenas {len(df)} barras obtidas de {count} solicitadas")
            if len(df) < 50:  # Mínimo absoluto
                print("❌ Dados insuficientes para análise")
                return pd.DataFrame()
        else:
            print(f"✅ {len(df)} barras obtidas com sucesso")
        
        # Atualizar cache
        CACHED_DF = df.copy()
        CACHED_DF_LAST_UPDATE = datetime.now()
        CACHED_DF_TIMEFRAME = timeframe
        CACHED_DF_SYMBOL = symbol
        
        return df
        
    except Exception as e:
        print(f"❌ Erro inesperado ao obter dados do MT5 (método simples): {e}")
        print(f"💡 Detalhes do erro: {type(e).__name__}")
        return pd.DataFrame()

def calculate_indicators(df, ma_short_period, ma_long_period, rsi_period):
    """Calcula as Médias Móveis e o RSI no DataFrame com parâmetros dinâmicos."""
    if df.empty:
        return df

    df['MA_Short'] = df['close'].rolling(window=ma_short_period).mean()
    df['MA_Long'] = df['close'].rolling(window=ma_long_period).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df_clean = df.dropna()
    
    return df_clean

# --- NOVO BLOCO: FUNÇÕES PARA O MACHINE LEARNING ---
def create_features(df):
    """Cria features técnicas para o modelo de ML."""
    df['MA_Short'] = df['close'].rolling(window=MA_SHORT_PERIOD).mean()
    df['MA_Long'] = df['close'].rolling(window=MA_LONG_PERIOD).mean()
    df['MA_Crossover_Signal'] = 0
    df.loc[df['MA_Short'] > df['MA_Long'], 'MA_Crossover_Signal'] = 1
    df.loc[df['MA_Short'] < df['MA_Long'], 'MA_Crossover_Signal'] = -1
    
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def create_target(df, prediction_horizon=15):
    """Cria a variável alvo para o ML. 1 se o preço subir, 0 se descer."""
    df['future_close'] = df['close'].shift(-prediction_horizon)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    return df

def train_ml_model(df):
    """Treina um modelo de Machine Learning usando dados do MT5."""
    print("--- Treinando o modelo de Machine Learning ---")
    print("🎯 Usando dados do MetaTrader 5 para treinar o modelo...")
    
    if df.empty:
        print("❌ Erro: DataFrame vazio para treinamento")
        return None
    
    # Criar features e target usando os mesmos parâmetros que o robô usará
    df_ml = create_features(df.copy())
    df_ml = create_target(df_ml)
    
    df_ml = df_ml.dropna()
    
    if len(df_ml) < 100:
        print(f"❌ Dados insuficientes para treinamento: {len(df_ml)} amostras (mínimo: 100)")
        return None
    
    # As features (variáveis de entrada) - mesmas que o robô usará
    features = ['close', 'MA_Short', 'MA_Long', 'MA_Crossover_Signal', 'RSI']
    
    # Verificar se todas as features estão disponíveis
    missing_features = [f for f in features if f not in df_ml.columns]
    if missing_features:
        print(f"❌ Features faltando: {missing_features}")
        return None
    
    # Prepara os dados para o treinamento
    X = df_ml[features]
    y = df_ml['target']
    
    # Divide os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inicializa e treina o modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Avalia a acurácia do modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Modelo treinado com {len(df_ml)} amostras")
    print(f"📊 Acurácia do modelo: {accuracy:.2f} ({accuracy*100:.1f}%)")
    
    # Salvar o modelo treinado
    try:
        joblib.dump(model, ML_MODEL_FILE)
        print(f"💾 Modelo salvo em: {ML_MODEL_FILE}")
    except Exception as e:
        print(f"⚠️  Não foi possível salvar o modelo: {e}")
    
    return model

# --- Funções de Operação e Notícias (sem alterações) ---
def send_order(symbol, order_type, volume, sl_price=None, tp_price=None, comment="Python Bot"):
    """Envia uma ordem de compra ou venda para o MT5 com SL e TP."""
    try:
        # 1. Verificar informações da conta primeiro (com cache)
        account_info = get_cached_account_info()
        if account_info is None:
            print(f"❌ Erro: Não foi possível obter informações da conta")
            return None
        
        # 2. Verificar se AutoTrading está habilitado
        terminal_info = mt5.terminal_info()
        if terminal_info is None:
            print(f"❌ Erro: Não foi possível obter informações do terminal")
            return None
        
        if not terminal_info.trade_allowed:
            print(f"❌ ERRO: AutoTrading está DESABILITADO no MetaTrader 5!")
            print(f"   💡 SOLUÇÃO: Habilite o AutoTrading no MT5:")
            print(f"      1. Abra o MetaTrader 5")
            print(f"      2. Clique no botão 'AutoTrading' na barra de ferramentas (ou pressione Ctrl+E)")
            print(f"      3. O botão deve ficar verde/ativado")
            print(f"      4. Execute o robô novamente")
            return None
        
        # 3. Obter informações do símbolo (com cache)
        symbol_info = get_cached_symbol_info(symbol)
        if symbol_info is None:
            print(f"❌ Erro: Não foi possível obter informações do símbolo {symbol}")
            return None
        
        # 4. Obter tick atual
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"❌ Erro: Não foi possível obter tick para {symbol}")
            return None
        
        if order_type == mt5.ORDER_TYPE_BUY:
            current_price = tick.ask
        else:
            current_price = tick.bid
        
        # 5. VERIFICAR MARGEM DISPONÍVEL ANTES DE ENVIAR ORDEM
        # Calcular margem necessária para esta operação
        margin_required = mt5.order_calc_margin(order_type, symbol, volume, current_price)
        if margin_required is None:
            print(f"⚠️  Não foi possível calcular margem necessária, tentando continuar...")
        else:
            margin_free = account_info.margin_free
            margin_level = account_info.margin_level if account_info.margin > 0 else 0
            
            print(f"💰 Verificação de Margem:")
            print(f"   Margem Livre: ${margin_free:.2f}")
            print(f"   Margem Necessária: ${margin_required:.2f}")
            print(f"   Margem Level: {margin_level:.2f}%")
            
            # Verificar se há margem suficiente
            if margin_required > margin_free:
                # Tentar reduzir o lote para caber na margem disponível
                volume_max = (margin_free * 0.95) / (margin_required / volume) if margin_required > 0 else volume
                volume_max = np.floor(volume_max / symbol_info.volume_step) * symbol_info.volume_step
                volume_max = max(symbol_info.volume_min, volume_max)
                
                if volume_max < symbol_info.volume_min:
                    print(f"❌ ERRO: Margem insuficiente para abrir posição!")
                    print(f"   Margem disponível: ${margin_free:.2f}")
                    print(f"   Margem necessária (lote {volume:.2f}): ${margin_required:.2f}")
                    print(f"   Lote mínimo permitido: {symbol_info.volume_min:.2f}")
                    return None
                else:
                    print(f"⚠️  Margem insuficiente! Reduzindo lote de {volume:.2f} para {volume_max:.2f}")
                    volume = volume_max
                    # Recalcular margem com novo volume
                    margin_required = mt5.order_calc_margin(order_type, symbol, volume, current_price)
                    if margin_required is None or margin_required > margin_free:
                        print(f"❌ ERRO: Mesmo com lote reduzido, margem ainda insuficiente!")
                        return None
        
        # 6. Verificar distância mínima para SL/TP
        min_distance = symbol_info.trade_tick_size * symbol_info.trade_stops_level
        if min_distance == 0:  # Se não especificado, usar valor padrão
            min_distance = symbol_info.trade_tick_size * 10  # 10 pips mínimo
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": current_price,
            "deviation": 20,
            "magic": 20230727,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Adicionar SL e TP se fornecidos e válidos
        sl_added = False
        tp_added = False
        
        if sl_price is not None:
            sl_distance = abs(current_price - sl_price)
            if sl_distance >= min_distance:
                # Verificar se SL está na direção correta
                if (order_type == mt5.ORDER_TYPE_BUY and sl_price < current_price) or \
                   (order_type == mt5.ORDER_TYPE_SELL and sl_price > current_price):
                    request["sl"] = sl_price
                    sl_added = True
                else:
                    print(f"⚠️  SL na direção incorreta para {order_type}")
                    print(f"   Preço atual: {current_price:.2f}, SL calculado: {sl_price:.2f}")
                    print(f"   Para {order_type}: SL deve ser {'abaixo' if order_type == mt5.ORDER_TYPE_BUY else 'acima'} do preço atual")
            else:
                print(f"⚠️  SL muito próximo do preço atual. Distância: {sl_distance:.2f}, Mínima: {min_distance:.2f}")
        
        if tp_price is not None:
            tp_distance = abs(current_price - tp_price)
            if tp_distance >= min_distance:
                # Verificar se TP está na direção correta
                if (order_type == mt5.ORDER_TYPE_BUY and tp_price > current_price) or \
                   (order_type == mt5.ORDER_TYPE_SELL and tp_price < current_price):
                    request["tp"] = tp_price
                    tp_added = True
                else:
                    print(f"⚠️  TP na direção incorreta para {order_type}")
            else:
                print(f"⚠️  TP muito próximo do preço atual. Distância: {tp_distance:.2f}, Mínima: {min_distance:.2f}")
        
        result = mt5.order_send(request)
        
        # Verificar se a ordem foi enviada com sucesso
        if result is None:
            print(f"❌ Falha ao enviar ordem: resultado None")
            return None
        
        # Verificar códigos de retorno válidos
        valid_codes = [
            mt5.TRADE_RETCODE_DONE,
            mt5.TRADE_RETCODE_PLACED,
            mt5.TRADE_RETCODE_NO_CHANGES
        ]
        
        if result.retcode not in valid_codes:
            # Mensagens de erro mais claras
            error_messages = {
                10004: "Requisição inválida",
                10006: "Sem conexão com o servidor",
                10007: "Erro de timeout",
                10008: "Requisição inválida",
                10009: "Ordem antiga",
                10010: "Sem conexão",
                10011: "Erro interno",
                10012: "Requisição inválida",
                10013: "Parâmetros inválidos",
                10014: "Erro de servidor",
                10015: "Erro de cliente",
                10016: "Requisição bloqueada",
                10017: "Ordem bloqueada",
                10018: "Sem dinheiro",
                10019: "Não há dinheiro suficiente (margem insuficiente)",
                10020: "Preço inválido",
                10021: "Paradas inválidas",
                10022: "Volume inválido",
                10023: "Mercado fechado",
                10024: "Negociação desabilitada",
                10025: "Não há dinheiro",
                10026: "Volume muito pequeno",
                10027: "Volume muito grande ou AutoTrading desabilitado",
                10028: "Preço incorreto",
                10029: "Paradas incorretas",
                10030: "Mercado fechado",
                10031: "Negociação desabilitada",
                10032: "Não há dinheiro",
                10033: "Volume muito pequeno",
                10034: "Volume muito grande",
                10035: "Preço incorreto",
                10036: "Paradas incorretas"
            }
            
            error_msg = error_messages.get(result.retcode, "Erro desconhecido")
            
            # Verificar se o erro é relacionado ao AutoTrading desabilitado
            if result.comment and "autotrading" in result.comment.lower():
                print(f"❌ ERRO: AutoTrading está DESABILITADO no MetaTrader 5!")
                print(f"   Mensagem do servidor: {result.comment}")
                print(f"   💡 SOLUÇÃO: Habilite o AutoTrading no MT5:")
                print(f"      1. Abra o MetaTrader 5")
                print(f"      2. Clique no botão 'AutoTrading' na barra de ferramentas (ou pressione Ctrl+E)")
                print(f"      3. O botão deve ficar verde/ativado")
                print(f"      4. Execute o robô novamente")
            else:
                print(f"❌ Erro ao enviar ordem: Código {result.retcode} - {error_msg}")
                print(f"   Mensagem do servidor: {result.comment}")
            
            if result.retcode == 10019:  # No money
                print(f"   💡 SOLUÇÃO: Reduza o tamanho do lote ou aumente o saldo da conta")
                print(f"   Margem livre atual: ${account_info.margin_free:.2f}")
                if margin_required:
                    print(f"   Margem necessária: ${margin_required:.2f}")
                    # Calcular lote máximo possível
                    if margin_required > 0:
                        volume_max_calc = (account_info.margin_free * 0.95) / (margin_required / volume) if volume > 0 else 0
                        volume_max_calc = np.floor(volume_max_calc / symbol_info.volume_step) * symbol_info.volume_step
                        volume_max_calc = max(symbol_info.volume_min, volume_max_calc)
                        print(f"   Lote máximo possível com margem disponível: {volume_max_calc:.2f}")
            
            print(f"   Detalhes completos: {result}")
            return None
        
        # Ordem enviada com sucesso
        sl_info = f", SL: {sl_price:.2f}" if sl_added else ""
        tp_info = f", TP: {tp_price:.2f}" if tp_added else ""
        print(f"✅ Ordem {order_type} enviada com sucesso! Ticket: {result.order}, Preço: {current_price:.4f}{sl_info}{tp_info}")
        
        # Invalidar cache após ordem executada
        invalidate_cache()
        
        # Se SL/TP não foram definidos na ordem, tentar definir depois
        if (sl_price is not None and not sl_added) or (tp_price is not None and not tp_added):
            time.sleep(1)  # Aguardar ordem ser processada
            set_sl_tp_after_order(result.order, sl_price, tp_price)
        
        return result
    except Exception as e:
        print(f"❌ Erro inesperado ao enviar ordem: {e}")
        return None

def set_sl_tp_after_order(order_ticket, sl_price, tp_price):
    """Define SL e TP após a ordem ser executada."""
    try:
        if sl_price is None and tp_price is None:
            return
        
        # Aguardar a posição aparecer
        time.sleep(2)
        
        # Buscar a posição pelo ticket
        positions = mt5.positions_get(ticket=order_ticket)
        if not positions:
            print(f"⚠️  Posição {order_ticket} não encontrada para definir SL/TP")
            return
        
        position = positions[0]
        
        # Verificar distância mínima do símbolo (com cache)
        symbol_info = get_cached_symbol_info(position.symbol)
        if symbol_info:
            min_distance = symbol_info.trade_tick_size * symbol_info.trade_stops_level
            if min_distance == 0:
                min_distance = symbol_info.trade_tick_size * 10
            
            current_price = position.price_current
            
            # Validar SL antes de definir
            if sl_price is not None:
                sl_distance = abs(current_price - sl_price)
                # Verificar direção correta
                is_buy = position.type == mt5.ORDER_TYPE_BUY
                if sl_distance < min_distance:
                    print(f"⚠️  SL muito próximo: {sl_distance:.2f} < {min_distance:.2f}")
                    return
                if (is_buy and sl_price >= current_price) or (not is_buy and sl_price <= current_price):
                    print(f"⚠️  SL na direção incorreta. Preço atual: {current_price:.2f}, SL: {sl_price:.2f}")
                    # Ajustar SL para uma distância mínima válida
                    if is_buy:
                        sl_price = current_price - min_distance * 2
                    else:
                        sl_price = current_price + min_distance * 2
                    sl_price = round(sl_price, 2)
                    print(f"   SL ajustado para: {sl_price:.2f}")
            
            # Validar TP antes de definir
            if tp_price is not None:
                tp_distance = abs(current_price - tp_price)
                if tp_distance < min_distance:
                    print(f"⚠️  TP muito próximo: {tp_distance:.2f} < {min_distance:.2f}")
                    return
                if (is_buy and tp_price <= current_price) or (not is_buy and tp_price >= current_price):
                    print(f"⚠️  TP na direção incorreta. Preço atual: {current_price:.2f}, TP: {tp_price:.2f}")
                    # Ajustar TP para uma distância mínima válida
                    if is_buy:
                        tp_price = current_price + min_distance * 2
                    else:
                        tp_price = current_price - min_distance * 2
                    tp_price = round(tp_price, 2)
                    print(f"   TP ajustado para: {tp_price:.2f}")
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": position.ticket,
            "sl": sl_price if sl_price else position.sl,
            "tp": tp_price if tp_price else position.tp,
        }
        
        result = mt5.order_send(request)
        if result:
            if result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_NO_CHANGES]:
                print(f"✅ SL/TP definidos para posição {order_ticket}: SL={request['sl']:.2f}, TP={request['tp']:.2f}")
            else:
                print(f"⚠️  Falha ao definir SL/TP para posição {order_ticket}")
                print(f"   Código de retorno: {result.retcode}, Mensagem: {result.comment}")
        else:
            print(f"⚠️  Falha ao definir SL/TP para posição {order_ticket}: {mt5.last_error()}")
            
    except Exception as e:
        print(f"❌ Erro ao definir SL/TP: {e}")

def close_position(position_ticket):
    """Fecha uma posição aberta pelo ticket."""
    try:
        # Buscar posição (forçar refresh para garantir dados atualizados)
        positions = get_cached_positions(SYMBOL, force_refresh=True)
        position = [p for p in positions if p.ticket == position_ticket] if positions else None
        if not position:
            return None
        position = position[0]
        
        # Invalidar cache após fechar posição
        invalidate_cache()
        if position.type == mt5.ORDER_TYPE_BUY:
            close_order_type = mt5.ORDER_TYPE_SELL
            close_price = mt5.symbol_info_tick(position.symbol).bid
        else:
            close_order_type = mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(position.symbol).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_order_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 20230727,
            "comment": "Fechamento por Bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Erro ao fechar posição {position_ticket}: {mt5.last_error()}")
            return None
        print(f"Posição {position_ticket} ({position.type}) fechada com sucesso! Preço: {close_price:.4f}")
        return result
    except Exception as e:
        print(f"Erro inesperado ao fechar posição: {e}")
        return None

def run_backtest(df, ma_short_period, ma_long_period, rsi_period, rsi_ob, rsi_os):
    """
    Simula trades com um conjunto de parâmetros e retorna os resultados.
    """
    df_test = calculate_indicators(df.copy(), ma_short_period, ma_long_period, rsi_period)
    df_test = df_test.dropna()
    if df_test.empty:
        return 0, 0

    total_profit = 0
    in_position = False
    entry_price = 0

    for i in range(1, len(df_test)):
        last_ma_short = df_test['MA_Short'].iloc[i]
        prev_ma_short = df_test['MA_Short'].iloc[i-1]
        last_ma_long = df_test['MA_Long'].iloc[i]
        prev_ma_long = df_test['MA_Long'].iloc[i-1]
        last_rsi = df_test['RSI'].iloc[i]
        current_price = df_test['close'].iloc[i]

        buy_signal = last_ma_short > last_ma_long and prev_ma_short <= prev_ma_long and rsi_os < last_rsi < rsi_ob
        if buy_signal and not in_position:
            in_position = True
            entry_price = current_price
        
        sell_signal = last_ma_short < last_ma_long and prev_ma_short >= prev_ma_long
        if sell_signal and in_position:
            total_profit += (current_price - entry_price)
            in_position = False
            
    if in_position:
        total_profit += (df_test['close'].iloc[-1] - entry_price)

    return total_profit, 0

def optimize_parameters():
    """
    Executa a otimização dos parâmetros usando dados históricos do MT5.
    Garante que os dados sejam exatamente os mesmos que o robô usará.
    """
    print("\n--- Iniciando Otimização de Parâmetros ---")
    
    # Verificar se MT5 está disponível
    if mt5.terminal_info() is None:
        print("❌ MT5 não está conectado, usando dados da web...")
        df_historical = get_web_data(SYMBOL, days=30)
        if not verify_data_consistency(df_historical, "Web"):
            print("❌ Erro: Nenhum dado disponível para otimização.")
            return None
        print("⚠️  Usando dados da web para otimização (menos preciso que MT5)")
    else:
        print("🎯 Usando dados históricos do MetaTrader 5 para otimização...")
        
        # Usar dados do MT5 com o mesmo timeframe e período que o robô usará
        df_historical = get_ohlc_data(SYMBOL, TIMEFRAME, 5000)  # 5000 barras para otimização robusta
        
        if not verify_data_consistency(df_historical, "MT5"):
            print("❌ Método original falhou, tentando método simples...")
            df_historical = get_ohlc_data_simple(SYMBOL, TIMEFRAME, 5000)
            
            if not verify_data_consistency(df_historical, "MT5 (simples)"):
                print("❌ Erro: Otimização cancelada. Dados do MT5 não disponíveis.")
                print("💡 Tentando dados da web como fallback...")
                df_historical = get_web_data(SYMBOL, days=30)
                if not verify_data_consistency(df_historical, "Web"):
                    print("❌ Erro: Nenhum dado disponível para otimização.")
                    return None
                else:
                    print("⚠️  Usando dados da web para otimização (menos preciso que MT5)")
            else:
                print(f"✅ Dados do MT5 obtidos (método simples): {len(df_historical)} barras de {TIMEFRAME}")
        else:
            print(f"✅ Dados do MT5 obtidos: {len(df_historical)} barras de {TIMEFRAME}")

    best_profit = -float('inf')
    best_params = {}

    ma_short_periods = range(3, 15, 1)
    ma_long_periods = range(15, 50, 5)
    rsi_overboughts = range(60, 85, 5)
    rsi_oversolds = range(15, 40, 5)

    total_combinations = len(ma_short_periods) * len(ma_long_periods) * len(rsi_overboughts) * len(rsi_oversolds)
    print(f"Testando {total_combinations} combinações...")

    for short_period in ma_short_periods:
        for long_period in ma_long_periods:
            if short_period >= long_period: continue
            for rsi_ob in rsi_overboughts:
                for rsi_os in rsi_oversolds:
                    if rsi_os >= rsi_ob: continue
                    
                    profit, _ = run_backtest(
                        df_historical, 
                        ma_short_period=short_period, 
                        ma_long_period=long_period,
                        rsi_period=14,
                        rsi_ob=rsi_ob,
                        rsi_os=rsi_os
                    )
                    
                    if profit > best_profit:
                        best_profit = profit
                        best_params = {
                            "MA_SHORT_PERIOD": short_period,
                            "MA_LONG_PERIOD": long_period,
                            "RSI_OVERBOUGHT": rsi_ob,
                            "RSI_OVERSOLD": rsi_os
                        }
                        
    print("\n--- Otimização Concluída ---")
    print(f"Melhor Lucro Encontrado: {best_profit:.2f}")
    print(f"Melhores Parâmetros: {best_params}")
    
    return best_params

def close_all_positions():
    """Fecha todas as posições abertas pelo robô."""
    try:
        print("🔄 Fechando todas as posições abertas...")
        positions = mt5.positions_get(symbol=SYMBOL)
        
        if not positions:
            print("✅ Nenhuma posição aberta para fechar")
            return
        
        closed_count = 0
        for position in positions:
            if position.magic == 20230727:  # Apenas posições do robô
                print(f"🔄 Fechando posição {position.ticket} ({position.type})...")
                result = close_position(position.ticket)
                if result:
                    closed_count += 1
                    print(f"✅ Posição {position.ticket} fechada com sucesso")
                else:
                    print(f"❌ Erro ao fechar posição {position.ticket}")
                time.sleep(1)  # Pequena pausa entre fechamentos
        
        print(f"✅ {closed_count} posições fechadas com sucesso")
        
    except Exception as e:
        print(f"❌ Erro ao fechar posições: {e}")

def manage_positions_timeboxed(symbol):
    """
    Gerencia posições abertas para realizar lucro dentro de 30 minutos e encerrar ao atingir o limite de tempo.
    Inclui trailing stop básico para proteger lucros.
    """
    try:
        # Usar cache de posições
        positions = get_cached_positions(symbol)
        if not positions:
            return

        now_dt = datetime.now()
        for position in positions:
            if position.magic != 20230727:
                continue

            # Calcular tempo de posição aberta
            try:
                open_time = datetime.fromtimestamp(position.time)
            except Exception:
                # Fallback: se time_update existir, usa-se ele
                try:
                    open_time = datetime.fromtimestamp(position.time_update)
                except Exception:
                    continue

            minutes_open = (now_dt - open_time).total_seconds() / 60.0

            # Lucro atual em moeda da conta
            current_profit = float(position.profit)
            entry_price = position.price_open
            current_price = position.price_current
            current_sl = position.sl
            current_tp = position.tp
            trade_type = 'BUY' if position.type == mt5.ORDER_TYPE_BUY else 'SELL'
            
            # Calcular distância até TP para trailing stop
            if current_tp > 0:
                if trade_type == 'BUY':
                    distance_to_tp = current_tp - entry_price
                    profit_pct_of_tp = (current_price - entry_price) / distance_to_tp if distance_to_tp > 0 else 0
                else:  # SELL
                    distance_to_tp = entry_price - current_tp
                    profit_pct_of_tp = (entry_price - current_price) / distance_to_tp if distance_to_tp > 0 else 0
            else:
                profit_pct_of_tp = 0

            # TRAILING STOP: Proteger lucros quando posição está em lucro significativo
            if ENABLE_TRAILING_STOP and current_profit > 0 and profit_pct_of_tp >= TRAILING_STOP_ACTIVATION:
                # Calcular novo SL para proteger lucro
                if trade_type == 'BUY':
                    # Para BUY: mover SL para cima (break-even ou acima)
                    profit_distance = current_price - entry_price
                    new_sl = entry_price + (profit_distance * (1 - TRAILING_STOP_DISTANCE))
                    # Garantir que novo SL seja melhor que o atual
                    if new_sl > current_sl and new_sl < current_price:
                        # Atualizar SL apenas se for melhor
                        try:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": position.symbol,
                                "position": position.ticket,
                                "sl": round(new_sl, 2),
                                "tp": current_tp if current_tp > 0 else None,
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_NO_CHANGES]:
                                print(f"🔄 Trailing Stop atualizado: Ticket {position.ticket} | SL: {current_sl:.2f} → {new_sl:.2f} | Lucro: ${current_profit:.2f}")
                        except Exception as e:
                            print(f"⚠️  Erro ao atualizar trailing stop: {e}")
                else:  # SELL
                    # Para SELL: mover SL para baixo (break-even ou abaixo)
                    profit_distance = entry_price - current_price
                    new_sl = entry_price - (profit_distance * (1 - TRAILING_STOP_DISTANCE))
                    # Garantir que novo SL seja melhor que o atual
                    if (current_sl == 0 or new_sl < current_sl) and new_sl > current_price:
                        # Atualizar SL apenas se for melhor
                        try:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "symbol": position.symbol,
                                "position": position.ticket,
                                "sl": round(new_sl, 2),
                                "tp": current_tp if current_tp > 0 else None,
                            }
                            result = mt5.order_send(request)
                            if result and result.retcode in [mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_NO_CHANGES]:
                                print(f"🔄 Trailing Stop atualizado: Ticket {position.ticket} | SL: {current_sl:.2f} → {new_sl:.2f} | Lucro: ${current_profit:.2f}")
                        except Exception as e:
                            print(f"⚠️  Erro ao atualizar trailing stop: {e}")

            # 1) Se está no lucro e ainda dentro da janela, realizar lucro
            if current_profit > MIN_PROFIT_USD and minutes_open <= MAX_HOLD_MINUTES:
                print(f"🟢 Realizando lucro cedo: ticket {position.ticket} | lucro {current_profit:.2f} | {minutes_open:.1f} min")
                close_position(position.ticket)
                # Atualizar resultados
                update_trading_results(trade_type, current_profit, entry_price, current_price, ['TIME_MANAGEMENT'])
                time.sleep(1)
                continue

            # 2) Se atingiu o limite de tempo, encerra a posição (independente do resultado)
            if minutes_open >= MAX_HOLD_MINUTES:
                print(f"⏰ Limite de {MAX_HOLD_MINUTES} min atingido: fechando ticket {position.ticket} | lucro {current_profit:.2f}")
                close_position(position.ticket)
                # Atualizar resultados
                update_trading_results(trade_type, current_profit, entry_price, current_price, ['TIME_LIMIT'])
                time.sleep(1)

    except Exception as e:
        print(f"Erro ao gerenciar posições por tempo: {e}")

def check_high_impact_news(currencies):
    """
    Verifica se há notícias de alto impacto programadas para as moedas em um curto período.
    Retorna True se um evento de alto impacto estiver se aproximando, False caso contrário.
    Usa cache para evitar requisições HTTP repetidas.
    """
    global CACHE_NEWS_CHECK, CACHE_NEWS_TIMESTAMP
    
    now = time.time()
    
    # Verificar cache de notícias
    if (CACHE_NEWS_CHECK is not None and 
        CACHE_NEWS_TIMESTAMP is not None and 
        (now - CACHE_NEWS_TIMESTAMP) < CACHE_NEWS_TTL):
        return CACHE_NEWS_CHECK
    
    # Cache expirado ou não existe, fazer nova verificação
    try:
        response = requests.get(INVESTING_CALENDAR_URL, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        if response.status_code != 200:
            print(f"Erro ao acessar o calendário econômico: {response.status_code}")
            return False

        soup = BeautifulSoup(response.content, 'html.parser')
        news_rows = soup.find_all('tr', class_=['js-event-item'])
        
        for row in news_rows:
            impact_div = row.find('td', class_='sentiment')
            if impact_div and impact_div.find('i', class_='impact3'):
                
                currency_td = row.find('td', class_='currency')
                time_td = row.find('td', class_='time')

                if currency_td and time_td:
                    currency = currency_td.get_text().strip()
                    event_time_str = time_td.get_text().strip()
                    
                    if currency in currencies:
                        try:
                            news_time = datetime.strptime(event_time_str, '%H:%M')
                            now = datetime.now()
                            news_time_full = now.replace(hour=news_time.hour, minute=news_time.minute, second=0, microsecond=0)
                            
                            time_to_event = (news_time_full - now).total_seconds() / 60
                            
                            if 0 < time_to_event <= SAFETY_PERIOD_MINUTES:
                                print(f"⚠️  AVISO: Evento de alto impacto para {currency} se aproxima em {int(time_to_event)} minutos.")
                                # Atualizar cache
                                CACHE_NEWS_CHECK = True
                                CACHE_NEWS_TIMESTAMP = now
                                return True

                        except ValueError:
                            continue
        
        # Nenhuma notícia encontrada, atualizar cache
        CACHE_NEWS_CHECK = False
        CACHE_NEWS_TIMESTAMP = now
                            
    except Exception as e:
        print(f"Erro na verificação de notícias: {e}")
        # Em caso de erro, não cachear (retornar False mas não atualizar cache)
        return False

    return False

# --- Função Principal do Robô ---
def run_bot():
    """Função principal que executa o loop de negociação do robô."""
    global CACHED_DF, CACHED_DF_LAST_UPDATE, CACHED_DF_TIMEFRAME, CACHED_DF_SYMBOL
    
    if not connect_mt5():
        return

    # Contador de operações para estatísticas
    total_operations = 0
    profitable_operations = 0
    # Controle para executar entradas apenas uma vez por barra fechada do timeframe escolhido
    last_processed_bar_time = None
    
    print("🚀 Robô iniciado! Monitorando mercado continuamente...")
    print("💡 Pressione Ctrl+C para parar o robô e fechar todas as posições")

    while True:
        try:
            print(f"\n--- Verificando em {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")
            print(f"📊 Estatísticas: {total_operations} operações, {profitable_operations} lucrativas")
            
            # Mostrar resumo rápido a cada 10 operações
            if TRADING_RESULTS['total_trades'] > 0 and TRADING_RESULTS['total_trades'] % 10 == 0:
                print(f"\n📈 Resumo rápido: {TRADING_RESULTS['total_trades']} trades | Win Rate: {TRADING_RESULTS['win_rate']:.1f}% | Lucro: ${TRADING_RESULTS['total_profit']:.2f}")
            
            # 🎯 PRIORIDADE: Usar dados do MT5 (mesmo tipo que otimização e ML)
            # Verificar conexão MT5 apenas uma vez por ciclo (otimizado)
            terminal_info = mt5.terminal_info()
            if terminal_info is None:
                print("❌ MT5 não está conectado, usando dados da web...")
                df = get_web_data(SYMBOL, days=14)
                if not verify_data_consistency(df, "Web"):
                    print("❌ Nenhum dado disponível. Aguardando próximo ciclo...")
                    time.sleep(30)
                    continue
                print("⚠️  Usando dados da web (menos preciso que MT5)")
            else:
                print("📊 Obtendo dados do MetaTrader 5...")
                # Tentar método simples primeiro (mais rápido e com cache)
                df = get_ohlc_data_simple(SYMBOL, TIMEFRAME, MA_LONG_PERIOD + RSI_PERIOD + 100, use_cache=True)
                
                if not verify_data_consistency(df, "MT5"):
                    print("❌ Método simples falhou, tentando método original...")
                    df = get_ohlc_data(SYMBOL, TIMEFRAME, MA_LONG_PERIOD + RSI_PERIOD + 100)
                    
                    if not verify_data_consistency(df, "MT5 (simples)"):
                        print("❌ Dados do MT5 não disponíveis, tentando dados da web...")
                        df = get_web_data(SYMBOL, days=14)
                        if not verify_data_consistency(df, "Web"):
                            print("❌ Nenhum dado disponível. Aguardando próximo ciclo...")
                            time.sleep(30)
                            continue
                        else:
                            print("⚠️  Usando dados da web (menos preciso que MT5)")
                    else:
                        print(f"✅ Dados do MT5 obtidos (método simples): {len(df)} barras")
                else:
                    print(f"✅ Dados do MT5 obtidos: {len(df)} barras")
            
            # Cria as features e o target para o ML
            # Verificar se já temos indicadores calculados no cache
            if (CACHED_DF is not None and 
                'MA_Short' in CACHED_DF.columns and 
                'MA_Long' in CACHED_DF.columns and 
                'RSI' in CACHED_DF.columns and
                len(df) == len(CACHED_DF) and
                df.index[-1] == CACHED_DF.index[-1]):
                # Usar indicadores do cache se não há novas barras
                df = CACHED_DF.copy()
                print("✅ Usando indicadores do cache (sem novas barras)")
            else:
                # Recalcular indicadores apenas quando necessário
                df = create_features(df)
                df = df.dropna()
                # Atualizar cache de indicadores
                if CACHED_DF is None or len(df) != len(CACHED_DF) or df.index[-1] != CACHED_DF.index[-1]:
                    CACHED_DF = df.copy()
                    print("✅ Indicadores recalculados e cache atualizado")
            
            if len(df) < MA_LONG_PERIOD + RSI_PERIOD + 1:
                print(f"Dados insuficientes para análise: {len(df)} barras disponíveis.")
                print("Aguardando mais dados...")
                time.sleep(30)
                continue
            
            # Faz a previsão com o modelo treinado (se disponível)
            ml_signal = "N/A"
            if ML_MODEL is not None:
                try:
                    current_data = df.tail(1)[['close', 'MA_Short', 'MA_Long', 'MA_Crossover_Signal', 'RSI']]
                    prediction = ML_MODEL.predict(current_data)
                    
                    if prediction[0] == 1:
                        ml_signal = "COMPRA (UP)"
                    else:
                        ml_signal = "VENDA (DOWN)"
                        
                    print(f"Previsão do Modelo de ML: {ml_signal}")
                except Exception as e:
                    print(f"Erro ao fazer predição ML: {e}")
                    ml_signal = "ERRO"
            else:
                print("Modelo de ML não disponível - operando sem filtro ML")

            last_close = df['close'].iloc[-1]
            last_ma_short = df['MA_Short'].iloc[-1]
            last_ma_long = df['MA_Long'].iloc[-1]
            last_rsi = df['RSI'].iloc[-1]
            prev_ma_short = df['MA_Short'].iloc[-2]
            prev_ma_long = df['MA_Long'].iloc[-2]

            # Obter a última barra fechada (penúltima no DataFrame, pois a última ainda está sendo formada)
            # Devemos pegar a última barra que foi fechada (não a que está sendo formada)
            try:
                # A penúltima barra é a última que foi completamente fechada
                closed_bar_time = df.index[-2]
            except (IndexError, KeyError):
                # Se não há penúltima, usar a última
                closed_bar_time = df.index[-1]
            
            # Arredondar para o slot do timeframe para comparação correta
            if isinstance(closed_bar_time, pd.Timestamp):
                # Arredondar baseado no timeframe
                closed_bar_time_rounded = closed_bar_time.replace(second=0, microsecond=0)
                # Ajustar para o slot do timeframe (ex: para M5, arredondar para múltiplos de 5)
                if TIMEFRAME_MINUTES < 60 and TIMEFRAME_MINUTES > 1:
                    minute_slot = (closed_bar_time_rounded.minute // TIMEFRAME_MINUTES) * TIMEFRAME_MINUTES
                    closed_bar_time_rounded = closed_bar_time_rounded.replace(minute=minute_slot)
                elif TIMEFRAME_MINUTES >= 60:
                    closed_bar_time_rounded = closed_bar_time_rounded.replace(minute=0)
                
                # Converter para timezone-naive se necessário para comparação
                if closed_bar_time_rounded.tzinfo is not None:
                    closed_bar_time_rounded = closed_bar_time_rounded.replace(tzinfo=None)
            else:
                closed_bar_time_rounded = closed_bar_time
            
            # Obter o tempo atual arredondado para comparação
            current_time = datetime.now().replace(second=0, microsecond=0)

            # Verificar posições atuais (com cache)
            positions = get_cached_positions(SYMBOL)
            current_position_ticket = None
            current_position_type = None
            if positions:
                for pos in positions:
                    if pos.magic == 20230727:
                        current_position_ticket = pos.ticket
                        current_position_type = 'buy' if pos.type == mt5.ORDER_TYPE_BUY else 'sell'
                        break
            
            print(f"Preço: {last_close:.4f} | MA Curta ({MA_SHORT_PERIOD}): {last_ma_short:.4f} | MA Longa ({MA_LONG_PERIOD}): {last_ma_long:.4f} | RSI ({RSI_PERIOD}): {last_rsi:.2f}")
            print(f"Posição atual do robô: {current_position_type if current_position_type else 'Nenhuma'}")
            print(f"Última barra {TIMEFRAME_NAME} fechada: {closed_bar_time_rounded} | Última processada: {last_processed_bar_time} | Tempo atual: {current_time}")

            # --- LÓGICA DE TRADING APRIMORADA ---
            
            is_news_coming = check_high_impact_news(NEWS_CURRENCIES)
            print(f"Verificação de Notícias: Há notícias de alto impacto? -> {is_news_coming}")

            # Gestão de posições por tempo/lucro (30 minutos)
            manage_positions_timeboxed(SYMBOL)

            if is_news_coming:
                print("🚫 Não há entrada de operações devido a notícia de alto impacto se aproximando.")
                # Fechar posições existentes para evitar risco
                if current_position_ticket:
                    print("🔄 Fechando posição existente para evitar risco de notícia.")
                    close_position(current_position_ticket)
                    time.sleep(2)
                # Atualizar última barra processada mesmo sem entrada
                if isinstance(closed_bar_time_rounded, pd.Timestamp):
                    closed_time_final = closed_bar_time_rounded.replace(second=0, microsecond=0)
                    if closed_time_final.tzinfo is not None:
                        closed_time_final = closed_time_final.replace(tzinfo=None)
                else:
                    closed_time_final = closed_bar_time_rounded
                last_processed_bar_time = closed_time_final
                time.sleep(5)
                continue

            # Verificar se é uma nova barra fechada baseado no timeframe escolhido
            is_new_bar = False
            
            if last_processed_bar_time is None:
                # Primeira execução, processar
                is_new_bar = True
                print(f"🔄 Primeira execução - processando barra atual")
            else:
                # Comparar timestamps arredondados baseado no timeframe
                last_processed_rounded = last_processed_bar_time
                if isinstance(last_processed_bar_time, pd.Timestamp):
                    # Arredondar baseado no timeframe
                    if TIMEFRAME_MINUTES < 60:
                        # Para timeframes em minutos, arredondar para o minuto
                        last_processed_rounded = last_processed_bar_time.replace(second=0, microsecond=0)
                        # Ajustar para o slot do timeframe (ex: para M5, arredondar para múltiplos de 5)
                        if TIMEFRAME_MINUTES > 1:
                            minute_slot = (last_processed_rounded.minute // TIMEFRAME_MINUTES) * TIMEFRAME_MINUTES
                            last_processed_rounded = last_processed_rounded.replace(minute=minute_slot)
                    else:
                        # Para timeframes de horas, arredondar para a hora
                        last_processed_rounded = last_processed_bar_time.replace(minute=0, second=0, microsecond=0)
                    
                    if last_processed_rounded.tzinfo is not None:
                        last_processed_rounded = last_processed_rounded.replace(tzinfo=None)
                
                # Nova barra se o timestamp da barra fechada é maior que o último processado
                if isinstance(closed_bar_time_rounded, pd.Timestamp):
                    closed_time_for_compare = closed_bar_time_rounded.replace(second=0, microsecond=0)
                    # Ajustar para o slot do timeframe
                    if TIMEFRAME_MINUTES < 60 and TIMEFRAME_MINUTES > 1:
                        minute_slot = (closed_time_for_compare.minute // TIMEFRAME_MINUTES) * TIMEFRAME_MINUTES
                        closed_time_for_compare = closed_time_for_compare.replace(minute=minute_slot)
                    elif TIMEFRAME_MINUTES >= 60:
                        closed_time_for_compare = closed_time_for_compare.replace(minute=0)
                    
                    if closed_time_for_compare.tzinfo is not None:
                        closed_time_for_compare = closed_time_for_compare.replace(tzinfo=None)
                else:
                    closed_time_for_compare = closed_bar_time_rounded
                
                # Nova barra se a barra fechada é mais recente que a última processada
                if closed_time_for_compare > last_processed_rounded:
                    is_new_bar = True
                    print(f"✅ Nova barra {TIMEFRAME_NAME} detectada! Barra anterior: {last_processed_rounded} | Nova: {closed_time_for_compare}")
                elif closed_time_for_compare == last_processed_rounded:
                    # Calcular tempo até próxima barra
                    wait_seconds = calcular_tempo_proxima_barra(TIMEFRAME_MINUTES)
                    print(f"⏳ Aguardando nova barra {TIMEFRAME_NAME} fechar...")
                    print(f"   Barra atual: {closed_time_for_compare} | Última processada: {last_processed_rounded}")
                    print(f"   Próxima barra em aproximadamente {wait_seconds:.0f} segundos")
                    # Aguardar até a próxima barra (com verificação periódica)
                    time.sleep(min(wait_seconds, 30))  # Máximo 30 segundos por verificação
                    continue
                else:
                    # Barra antiga (não deveria acontecer, mas por segurança processar)
                    print(f"⚠️  Barra fechada ({closed_time_for_compare}) é anterior à última processada ({last_processed_rounded}). Processando mesmo assim...")
                    is_new_bar = True
            
            # Se não é uma nova barra, aguardar até próxima barra
            if not is_new_bar:
                wait_seconds = calcular_tempo_proxima_barra(TIMEFRAME_MINUTES)
                print(f"⏳ Aguardando nova barra {TIMEFRAME_NAME} fechar (próxima em ~{wait_seconds:.0f}s)...")
                time.sleep(min(wait_seconds, 30))  # Máximo 30 segundos por verificação
                continue
            
            # Avaliar sinais usando os indicadores selecionados pelo usuário
            signals = evaluate_signals(df, last_ma_short, last_ma_long, prev_ma_short, prev_ma_long, last_rsi, ml_signal)
            
            print(f"Indicadores selecionados: {list(SELECTED_INDICATORS.keys())}")
            print(f"Sinais detectados: {signals['indicators_triggered']}")
            print(f"✅ Sinal BUY: {signals['buy']} | Sinal SELL: {signals['sell']}")
            print(f"Posição atual: {current_position_type if current_position_type else 'Nenhuma'} | Ticket: {current_position_ticket}")
            
            # Condição de COMPRA usando indicadores selecionados
            if signals['buy']:
                if current_position_type != 'buy':
                    if current_position_ticket:
                        print("🔄 Sinal de COMPRA detectado! Fechando posição de VENDA existente...")
                        close_position(current_position_ticket)
                        time.sleep(1)
                    print("🟢 Sinal de COMPRA detectado! Abrindo nova posição LONG...")
                    print(f"Indicadores que geraram o sinal: {', '.join(signals['indicators_triggered'])}")
                    
                    # Obter preço atual do tick antes de calcular SL/TP
                    tick = mt5.symbol_info_tick(SYMBOL)
                    if tick is None:
                        print("❌ Erro ao obter tick atual. Pulando esta entrada...")
                        time.sleep(5)
                        continue
                    current_entry_price = tick.ask  # Para BUY, usa ask
                    
                    # Calcular gestão de risco para COMPRA com preço atual
                    print(f"📊 Calculando gestão de risco para COMPRA @ {current_entry_price:.2f}...")
                    risk_params = gerenciar_risco(df, current_entry_price, "BUY")
                    if risk_params:
                        print(f"📊 Parâmetros calculados: Lote={risk_params['lote']:.2f}, SL={risk_params['sl']:.2f}, TP={risk_params['tp']:.2f}")
                        result = send_order(SYMBOL, mt5.ORDER_TYPE_BUY, risk_params['lote'], 
                                          risk_params['sl'], risk_params['tp'])
                        if result:
                            total_operations += 1
                            print(f"✅ Ordem de COMPRA executada! Ticket: {result.order}")
                        else:
                            print(f"❌ Erro: Ordem de COMPRA não foi executada. Verifique os logs acima.")
                    else:
                        print(f"❌ Erro: Não foi possível calcular parâmetros de risco para COMPRA")
                    # Marca esta barra como processada (independente de execução bem-sucedida)
                    if isinstance(closed_bar_time_rounded, pd.Timestamp):
                        closed_time_final = closed_bar_time_rounded.replace(second=0, microsecond=0)
                        if closed_time_final.tzinfo is not None:
                            closed_time_final = closed_time_final.replace(tzinfo=None)
                    else:
                        closed_time_final = closed_bar_time_rounded
                    last_processed_bar_time = closed_time_final
                    time.sleep(3)
                    continue

            # Condição de VENDA usando indicadores selecionados
            if signals['sell']:
                if current_position_type != 'sell':
                    if current_position_ticket:
                        print("🔄 Sinal de VENDA detectado! Fechando posição de COMPRA existente...")
                        close_position(current_position_ticket)
                        time.sleep(1)
                    print("🔴 Sinal de VENDA detectado! Abrindo nova posição SHORT...")
                    print(f"Indicadores que geraram o sinal: {', '.join(signals['indicators_triggered'])}")
                    
                    # Obter preço atual do tick antes de calcular SL/TP
                    tick = mt5.symbol_info_tick(SYMBOL)
                    if tick is None:
                        print("❌ Erro ao obter tick atual. Pulando esta entrada...")
                        time.sleep(5)
                        continue
                    current_entry_price = tick.bid  # Para SELL, usa bid
                    
                    # Calcular gestão de risco para VENDA com preço atual
                    print(f"📊 Calculando gestão de risco para VENDA @ {current_entry_price:.2f}...")
                    risk_params = gerenciar_risco(df, current_entry_price, "SELL")
                    if risk_params:
                        print(f"📊 Parâmetros calculados: Lote={risk_params['lote']:.2f}, SL={risk_params['sl']:.2f}, TP={risk_params['tp']:.2f}")
                        result = send_order(SYMBOL, mt5.ORDER_TYPE_SELL, risk_params['lote'], 
                                          risk_params['sl'], risk_params['tp'])
                        if result:
                            total_operations += 1
                            print(f"✅ Ordem de VENDA executada! Ticket: {result.order}")
                        else:
                            print(f"❌ Erro: Ordem de VENDA não foi executada. Verifique os logs acima.")
                    else:
                        print(f"❌ Erro: Não foi possível calcular parâmetros de risco para VENDA")
                    # Marca esta barra como processada (independente de execução bem-sucedida)
                    if isinstance(closed_bar_time_rounded, pd.Timestamp):
                        closed_time_final = closed_bar_time_rounded.replace(second=0, microsecond=0)
                        if closed_time_final.tzinfo is not None:
                            closed_time_final = closed_time_final.replace(tzinfo=None)
                    else:
                        closed_time_final = closed_bar_time_rounded
                    last_processed_bar_time = closed_time_final
                    time.sleep(3)
                    continue

            # IMPORTANTE: Atualizar last_processed_bar_time APENAS quando processamos uma nova barra
            # Isso garante que não processaremos a mesma barra novamente
            if isinstance(closed_bar_time_rounded, pd.Timestamp):
                closed_time_final = closed_bar_time_rounded.replace(second=0, microsecond=0)
                # Ajustar para o slot do timeframe
                if TIMEFRAME_MINUTES < 60 and TIMEFRAME_MINUTES > 1:
                    minute_slot = (closed_time_final.minute // TIMEFRAME_MINUTES) * TIMEFRAME_MINUTES
                    closed_time_final = closed_time_final.replace(minute=minute_slot)
                elif TIMEFRAME_MINUTES >= 60:
                    closed_time_final = closed_time_final.replace(minute=0)
                
                if closed_time_final.tzinfo is not None:
                    closed_time_final = closed_time_final.replace(tzinfo=None)
            else:
                closed_time_final = closed_bar_time_rounded
            
            last_processed_bar_time = closed_time_final
            print(f"✅ Barra {TIMEFRAME_NAME} {closed_time_final} avaliada (sinais: {signals['indicators_triggered']}). Última processada atualizada.")

            # Fechamento de Posições Existentes (Lógica mais agressiva)
            if current_position_type == 'buy':
                close_signal = (
                    (last_ma_short < last_ma_long and prev_ma_short >= prev_ma_long) or  # Cruzamento reverso
                    (last_rsi >= RSI_OVERBOUGHT) or  # RSI sobrecomprado
                    (last_ma_short < last_ma_long * 0.998)  # MA curta muito abaixo da longa
                )
                if close_signal:
                    print("🔄 Fechando posição de COMPRA (sinal de saída detectado)")
                    # Obter lucro no momento do fechamento (forçar refresh do cache)
                    pos = get_cached_positions(SYMBOL, force_refresh=True)
                    pos = [p for p in pos if p.ticket == current_position_ticket] if pos else None
                    if pos:
                        pos = pos[0]
                        profit_val = float(pos.profit)
                        update_trading_results('BUY', profit_val, pos.price_open, pos.price_current, signals['indicators_triggered'])
                    close_position(current_position_ticket)
                    time.sleep(2)
            
            elif current_position_type == 'sell':
                close_signal = (
                    (last_ma_short > last_ma_long and prev_ma_short <= prev_ma_long) or  # Cruzamento reverso
                    (last_rsi <= RSI_OVERSOLD) or  # RSI sobrevendido
                    (last_ma_short > last_ma_long * 1.002)  # MA curta muito acima da longa
                )
                if close_signal:
                    print("🔄 Fechando posição de VENDA (sinal de saída detectado)")
                    # Obter lucro no momento do fechamento (forçar refresh do cache)
                    pos = get_cached_positions(SYMBOL, force_refresh=True)
                    pos = [p for p in pos if p.ticket == current_position_ticket] if pos else None
                    if pos:
                        pos = pos[0]
                        profit_val = float(pos.profit)
                        update_trading_results('SELL', profit_val, pos.price_open, pos.price_current, signals['indicators_triggered'])
                    close_position(current_position_ticket)
                    time.sleep(2)

            # Aguardar até próxima barra do timeframe escolhido
            wait_seconds = calcular_tempo_proxima_barra(TIMEFRAME_MINUTES)
            print(f"⏳ Aguardando próxima barra {TIMEFRAME_NAME} (em ~{wait_seconds:.0f}s)...")
            time.sleep(min(wait_seconds, 30))  # Verificar a cada 30 segundos no máximo
            
        except KeyboardInterrupt:
            print("\n🛑 Robô sendo desligado pelo usuário...")
            close_all_positions()
            break
        except Exception as e:
            print(f"❌ Erro no loop principal: {e}")
            time.sleep(10)
            continue

# --- Executar o Robô ---
# 🎯 MELHORIAS IMPLEMENTADAS:
# ✅ Otimização usa dados do MT5 (mesmo timeframe que o robô)
# ✅ Treinamento ML usa dados do MT5 (consistência total)
# ✅ Robô prioriza dados do MT5 (mais preciso)
# ✅ Verificação de consistência dos dados
# ✅ Fallback para dados da web quando MT5 não disponível
# ✅ Modelo ML salvo/carregado automaticamente
# ✅ Mensagens informativas sobre fonte dos dados
# ✅ Teste de conexão e extração de dados
# ✅ Seleção dinâmica de indicadores pelo usuário
# ✅ Sistema de tracking de resultados de trading
# ✅ Relatórios em tempo real e resumo final
# ✅ Execução controlada por barra do timeframe escolhido pelo usuário
# ✅ Gestão de risco dinâmica baseada em ATR (cálculo manual)
# ✅ Cálculo automático de SL/TP e tamanho do lote
# ✅ Risco controlado por percentual do capital

if __name__ == "__main__":
    
    print("🤖 --- ROBÔ DE TRADING COM MT5 ---")
    print(f"📊 Símbolo: {SYMBOL}")
    print(f"⏰ Timeframe: {TIMEFRAME}")
    print(f"💰 Gestão de Risco: {RISK_PERCENT*100:.1f}% por trade")
    print(f"📈 ATR Period: {ATR_PERIOD}, SL Multiplier: {SL_MULTIPLIER}x, R:R: {RR_RATIO}:1")
    
    # Seleção de indicadores pelo usuário
    print("\n" + "="*50)
    if not select_indicators():
        print("❌ Seleção de indicadores cancelada. Encerrando...")
        exit()
    
    # Teste inicial de conexão e extração de dados
    if not test_mt5_connection():
        print("\n❌ Teste de conexão falhou. Verifique:")
        print("   1. MetaTrader 5 está instalado e aberto")
        print("   2. Suas credenciais de login estão corretas")
        print("   3. O símbolo XAUUSD está disponível na sua conta")
        print("   4. AutoTrading está habilitado no MT5")
        print("\n🔄 Tentando continuar com dados da web...")
    else:
        print("\n✅ Conexão com MT5 verificada e funcionando!")
    
    # Otimização
    best_params = optimize_parameters()
    if best_params:
        # Atualizar as variáveis globais com os melhores parâmetros encontrados
        # Não precisa de 'global' aqui pois estamos no nível do módulo (não dentro de função)
        MA_SHORT_PERIOD = best_params["MA_SHORT_PERIOD"]
        MA_LONG_PERIOD = best_params["MA_LONG_PERIOD"]
        RSI_OVERBOUGHT = best_params["RSI_OVERBOUGHT"]
        RSI_OVERSOLD = best_params["RSI_OVERSOLD"]
        print(f"\n✅ Variáveis globais atualizadas para a negociação com os melhores parâmetros:")
        print(f"   MA_SHORT_PERIOD = {MA_SHORT_PERIOD}")
        print(f"   MA_LONG_PERIOD = {MA_LONG_PERIOD}")
        print(f"   RSI_OVERBOUGHT = {RSI_OVERBOUGHT}")
        print(f"   RSI_OVERSOLD = {RSI_OVERSOLD}")

    # Treina o modelo de ML uma única vez usando dados do MT5
    print("\n--- Iniciando Treinamento do Modelo de ML ---")
    try:
        # Tentar carregar modelo salvo primeiro
        try:
            ML_MODEL = joblib.load(ML_MODEL_FILE)
            print(f"✅ Modelo carregado de: {ML_MODEL_FILE}")
        except:
            print("📝 Modelo não encontrado, treinando novo modelo...")
            ML_MODEL = None
        
        # Se não há modelo salvo, treinar novo
        if ML_MODEL is None:
            # Verificar se MT5 está disponível
            if mt5.terminal_info() is None:
                print("❌ MT5 não está conectado, usando dados da web para treinamento...")
                df_ml_train = get_web_data(SYMBOL, days=30)
                if not verify_data_consistency(df_ml_train, "Web"):
                    print("❌ Nenhum dado disponível para treinamento")
                    ML_MODEL = None
                else:
                    print("⚠️  Treinando com dados da web (menos preciso)")
                    ML_MODEL = train_ml_model(df_ml_train)
            else:
                print("🎯 Obtendo dados do MT5 para treinar o modelo...")
                df_ml_train = get_ohlc_data(SYMBOL, TIMEFRAME, 5000)  # 5000 barras para treinamento robusto
                
                if not verify_data_consistency(df_ml_train, "MT5"):
                    print("❌ Método original falhou, tentando método simples...")
                    df_ml_train = get_ohlc_data_simple(SYMBOL, TIMEFRAME, 5000)
                    
                    if not verify_data_consistency(df_ml_train, "MT5 (simples)"):
                        print("❌ Dados do MT5 não disponíveis para treinamento")
                        print("💡 Tentando dados da web como fallback...")
                        df_ml_train = get_web_data(SYMBOL, days=30)
                        if not verify_data_consistency(df_ml_train, "Web"):
                            print("❌ Nenhum dado disponível para treinamento")
                            ML_MODEL = None
                        else:
                            print("⚠️  Treinando com dados da web (menos preciso)")
                            ML_MODEL = train_ml_model(df_ml_train)
                    else:
                        print(f"✅ Treinando com dados do MT5 (método simples): {len(df_ml_train)} barras")
                        ML_MODEL = train_ml_model(df_ml_train)
                else:
                    print(f"✅ Treinando com dados do MT5: {len(df_ml_train)} barras")
                    ML_MODEL = train_ml_model(df_ml_train)
                
    except Exception as e:
        print(f"❌ Erro no treinamento do modelo de ML: {e}")
        ML_MODEL = None
        
    if ML_MODEL is None:
        print("⚠️  Aviso: O modelo de ML não foi treinado. O robô vai operar sem o filtro de ML.")
    else:
        print("✅ Modelo de ML carregado e pronto para uso!")

    try:
        run_bot()
    except KeyboardInterrupt:
        print("\n🛑 Robô sendo desligado pelo usuário...")
        print("🔄 Fechando todas as posições abertas...")
        close_all_positions()
    except Exception as e:
        print(f"❌ Erro inesperado no robô: {e}")
        print("🔄 Fechando todas as posições por segurança...")
        close_all_positions()
    finally:
        # Relatório final dos resultados
        print("\n" + "="*60)
        print_trading_summary()
        print("="*60)
        
        print("🔌 Desconectando do MetaTrader 5...")
        mt5.shutdown()
        print("✅ Desconectado do MetaTrader 5.")
        print("🤖 Robô finalizado com segurança!")