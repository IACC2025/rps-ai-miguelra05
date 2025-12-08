"""
RPSAI - Modelo de IA para Piedra, Papel o Tijera
=================================================

INSTRUCCIONES PARA EL ALUMNO:
-----------------------------
Este archivo contiene la plantilla para tu modelo de IA.
Debes completar las secciones marcadas con TODO.

El objetivo es crear un modelo que prediga la PROXIMA jugada del oponente
y responda con la jugada que le gana.

FORMATO DEL CSV (minimo requerido):
-----------------------------------
Tu archivo data/partidas.csv debe tener AL MENOS estas columnas:
    - numero_ronda: Numero de la ronda (1, 2, 3...)
    - jugada_j1: Jugada del jugador 1 (piedra/papel/tijera)
    - jugada_j2: Jugada del jugador 2/oponente (piedra/papel/tijera)

Ejemplo:
    numero_ronda,jugada_j1,jugada_j2
    1,piedra,papel
    2,tijera,piedra
    3,papel,papel

Si has capturado datos adicionales (tiempo_reaccion, timestamp, etc.),
puedes usarlos para crear features extra.

EVALUACION:
- 30% Extraccion de datos (documentado en DATOS.md)
- 30% Feature Engineering
- 40% Entrenamiento y funcionamiento del modelo

FLUJO:
1. Cargar datos del CSV
2. Crear features (caracteristicas predictivas)
3. Entrenar modelo(s)
4. Evaluar y seleccionar el mejor
5. Usar el modelo para predecir y jugar
"""

import os
import pickle
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

# Descomenta esta linea si te molesta el warning de sklearn sobre feature names:
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Importa aqui los modelos que vayas a usar
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Configuracion de rutas
RUTA_PROYECTO = Path(__file__).parent.parent
RUTA_DATOS = RUTA_PROYECTO / "data" / "partidas_auto.csv"
RUTA_MODELO = RUTA_PROYECTO / "models" / "modelo_entrenado.pkl"

# Mapeo de jugadas a numeros (para el modelo)
JUGADA_A_NUM = {"piedra": 0, "papel": 1, "tijera": 2}
NUM_A_JUGADA = {0: "piedra", 1: "papel", 2: "tijera"}

# Que jugada gana a cual
GANA_A = {"piedra": "tijera", "papel": "piedra", "tijera": "papel"}
PIERDE_CONTRA = {"piedra": "papel", "papel": "tijera", "tijera": "piedra"}


# =============================================================================
# PARTE 1: EXTRACCION DE DATOS (30% de la nota)
# =============================================================================

def cargar_datos(ruta_csv: str = None) -> pd.DataFrame:
    """
    Carga los datos del CSV de partidas.

    Args:
        ruta_csv: Ruta al archivo CSV (usa RUTA_DATOS por defecto)
    Returns:
        DataFrame con los datos de las partidas
    """
    #cargar csv
    if ruta_csv is None:
        ruta_csv = RUTA_DATOS
    #manejo en caso de que el archivo no exista
    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(f"El archivo {ruta_csv} no existe")
    #lee vel csv
    df = pd.read_csv(ruta_csv)
    #como mi csv tiene las columnas con un diferente nombre, aqui se ajusta el nombre de las columnas para hacer la comprobación
    mapeo_nombres = {}
    if 'ronda' in df.columns and 'numero_ronda' not in df.columns:
        mapeo_nombres['ronda'] = 'numero_ronda'
    if 'movimiento_j1' in df.columns and 'jugada_j1' not in df.columns:
        mapeo_nombres['movimiento_j1'] = 'jugada_j1'
    if 'movimiento_j2' in df.columns and 'jugada_j2' not in df.columns:
        mapeo_nombres['movimiento_j2'] = 'jugada_j2'
    if mapeo_nombres:
        df = df.rename(columns=mapeo_nombres)
        print(f"Columnas renombradas: {mapeo_nombres}")
    columnas_necesarias = ['numero_ronda', 'jugada_j1', 'jugada_j2']
    # comprueba las columnas esenciales y si falta alguna se manda un mensaje de error.
    for col in columnas_necesarias:
        if col not in df.columns:
            raise ValueError(f"Falta la columna requerida: '{col}'. Columnas disponibles: {df.columns.tolist()}")

    return df


def preparar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara los datos para el modelo.

    """
    df = df.copy()

    #Convertir jugadas a números
    df['jugada_j1_num'] = df['jugada_j1'].map(JUGADA_A_NUM)
    df['jugada_j2_num'] = df['jugada_j2'].map(JUGADA_A_NUM)

    # próxima jugada del oponente
    df['proxima_jugada_j2'] = df['jugada_j2_num'].shift(-1)

    df = df.dropna(subset=['proxima_jugada_j2'])

    return df


# =============================================================================
# PARTE 2: FEATURE ENGINEERING (30% de la nota)
# =============================================================================

def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea las features (caracteristicas) para el modelo.
    """
    df = df.copy()

    # FEATURE 1: Frecuencia de cada jugada (expanding y rolling)
    for jugada, num in JUGADA_A_NUM.items():
        # Frecuencia histórica total
        df[f'freq_{jugada}_total'] = (df['jugada_j2_num'] == num).expanding().mean().shift(1)

        # Frecuencia en ventana corta (últimas 5 rondas)
        df[f'freq_{jugada}_corto'] = df['jugada_j2_num'].rolling(
            window=5, min_periods=1
        ).apply(lambda x: (x == num).mean()).shift(1)

        # Frecuencia en ventana media (últimas 10 rondas)
        df[f'freq_{jugada}_medio'] = df['jugada_j2_num'].rolling(
            window=10, min_periods=1
        ).apply(lambda x: (x == num).mean()).shift(1)

    # FEATURE 2: Historial extendido (últimas 5 jugadas)
    for i in range(1, 6):
        df[f'jugada_t-{i}'] = df['jugada_j2_num'].shift(i)

    # FEATURE 3: Patrones de transición (qué hace después de secuencias)
    # Últimos 2 movimientos como patrón
    df['patron_2mov'] = df['jugada_j2_num'].shift(1).astype(str) + "_" + df['jugada_j2_num'].shift(2).astype(str)

    # Frecuencia de cada patrón de 2 movimientos
    patrones_2mov = df['patron_2mov'].unique()
    for patron in patrones_2mov:
        if isinstance(patron, str) and patron != 'nan_nan':
            df[f'patron_{patron}_freq'] = (df['patron_2mov'] == patron).expanding().mean().shift(1)

    # FEATURE 4: Comportamiento por resultado (mejorado)
    # Calcular resultado
    def calcular_resultado(j1, j2):
        if j1 == j2:
            return 0  # empate
        if (j1 == 0 and j2 == 2) or (j1 == 2 and j2 == 1) or (j1 == 1 and j2 == 0):
            return 1  # j1 gana
        return -1  # j2 gana

    df['resultado'] = df.apply(
        lambda row: calcular_resultado(row['jugada_j1_num'], row['jugada_j2_num']),
        axis=1
    )

    # Resultado anterior
    df['resultado_anterior'] = df['resultado'].shift(1)

    # Momentum (diferencia de victorias en últimas 3 rondas)
    df['momentum_3'] = df['resultado'].rolling(window=3, min_periods=1).sum()

    # ¿Qué juega después de ganar/perder/empatar? (one-hot por jugada)
    for res_nombre, res_valor in [('ganar', -1), ('perder', 1), ('empatar', 0)]:
        for jugada, num in JUGADA_A_NUM.items():
            col_name = f'despues_{res_nombre}_{jugada}'
            mask = df['resultado'].shift(1) == res_valor
            df[col_name] = 0
            df.loc[mask & (df['jugada_j2_num'] == num), col_name] = 1
            df[col_name] = df[col_name].expanding().mean().shift(1)

    # FEATURE 5: Racha y patrones de repetición (mejorado)
    if 'racha_actual' in df.columns:
        df['racha'] = df['racha_actual'].shift(1)
    else:
        df['racha'] = 0
        for i in range(1, len(df)):
            if df['jugada_j2_num'].iloc[i] == df['jugada_j2_num'].iloc[i - 1]:
                df['racha'].iloc[i] = df['racha'].iloc[i - 1] + 1

    # Probabilidad de cambiar tras racha larga
    df['prob_cambiar_racha'] = 0
    for i in range(1, len(df)):
        if df['racha'].iloc[i - 1] >= 2:
            df['prob_cambiar_racha'].iloc[i] = 1 if df['jugada_j2_num'].iloc[i] != df['jugada_j2_num'].iloc[
                i - 1] else 0
    df['prob_cambiar_racha'] = df['prob_cambiar_racha'].expanding().mean().shift(1)

    # FEATURE 6: Reacción a tu comportamiento
    # ¿Qué juega cuando TÚ repites?
    df['tu_repetiste'] = (df['jugada_j1_num'].shift(1) == df['jugada_j1_num'].shift(2)).astype(float)

    # Frecuencia por jugada cuando tú repites
    for jugada, num in JUGADA_A_NUM.items():
        col_name = f'vs_tu_repite_{jugada}'
        mask = df['tu_repetiste'] == 1
        df[col_name] = 0
        df.loc[mask & (df['jugada_j2_num'] == num), col_name] = 1
        df[col_name] = df[col_name].expanding().mean().shift(1)

    # FEATURE 7: Tendencia y cambios recientes
    # ¿Ha cambiado en las últimas 3 rondas?
    df['cambio_reciente'] = (
                                    (df['jugada_j2_num'] != df['jugada_j2_num'].shift(1)).astype(int) +
                                    (df['jugada_j2_num'].shift(1) != df['jugada_j2_num'].shift(2)).astype(int)
                            ) / 2

    # Tendencia dominante en ventana 3
    for jugada, num in JUGADA_A_NUM.items():
        df[f'tendencia_{jugada}_3'] = df['jugada_j2_num'].rolling(
            window=3, min_periods=1
        ).apply(lambda x: (x == num).mean()).shift(1)

    # FEATURE 8: Posición en el juego (si es relevante)
    if 'numero_ronda' in df.columns:
        max_ronda = df['numero_ronda'].max()
        df['fase_juego'] = df['numero_ronda'] / max_ronda if max_ronda > 0 else 0

    # Eliminar filas con NaN
    df = df.dropna()

    # Contar features creadas
    total_features = len([c for c in df.columns if any(x in c for x in
                                                       ['freq_', 'jugada_t-', 'patron_', 'despues_', 'tendencia_',
                                                        'racha', 'prob_', 'vs_tu_', 'cambio_', 'fase_', 'resultado_',
                                                        'momentum_'])])

    print(f"✅ Features creadas: {total_features} características en {len(df)} filas")

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.
    """
    print("\n🎯 Seleccionando features óptimas...")

    # Columnas que NO son features
    exclude_cols = [
        'numero_ronda', 'jugada_j1', 'jugada_j2',
        'jugada_j1_num', 'jugada_j2_num', 'proxima_jugada_j2',
        'resultado', 'patron_2mov'
    ]

    # Añadir columnas extras si existen
    for col in ['ganador', 'partida_id']:
        if col in df.columns:
            exclude_cols.append(col)

    # Todas las columnas excepto las excluidas son candidatas
    todas_features = [col for col in df.columns if col not in exclude_cols]

    # Selección basada en importancia (si hay muchas features)
    if len(todas_features) > 20:
        from sklearn.ensemble import RandomForestClassifier

        # Entrenar modelo rápido para importancia
        X_temp = df[todas_features]
        y_temp = df['proxima_jugada_j2']

        # Muestreo para velocidad
        if len(X_temp) > 500:
            sample_idx = np.random.choice(len(X_temp), 500, replace=False)
            X_sample = X_temp.iloc[sample_idx]
            y_sample = y_temp.iloc[sample_idx]
        else:
            X_sample = X_temp
            y_sample = y_temp

        rf_temp = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        rf_temp.fit(X_sample, y_sample)

        # Obtener importancia
        importancias = pd.DataFrame({
            'feature': todas_features,
            'importancia': rf_temp.feature_importances_
        }).sort_values('importancia', ascending=False)

        # Seleccionar top features
        n_features = min(25, len(todas_features))
        feature_cols = importancias.head(n_features)['feature'].tolist()

        print(f"   Seleccionadas top {n_features}/{len(todas_features)} features por importancia")
        print(f"   Top 5: {feature_cols[:5]}")

    else:
        feature_cols = todas_features
        print(f"   Usando todas las {len(feature_cols)} features")

    # Crear X e y
    X = df[feature_cols].values
    y = df['proxima_jugada_j2'].values

    print(f"   X shape: {X.shape}, y shape: {y.shape}")

    return X, y

# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de predicción mejorado.
    """
    print("\n" + "=" * 60)
    print("🎯 ENTRENANDO MODELOS MEJORADOS")
    print("=" * 60)

    # 1. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y
    )

    print(f"📊 Datos divididos:")
    print(f"   • Entrenamiento: {X_train.shape[0]} muestras")
    print(f"   • Prueba: {X_test.shape[0]} muestras")
    print(f"   • Features: {X_train.shape[1]}")

    # 2. Definir modelos mejorados
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score

    modelos = {
        'Random Forest (profundo)': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        ),
        'KNN (adaptativo)': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            p=2
        ),
        'Árbol optimizado': DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='log2',
            random_state=42
        )
    }

    # 3. Entrenar y evaluar con validación cruzada
    resultados = {}
    mejor_modelo = None
    mejor_nombre = ""
    mejor_accuracy = 0

    print(f"\n🔬 Probando {len(modelos)} modelos con validación cruzada:")

    for nombre, modelo in modelos.items():
        print(f"\n   🔄 {nombre}")

        try:
            # Validación cruzada para estimar rendimiento
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            print(f"      📊 CV Score (5-fold): {cv_mean:.3f} ± {cv_std:.3f}")

            # Entrenar con todos los datos de train
            modelo.fit(X_train, y_train)

            # Evaluar en test
            y_pred = modelo.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Guardar resultados
            resultados[nombre] = {
                'modelo': modelo,
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

            print(f"      ✅ Accuracy test: {accuracy:.3f}")

            # Seleccionar el mejor (pesando CV y test)
            score_total = cv_mean * 0.4 + accuracy * 0.6  # Ponderación

            if score_total > mejor_accuracy:
                mejor_accuracy = score_total
                mejor_modelo = modelo
                mejor_nombre = nombre

        except Exception as e:
            print(f"      ❌ Error: {str(e)[:50]}...")

    # 4. Resultados detallados
    print("\n" + "=" * 60)
    print("📈 RESULTADOS DETALLADOS")
    print("=" * 60)

    print("\nModelo                    | CV Mean  | CV Std   | Test Acc | Score")
    print("-" * 70)

    for nombre, datos in resultados.items():
        score = datos['cv_mean'] * 0.4 + datos['accuracy'] * 0.6
        print(
            f"{nombre:25} | {datos['cv_mean']:.3f}    | {datos['cv_std']:.3f}    | {datos['accuracy']:.3f}    | {score:.3f}")

    print("\n" + "=" * 60)
    print(f"🏆 MEJOR MODELO: {mejor_nombre}")

    if mejor_nombre in resultados:
        print(f"   • Accuracy test: {resultados[mejor_nombre]['accuracy']:.3f}")
        print(f"   • CV Score: {resultados[mejor_nombre]['cv_mean']:.3f} ± {resultados[mejor_nombre]['cv_std']:.3f}")

    # Análisis de rendimiento
    baseline = 1 / 3
    mejora_test = ((resultados[mejor_nombre]['accuracy'] / baseline) - 1) * 100 if mejor_nombre in resultados else 0

    print(f"\n ANÁLISIS DE RENDIMIENTO:")
    print(f"   • Baseline (aleatorio): {baseline:.3f}")
    print(f"   • Mejora sobre baseline: {mejora_test:+.1f}%")


    return mejor_modelo


def guardar_modelo(modelo, ruta: str = None):
    """Guarda el modelo entrenado en un archivo."""
    if ruta is None:
        ruta = RUTA_MODELO

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(modelo, f)
    print(f"Modelo guardado en: {ruta}")


def cargar_modelo(ruta: str = None):
    """Carga un modelo previamente entrenado."""
    if ruta is None:
        ruta = RUTA_MODELO

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontro el modelo en: {ruta}")

    with open(ruta, "rb") as f:
        return pickle.load(f)


# =============================================================================
# PARTE 4: PREDICCION Y JUEGO
# =============================================================================

class JugadorIA:
    """
    Clase que encapsula el modelo para jugar.

    TODO: Completa esta clase para que pueda:
    - Cargar un modelo entrenado
    - Mantener historial de la partida actual
    - Predecir la proxima jugada del oponente
    - Decidir que jugada hacer para ganar
    """

    def __init__(self, ruta_modelo: str = None):
        """Inicializa el jugador IA."""
        self.modelo = None
        self.historial = []  # Lista de (jugada_j1, jugada_j2)

        # TODO: Carga el modelo si existe
        # try:
        #     self.modelo = cargar_modelo(ruta_modelo)
        # except FileNotFoundError:
        #     print("Modelo no encontrado. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        """
        Registra una ronda jugada para actualizar el historial.

        Args:
            jugada_j1: Jugada del jugador 1
            jugada_j2: Jugada del oponente
        """
        self.historial.append((jugada_j1, jugada_j2))

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las features basadas en el historial actual.

        TODO: Implementa esta funcion
        - Usa el historial para calcular las mismas features que usaste en entrenamiento
        - Retorna un array con las features

        Returns:
            Array con las features para la prediccion
        """
        # TODO: Calcula las features basadas en self.historial
        # Deben ser LAS MISMAS features que usaste para entrenar

        pass  # Elimina esta linea cuando implementes

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.

        TODO: Implementa esta funcion
        - Usa obtener_features_actuales() para obtener las features
        - Usa el modelo para predecir
        - Convierte la prediccion numerica a texto

        Returns:
            Jugada predicha del oponente (piedra/papel/tijera)
        """
        if self.modelo is None:
            # Si no hay modelo, juega aleatorio
            return np.random.choice(["piedra", "papel", "tijera"])

        # TODO: Implementa la prediccion
        # features = self.obtener_features_actuales()
        # prediccion = self.modelo.predict([features])[0]
        # return NUM_A_JUGADA[prediccion]

        pass  # Elimina esta linea cuando implementes

    def decidir_jugada(self) -> str:
        """
        Decide que jugada hacer para ganar al oponente.

        Returns:
            La jugada que gana a la prediccion del oponente
        """
        prediccion_oponente = self.predecir_jugada_oponente()

        if prediccion_oponente is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        # Juega lo que le gana a la prediccion
        return PIERDE_CONTRA[prediccion_oponente]


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================

def main():
    """
    Funcion principal para entrenar el modelo.

    Ejecuta: python src/modelo.py
    """
    print("=" * 50)
    print("   RPSAI - Entrenamiento del Modelo")
    print("=" * 50)

    # 1. Cargar datos
    df = cargar_datos()

    # 2. Preparar datos
    df_preparado = preparar_datos(df)

    # 3. Crear features
    df_features = crear_features(df_preparado)

    # 4. Seleccionar features
    X, y = seleccionar_features(df_features)

    # 5. Entrenar modelo
    modelo = entrenar_modelo(X, y)

    # 6. Guardar modelo
    guardar_modelo(modelo)

    print("\n✅ Modelo entrenado y guardado exitosamente!")

if __name__ == "__main__":
    main()