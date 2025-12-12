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
    Crea las features (características) para el modelo.
    SOLO 8 FEATURES BÁSICAS para garantizar consistencia.
    """
    df = df.copy()

    # FEATURE 1-3: Frecuencias básicas de las últimas 5 jugadas
    for jugada, num in JUGADA_A_NUM.items():
        df[f'freq_{jugada}_5'] = df['jugada_j2_num'].rolling(
            window=5, min_periods=1
        ).apply(lambda x: (x == num).mean()).shift(1)

    # FEATURE 4: Última jugada del oponente
    df['ultima_jugada'] = df['jugada_j2_num'].shift(1)

    # FEATURE 5: ¿Repitió la última jugada?
    df['repetido'] = (df['jugada_j2_num'].shift(1) == df['jugada_j2_num'].shift(2)).astype(float)

    # FEATURE 6: Resultado de la ronda anterior
    df['resultado'] = df.apply(
        lambda row: 1 if GANA_A[row['jugada_j1']] == row['jugada_j2'] else
        (-1 if GANA_A[row['jugada_j2']] == row['jugada_j1'] else 0),
        axis=1
    )
    df['resultado_anterior'] = df['resultado'].shift(1)

    # FEATURE 7: Fase del juego (normalizada)
    if 'numero_ronda' in df.columns:
        df['fase_juego'] = df['numero_ronda'] / 50.0
    else:
        df['fase_juego'] = df.index / 50.0

    # FEATURE 8: Tendencia (qué jugada domina en últimas 3)
    # Calculamos cuál es la jugada más frecuente en últimas 3
    def tendencia_dominante(x):
        if len(x) == 0:
            return 0
        counts = np.bincount(x.astype(int), minlength=3)
        return np.argmax(counts)  # 0, 1 o 2

    df['tendencia_3'] = df['jugada_j2_num'].rolling(
        window=3, min_periods=1
    ).apply(tendencia_dominante).shift(1)

    # Eliminar filas con NaN
    df = df.dropna()

    print(f" Features creadas: 8 características básicas en {len(df)} filas")

    return df


def seleccionar_features(df: pd.DataFrame) -> tuple:
    """
    Selecciona las features para entrenar y el target.
    Usa TODAS las 8 features básicas.
    """
    print("\n Seleccionando features...")

    # SOLO estas 8 features básicas
    feature_cols = [
        'freq_piedra_5', 'freq_papel_5', 'freq_tijera_5',
        'ultima_jugada', 'repetido', 'resultado_anterior',
        'fase_juego', 'tendencia_3'
    ]

    # Verificar que existen
    for col in feature_cols:
        if col not in df.columns:
            print(f"⚠️  Advertencia: Feature '{col}' no encontrada")

    # Crear X e y
    X = df[feature_cols].values
    y = df['proxima_jugada_j2'].values

    print(f"   Usando {len(feature_cols)} features básicas")
    print(f"   X shape: {X.shape}, y shape: {y.shape}")

    return X, y

# =============================================================================
# PARTE 3: ENTRENAMIENTO Y FUNCIONAMIENTO (40% de la nota)
# =============================================================================

def entrenar_modelo(X, y, test_size: float = 0.2):
    """
    Entrena el modelo de predicción.
    """
    print("\n" + "=" * 60)
    print("ENTRENANDO MODELOS ")
    print("=" * 60)

    # 1. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True, stratify=y
    )

    print(f" Datos divididos:")
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

    print(f"\n Probando {len(modelos)} modelos con validación cruzada:")

    for nombre, modelo in modelos.items():
        print(f"\n    {nombre}")

        try:
            # Validación cruzada para estimar rendimiento
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            print(f"       CV Score (5-fold): {cv_mean:.3f} ± {cv_std:.3f}")

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

            print(f"       Accuracy test: {accuracy:.3f}")

            # Seleccionar el mejor (pesando CV y test)
            score_total = cv_mean * 0.4 + accuracy * 0.6  # Ponderación

            if score_total > mejor_accuracy:
                mejor_accuracy = score_total
                mejor_modelo = modelo
                mejor_nombre = nombre

        except Exception as e:
            print(f"       Error: {str(e)[:50]}...")

    # 4. Resultados detallados
    print("\n" + "=" * 60)
    print(" RESULTADOS DETALLADOS")
    print("=" * 60)

    print("\nModelo                    | CV Mean  | CV Std   | Test Acc | Score")
    print("-" * 70)

    for nombre, datos in resultados.items():
        score = datos['cv_mean'] * 0.4 + datos['accuracy'] * 0.6
        print(
            f"{nombre:25} | {datos['cv_mean']:.3f}    | {datos['cv_std']:.3f}    | {datos['accuracy']:.3f}    | {score:.3f}")

    print("\n" + "=" * 60)
    print(f"MEJOR MODELO: {mejor_nombre}")

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

        try:
            self.modelo = cargar_modelo(ruta_modelo)
        except FileNotFoundError:
            print("Modelo no encontrado. Entrena primero.")

    def registrar_ronda(self, jugada_j1: str, jugada_j2: str):
        self.historial.append((jugada_j1, jugada_j2))
        if len(self.historial) > 50:
            self.historial = self.historial[-50:]

    def obtener_features_actuales(self) -> np.ndarray:
        """
        Genera las 8 features básicas basadas en el historial actual.
        """
        # Si no hay historial, retornar valores por defecto
        if len(self.historial) < 2:
            # 8 features con valores por defecto
            return np.array([[0.333, 0.333, 0.333, 0, 0, 0, 0.5, 0]]).reshape(1, -1)

        # Convertir historial a numérico
        jugadas_j1 = []
        jugadas_j2 = []
        for j1, j2 in self.historial:
            jugadas_j1.append(JUGADA_A_NUM[j1])
            jugadas_j2.append(JUGADA_A_NUM[j2])

        jugadas_j1 = np.array(jugadas_j1)
        jugadas_j2 = np.array(jugadas_j2)

        # --------------------------------------------------------------------
        # CALCULAR LAS 8 FEATURES BÁSICAS
        # --------------------------------------------------------------------

        # FEATURE 1-3: Frecuencias en últimas 5 jugadas
        ultimas_5 = jugadas_j2[-5:] if len(jugadas_j2) >= 5 else jugadas_j2
        freq_piedra = np.sum(ultimas_5 == 0) / max(len(ultimas_5), 1)
        freq_papel = np.sum(ultimas_5 == 1) / max(len(ultimas_5), 1)
        freq_tijera = np.sum(ultimas_5 == 2) / max(len(ultimas_5), 1)

        # FEATURE 4: Última jugada
        ultima_jugada = jugadas_j2[-1] if len(jugadas_j2) > 0 else 0

        # FEATURE 5: ¿Repitió?
        repetido = 0
        if len(jugadas_j2) >= 2:
            repetido = 1 if jugadas_j2[-1] == jugadas_j2[-2] else 0

        # FEATURE 6: Resultado anterior
        resultado_anterior = 0
        if len(self.historial) >= 2:
            j1_ant, j2_ant = self.historial[-2]
            if GANA_A[j1_ant] == j2_ant:  # Ganó J1
                resultado_anterior = 1
            elif GANA_A[j2_ant] == j1_ant:  # Ganó J2
                resultado_anterior = -1
            else:
                resultado_anterior = 0

        # FEATURE 7: Fase del juego
        fase_juego = min(len(self.historial) / 50.0, 1.0)

        # FEATURE 8: Tendencia dominante en últimas 3
        tendencia_3 = 0
        if len(jugadas_j2) >= 1:
            ultimas_3 = jugadas_j2[-3:] if len(jugadas_j2) >= 3 else jugadas_j2
            counts = np.bincount(ultimas_3.astype(int), minlength=3)
            tendencia_3 = np.argmax(counts)

        # --------------------------------------------------------------------
        # CREAR ARRAY CON LAS 8 FEATURES EN ORDEN CORRECTO
        # --------------------------------------------------------------------
        features = np.array([[
            freq_piedra,  # Feature 1
            freq_papel,  # Feature 2
            freq_tijera,  # Feature 3
            ultima_jugada,  # Feature 4
            repetido,  # Feature 5
            resultado_anterior,  # Feature 6
            fase_juego,  # Feature 7
            tendencia_3  # Feature 8
        ]], dtype=np.float64)

        # Asegurar que no hay NaN
        features = np.nan_to_num(features, nan=0.0)

        return features

    def predecir_jugada_oponente(self) -> str:
        """
        Predice la proxima jugada del oponente.
        """
        if self.modelo is None:
            return np.random.choice(["piedra", "papel", "tijera"])

        features = self.obtener_features_actuales()

        # Siempre retorna algo (no necesita chequeo)
        try:
            prediccion_num = self.modelo.predict(features)[0]
            return NUM_A_JUGADA[prediccion_num]
        except Exception as e:
            print(f"Error en predicción: {e}")
            return np.random.choice(["piedra", "papel", "tijera"])


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
    X, y= seleccionar_features(df_features)

    # 5. Entrenar modelo
    modelo = entrenar_modelo(X, y)

    # 6. Guardar modelo
    guardar_modelo(modelo)

    print("\n✅ Modelo entrenado y guardado exitosamente!")

if __name__ == "__main__":
    main()