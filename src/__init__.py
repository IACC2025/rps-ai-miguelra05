# RPSAI - Modulo principal
import csv
import os
import time


class JuegoPPT:
    def __init__(self, archivo_csv='../data/datos.csv'):
        self.archivo_csv = archivo_csv
        self.ronda_actual = 1
        self.ultimo_ganador = None
        self.racha_actual = 0

        self.resultados = {
            ('piedra', 'tijera'): 'j1',
            ('tijera', 'papel'): 'j1',
            ('papel', 'piedra'): 'j1',
            ('tijera', 'piedra'): 'j2',
            ('papel', 'tijera'): 'j2',
            ('piedra', 'papel'): 'j2',
            ('piedra', 'piedra'): 'empate',
            ('papel', 'papel'): 'empate',
            ('tijera', 'tijera'): 'empate'
        }

        # Mapeo de teclas a movimientos
        self.teclas_j1 = {'a': 'piedra', 's': 'papel', 'd': 'tijera'}
        self.teclas_j2 = {'j': 'piedra', 'k': 'papel', 'l': 'tijera'}

        # Crear archivo CSV
        if not os.path.exists(self.archivo_csv):
            with open(self.archivo_csv, 'w', newline='', encoding='utf-8') as archivo:
                escritor = csv.writer(archivo)
                escritor.writerow([
                    'partida_id', 'ronda',
                    'movimiento_j1', 'movimiento_j2',
                    'ganador', 'racha_actual'
                ])

        # Crear partida automáticamente
        self.partida_actual = f"partida_{int(time.time())}"
        print(f"PARTIDA INICIADA: {self.partida_actual}")
        print("  Jugador 1: a = piedra, s = papel, d = tijera")
        print("  Jugador 2: j = piedra, k = papel, l = tijera")
        print("  n = nueva partida, v = ver datos, q = salir")
        print("\n  Ingresa las teclas en CUALQUIER orden (ej: 'aj', 'ja', 'sk', 'ks')")

    def determinar_ganador(self, movimiento_j1, movimiento_j2):
        return self.resultados.get((movimiento_j1, movimiento_j2), 'error')

    def registrar_jugada(self, tecla_j1, tecla_j2):
        # Convertir teclas a movimientos
        movimiento_j1 = self.teclas_j1[tecla_j1]
        movimiento_j2 = self.teclas_j2[tecla_j2]

        # Determinar ganador automáticamente
        ganador = self.determinar_ganador(movimiento_j1, movimiento_j2)

        # Actualizar racha
        self.actualizar_racha(ganador)

        # Guardar en CSV
        with open(self.archivo_csv, 'a', newline='', encoding='utf-8') as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow([
                self.partida_actual,
                self.ronda_actual,
                movimiento_j1,
                movimiento_j2,
                ganador,
                self.racha_actual
            ])

        # Mostrar resultado
        self.mostrar_resultado(movimiento_j1, movimiento_j2, ganador)

        self.ronda_actual += 1
        return True

    def mostrar_resultado(self, mov_j1, mov_j2, ganador):
        """Muestra el resultado de forma legible"""
        print(f"\n{'-' * 40}")
        print(f"Ronda {self.ronda_actual}:")
        print(f"  J1: {mov_j1.upper()}  vs  J2: {mov_j2.upper()}")

        if ganador == 'empate':
            print(f"  Resultado: ¡EMPATE!")
        elif ganador == 'j1':
            print(f"  Resultado: ¡J1 GANA! ({mov_j1} gana a {mov_j2})")
        else:  # j2
            print(f"  Resultado: ¡J2 GANA! ({mov_j2} gana a {mov_j1})")

        print(f"  Racha actual: {self.racha_actual}")
        print(f"{'-' * 40}")

    def actualizar_racha(self, ganador):
        if ganador == 'empate':
            self.racha_actual = 0
        elif ganador == self.ultimo_ganador:
            self.racha_actual += 1
        else:
            self.racha_actual = 1
            self.ultimo_ganador = ganador

    def nueva_partida(self):
        """Para cambiar de jugadores o reiniciar"""
        self.partida_actual = f"partida_{int(time.time())}"
        self.ronda_actual = 1
        self.ultimo_ganador = None
        self.racha_actual = 0
        print(f"\n{'═' * 40}")
        print(f" NUEVA PARTIDA: {self.partida_actual} ")
        print(f"{'═' * 40}")

    def ver_datos(self):
        """Muestra los datos guardados"""
        if not os.path.exists(self.archivo_csv):
            print("No hay datos guardados aún.")
            return

        print(f"\n{'═' * 60}")
        print("DATOS GUARDADOS:")
        print(f"{'═' * 60}")

        try:
            with open(self.archivo_csv, 'r', encoding='utf-8') as f:
                # Leer todo el contenido
                contenido = f.read()
                print(contenido)
        except Exception as e:
            print(f"Error al leer el archivo: {e}")

    def procesar_entrada(self, entrada):
        entrada = entrada.lower().strip()

        # Comandos especiales de un solo carácter
        if len(entrada) == 1:
            if entrada == 'q':
                print("¡Hasta luego!")
                return False
            elif entrada == 'n':
                self.nueva_partida()
                return True
            elif entrada == 'v':
                self.ver_datos()
                return True
            # Si es solo una tecla válida pero no comando, simplemente ignorar
            return True

        # Para entradas de 2 o más caracteres
        if len(entrada) >= 2:
            # Buscar UNA tecla de J1 y UNA tecla de J2 en CUALQUIER orden
            teclas_j1_encontradas = []
            teclas_j2_encontradas = []

            # Recolectar todas las teclas válidas
            for tecla in entrada:
                if tecla in self.teclas_j1:
                    teclas_j1_encontradas.append(tecla)
                elif tecla in self.teclas_j2:
                    teclas_j2_encontradas.append(tecla)

            # Si tenemos al menos una tecla de cada jugador
            if teclas_j1_encontradas and teclas_j2_encontradas:
                # Tomar la primera tecla de cada jugador encontrada
                tecla_j1 = teclas_j1_encontradas[0]
                tecla_j2 = teclas_j2_encontradas[0]
                self.registrar_jugada(tecla_j1, tecla_j2)
                return True

        # Si no se encontró combinación válida, ignorar silenciosamente
        return True


# PROGRAMA PRINCIPAL
def main():
    juego = JuegoPPT('../data/partidas_auto.csv')

    print("\n PIEDRA, PAPEL O TIJERA ")
    print("  (Las teclas se pueden ingresar en cualquier orden)")

    while True:
        print(f"\nRonda {juego.ronda_actual} - Esperando entrada...")

        try:
            # Leer entrada del usuario
            entrada = input("> ")

            # Procesar la entrada
            if not juego.procesar_entrada(entrada):
                break  # Salir si procesar_entrada retorna False

        except KeyboardInterrupt:
            print("\n\nPrograma interrumpido. ¡Hasta luego!")
            break
        except EOFError:
            print("\nFin de entrada. ¡Hasta luego!")
            break


if __name__ == "__main__":
    main()