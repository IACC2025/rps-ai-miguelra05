# Documentacion de Recogida de Datos

**Alumno:** ___Miguel Ramire<___

## Formato del CSV

Tu archivo `data/partidas.csv` debe tener **minimo** estas columnas:

| Columna | Descripcion | Ejemplo |
|---------|-------------|---------|
| `numero_ronda` | Numero de la ronda (1, 2, 3...) | 1 |
| `jugada_j1` | Jugada del jugador 1 | piedra |
| `jugada_j2` | Jugada del jugador 2 (oponente) | papel |

### Ejemplo de CSV minimo:

```csv
numero_ronda,jugada_j1,jugada_j2
1,piedra,papel
2,tijera,piedra
3,papel,papel
4,piedra,tijera
...
```

---

## Como recogiste los datos?

Marca con [x] el metodo usado y describe brevemente:

### Metodo de recogida:

- [X] **Programa propio**: Cree un programa para jugar y guardar datos
- [ ] **Manual**: Jugue partidas y apunte los resultados a mano
- [ ] **Aplicacion/Web externa**: Use una app y exporte los datos
- [ ] **Otro**: _________________

### Descripcion del proceso:

```
(Explica aqui como recogiste los datos. Si usaste un programa,
describe brevemente como funciona. Si fue manual, explica el proceso.)

El programa era un juego basico de piedra papel tijera donde los dos jugadores
pulsaban sus respectivas teclas asd y jkl y dependiendo de la tecla pulsada por cada uno corresponde una jugada (piedra papel o tijera)
el programa calculaba el resultado y la racha todo se guardaba automaticamente en el csv y también se mostraba por pantalla toda la información
de la partida para hacerla mas realista e influir en la decisión de cada jugador.

```
---

## Datos adicionales capturados

Si capturaste datos extra ademas de los basicos, marcalos aqui:

- [ ] `tiempo_reaccion_ms` - Tiempo que tardo el jugador en responder
- [ ] `timestamp` - Fecha/hora de cada jugada
- [ ] `sesion` - ID de sesion de juego
- [x] `resultado` - victoria/derrota/empate
- [x] Otro: __Racha de victorias__

### Descripcion de datos adicionales:

```
(Si capturaste datos extra, explica aqui por que y como los usas)
La racha de victorias la utilice ya que si un jugador tiene una racha tiende a pensar mas los movimientos e influye su
decisión con el objetivo de seguir la racha. el resultado lo recolecte por que un jugador cambia su jugada dependiendo
de si ha ganado empatado o perdido.

```

---

## Estadisticas del dataset

- **Total de rondas:** _125_
- **Numero de sesiones/partidas:** __3__
- **Contra cuantas personas diferentes:** _2_

### Tipo de IA:

- [ ] **IA Especifica**: Entrenada para ganar a UNA persona concreta
  - Nombre/identificador del oponente: _________________
- [x] **IA General**: Entrenada para ganar a cualquier oponente
