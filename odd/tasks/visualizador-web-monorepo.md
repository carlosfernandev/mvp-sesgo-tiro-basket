# Feature: Visualizador web de análisis de tiros (monorepo)

## Objetivo
Subir un video MP4 desde el navegador, procesarlo **sin guardarlo en disco**, y ver el resultado gráficamente: video reproducido en RAM, curvas de ángulos y línea de fases. Los CSV quedan persistidos y son consultables después desde un menú **"Historial de registros"**.

## Decisión de producto (CAMBIO DE ALCANCE)
El diseño original era "solo visualizador": la API servía artefactos ya generados. El usuario cambió el alcance a **subir + procesar**, con la restricción explícita de **no persistir el video**.

Vía elegida: **B — CSV permanente, video efímero**.
- El video **nunca** se escribe en disco. Vive en la RAM del browser (Blob URL) durante la sesión.
- El CSV sí se persiste en `outputs/<id>/angulos_tiro.csv`.
- El PNG del perfil **tambien se deja de generar**: es derivable del CSV y la UI dibuja la curva con Recharts, así que un PNG en disco es redundante.
- Consecuencia de UX que hay que diseñar explícitamente: **el historial no tiene video**. Solo curvas y fases. La UI debe decirlo con un estado vacío informado, no dejar un player muerto.

## Restricción técnica verificada
El video no puede pasar por `cv2.VideoCapture` (solo acepta rutas). La tubería en memoria usa **ffmpeg por pipe**, y esto se verificó empíricamente antes de diseñar:

| Prueba | Resultado |
|---|---|
| MP4 → pipe → ffmpeg → frames crudos en RAM | OK (436 frames) |
| Frames → pipe → ffmpeg → MP4 H.264 | OK (153 KB, `ftyp` válido) |
| `libx264` | **NO existe** en este build |
| `libopenh264` | **disponible** — es el encoder H.264 a usar |
| MP4 normal hacia un pipe | **FALLA**: *"muxer does not support non seekable output"* |

Por eso el encode en memoria exige **MP4 fragmentado**: `-movflags frag_keyframe+empty_moov+default_base_moof`.

### Causa raíz del síntoma "el video no reproduce"
`outputs/tiro_demo/feedback.mp4` tiene `codec_name=mpeg4` (MPEG-4 **Parte 2**, el `mp4v` del script). **Chrome y Firefox no reproducen MPEG-4 Parte 2.** No era un bug de UI: era un bug de compatibilidad de codec. `libopenh264` lo resuelve.

## Problema
Hoy la salida de un análisis son tres artefactos sueltos y desconectados:
- `angulos_tiro.csv` — datos tabulares
- `perfil_angular.png` — un gráfico estático
- `*_feedback.mp4` — el video con overlay

Para entender un tiro el usuario tiene que saltar entre tres aplicaciones y correlacionar mentalmente frame ↔ tiempo ↔ curva. No hay forma de barrer el cursor sobre la curva y ver el video en ese frame, ni de ver en un vistazo dónde caen las fronteras de fase.

## Por qué ahora
El trabajo de segmentación ya está verificado y estable (44 tests). El cuello de botella dejó de ser el cálculo y pasó a ser la **inspección humana** de sus resultados. Esta feature ataca exactamente eso.

## Alcance (autorizado)

### Incluido
- Monorepo con npm workspaces.
- Servicio API ligero (Python/Flask) que **solo sirve artefactos ya generados**. No ejecuta el análisis.
- Frontend React + Vite que consume la API y dibuja:
  - reproductor de video con contador de frame
  - curva de ángulos con las 4 fases sombreadas
  - línea de tiempo de fases
  - sincronización: el cursor en la curva hace seek en el video
- Arranque en vacío: sin corridas, la UI lo dice explícitamente en vez de romper.

### Explícitamente fuera de alcance
- **Ejecución del análisis desde el browser** (subida de video, progreso, polling). Se decidió la variante "visualizador": el usuario corre el script Python por separado.
- **Calibración / marcado manual de fronteras** — Although the sweep-the-curve design is a prerequisite for it, marking boundaries is a separate feature. No se incluye acá.
- Autotrackeo de la pelota. MediaPipe Pose no rastrea la pelota; la UI **no** va a pretender otherwise.

## Contrato de datos (derivado de la salida real, no inventado)

`angulos_tiro.csv` tiene columnas exactas:
`frame`, `timestamp`, `ángulo_codo`, `ángulo_rodilla`, `ángulo_hombro`, `fase_asignada`

Valores de `fase_asignada` (español, fijo): `preparacion`, `elevacion`, `liberacion`, `seguimiento`, `sin_deteccion`.

Directorio de resultados: configurable. El default histórico es la raíz del repo (por eso `.gitignore` lista los artefactos sueltos). Se acepta un override por variable de entorno para no acoplar la API al repo.

## Tareas

- [x] **T1 — Scaffolding del monorepo.** Hecho.
- [x] **T2 — Servicio API ligero (Flask).** Hecho y verificado. 5 endpoints.
- [x] **T3 — Scaffold del frontend React + Vite.** Hecho. tsc limpio, build OK.
- [x] **T4 — Reproductor + línea de tiempo de fases.** Hecho.
- [x] **T5 — Curvas de ángulos sincronizadas.** Hecho.
- [x] **T6 — Verificación end-to-end.** API verificada; render en browser pendiente de confirmación humana.
- [ ] **T7 — Core del análisis sobre iterador de frames.** Refactor de `procesar_video` para extraer el loop MediaPipe a una función que acepte cualquier iterador de frames. **Sin tocar la lógica de ángulos ni de fases.** Debe preservar el comportamiento del CLI actual (71 tests).
- [ ] **T8 — Endpoint `POST /api/analizar`.** Acepta el MP4, decodifica por pipe ffmpeg, corre el core, **persiste solo el CSV** en `outputs/<id>/`, y devuelve en la respuesta: (a) el JSON de análisis y (b) el video anotado H.264 fragmentado en memoria. Errores claros: video ilegible, sin persona detectada, sin tiro.
- [ ] **T9 — Panel de subida en el frontend.** Drag & drop, validación de que sea MP4, barra de progreso real (el upload es lento: MediaPipe corre por frame), y render del resultado con el video efímero.
- [ ] **T10 — Menú "Historial de registros".** Lista de corridas persistidas, con la curva y las fases. **Estado vacío informado para el video**: "este registro no conserva el video".
- [ ] **T11 — Verificación de la ruta efímera.** Subir un video real, confirmar que no queda ningún archivo de video en `outputs/`, y que el video devuelto es reproducible en browser.

## Hallazgos de la integración (bugs reales encontrados y corregidos)

1. **`--salida-dir` no existía.** El contrato original asumía un flag que el script no tenía: los outputs iban a `--csv`/`--plot` con nombres fijos. Sin `--salida-dir` cada corrida **pisa la anterior** (`angulos_tiro.csv` siempre igual), lo que destruye justo la referencia que hace falta para calibrar. Se agregó el flag a `main()`: coloca los defaults dentro del directorio, respeta verbatim cualquier path explícito, y **no** habilita el video implícitamente (costaría una segunda pasada de inferencia).
2. **`VIDEO_GLOB = "*_feedback*.mp4"` no matcheaba nada real.** El comentario del código afirmaba que `generar_video_feedback()` siempre agrega el infijo `_feedback`: es falso, el nombre viene del path que pasa el usuario. Los archivos reales (`feedback.mp4`, `feedback2.mp4`) devolvían `has_video: false`. Corregido a `*feedback*.mp4` con el comentario corregido.

## Estado de verificación

Verificado con ejecución real:
- `.venv/bin/python -m unittest test_analisis_tiro` → **71 tests OK**
- API: `206 Partial Content` + `Content-Range` (seek funciona), traversal `404`, sesión inexistente `404`, PNG `200 image/png`
- Claves acentuadas del CSV parsean bien en el JSON y en TypeScript
- Frontend: `tsc --noEmit` exit 0; `npm run build` OK

**NO verificado** (requiere browser humano): el render real, la suavidad del hover-seek y la reproducción del video. El dev server de Vite reporta `ready` pero no acepta conexiones en este entorno, así que no se pudo ejercitar la app en runtime.

## Deuda técnica conocida
- Los 27 tests de la API quedaron dentro de `test_analisis_tiro.py`, mezclados con los del análisis. Deberían pasar a `packages/api/test_api.py`.

## Criterios de aceptación
- Una corrida real se ve en el browser sin errores de consola.
- El seek desde la curva mueve el video al frame correcto.
- Sin corridas, la UI lo dice en claro.
- `analisis_tiro.py` y sus 44 tests siguen intactos.
- El build de producción del frontend compila.

## Checks aplicables
- `.venv/bin/python -m unittest test_analisis_tiro` → debe seguir en 44 OK
- `npm run build` en el frontend → debe compilar
- Prueba manual: API responde, frontend carga, video hace seek

## Progreso

_(se completa a medida que se avanzan las tareas)_

## Ruta y triggers de delegación
- **T1, T2** —writer delegado (2+ archivos no triviales, contrato de datos nuevo)
- **T3, T4, T5** —writer delegado (frontend, múltiples componentes)
- **T6** —worker de verificación dedicado

## Estado
- Feature creada. Pendiente de arrancar T1.
