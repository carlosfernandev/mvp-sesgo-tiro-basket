# Feature: UX de subida de video con soporte de parámetros (--side y --feedback)

## Objetivo
Reemplazar el flujo manual de consola por una experiencia de usuario (UX) integral en la plataforma web, permitiendo al usuario configurar el lado dominante (`--side`: auto, left, right) y la generación de video con anotaciones biomecánicas (`--feedback`), junto con la corrección del puerto proxy de Vite hacia el backend (5001).

## Problema
1. El proxy de desarrollo en `packages/web/vite.config.ts` apuntaba por defecto al puerto `5097`, mientras que la API corre en el `5001`.
2. El endpoint `POST /api/analizar` no aceptaba parámetros: forzaba `side="auto"` y siempre renderizaba el video con anotaciones (incluso si no se requería).
3. `UploadPanel.tsx` disparaba la subida inmediatamente al soltar el archivo, sin permitir al usuario configurar el tiro ni elegir si requiere el video con feedback visual o solo la telemetría biomecánica.

## Alcance Autorizado
- **Ajuste de configuración:** Cambiar el puerto por defecto de Vite en `packages/web/vite.config.ts` de `5097` a `5001`.
- **Backend (`packages/api/app.py`):**
  - Soporte de query parameters `side` (`auto`, `left`, `right`) y `feedback` (`true`, `false`) en `POST /api/analizar`.
  - Validación de valores y paso a `analizar_frames_con_cache(..., side=side)`.
  - Renderizado condicional de `_render_feedback_mp4`: si `feedback=false`, omitir el pipeline de video y retornar `video_url: null`.
  - Retorno del lado seleccionado (`chosen_side` o `side`) en la respuesta JSON.
  - Tests automatizados para verificar estos parámetros y validaciones.
- **Frontend (`packages/web`):**
  - Modelos y tipos en `types.ts` (`DominantSide`, campos en `AnalyzeResult`).
  - Cliente HTTP en `api.ts` pasando query params en `uploadVideo`.
  - Rediseño de `UploadPanel.tsx` y `styles.css`:
    - Selección/arrastre de archivo con estadificación visual (nombre, tamaño, estado listo).
    - Selector ergonómico de lado dominante (Auto, Diestro / Derecha, Zurdo / Izquierda).
    - Toggle de video de feedback (activado por defecto).
    - Botón de acción explícito "Analizar tiro".
    - Estados visuales claros de progreso y error.
- **Verificación:** Ejecución de tests Python existentes y nuevos, más build del frontend con TypeScript y Vite.

## Tareas

- [x] **T1 — Ajuste de puerto proxy en Vite.** Modificar `packages/web/vite.config.ts` para usar 5001 por defecto. (Completado: default proxy target actualizado a 5001 y verificado con `npm run build`).
- [x] **T2 — Soporte de `side` y `feedback` en `packages/api/app.py` con tests.** Aceptar y validar query params en `POST /api/analizar`, renderizado condicional de feedback, retorno de lado elegido y tests de API. (Completado: implementado en `packages/api/app.py`, 13 tests en `packages/api/test_app.py` y 71 tests existentes en verde).
- [ ] **T3 — Tipos y cliente API en frontend (`types.ts`, `api.ts`).** Agregar `DominantSide` y opciones en `uploadVideo`.
- [ ] **T4 — Rediseño de UX en `UploadPanel.tsx` y estilos en `styles.css`.** Estadificación de archivo, selector de lado, toggle de feedback y botón de ejecución con feedback visual.
- [ ] **T5 — Verificación integral.** Ejecutar suite de tests backend (`pytest` / `unittest`) y build frontend (`npm run build`).

## Criterios de Aceptación
1. `npm run dev --workspace packages/web` conecta de forma transparente con Flask en el puerto `5001`.
2. `POST /api/analizar?side=left&feedback=false` analiza desde el lado izquierdo sin renderizar video de feedback y retorna `video_url: null`.
3. Parámetro `side` inválido responde HTTP 400 con mensaje de error explicativo.
4. El panel de subida web permite seleccionar archivo, configurar lado dominante y feedback, y disparar el análisis con un botón claro.
5. `npm run build` en `packages/web` compila sin errores de TypeScript ni bundling.
6. La suite de pruebas de Python pasa exitosamente.
