# Protocolo experimental

| Parámetro | Valor |
|---|---|
| Casos cuantitativos | C02–C21; 20 pares sobre 7 mallas |
| Ejemplo cualitativo adicional | Q01, vaca→jirafa; manifiesto independiente |
| Semilla | 42, una optimización por caso/variante |
| Presupuesto | 2.500 iteraciones |
| Variables optimizadas | Jacobianos por cara; reconstrucción Poisson |
| Optimizador | Adam, lr 0,0025; betas/epsilon por defecto de PyTorch 1.13.1 |
| Guía | Uni3D-B + texto EVA02-E-14-plus, congelados en evaluación |
| Puntos | 10.000 XYZ, RGB constante 0,4; un batch de nube |
| Normalización | Media de la nube y radio máximo; radio mínimo 1e−6 |
| VC | 10 cámaras aleatorias por iteración; render OpenGL 512×512 |
| Features VC | CLIP ViT-B/32, stride 16, bloque 11, resize cúbico a 224 |
| Filtros VC | Diferencia de elevación <30°, azimut <20° |
| Inputs | Sin remallado ni rotación adicional; UVs generadas si faltan |

Los prompts, inputs y hashes están en `configs/cases.json` y
`configs/qualitative.json`. Los rangos de cámara, aumentos y ajuste de distancia
están en `configs/experiment.yml`. Resolución, cámaras y stride sólo afectan a VC
en las variantes Uni3D.

## Ejecución

Primero se completan los 20 casos de TextDeformer en su repositorio y entorno
originales. `scripts/run_textdeformer.py` invoca su `main.py` con el manifiesto
y los parámetros comunes: semilla 42, 2.500 iteraciones, lr 0,0025, 10 cámaras,
resolución 512 y stride 16. Las pérdidas, el optimizador y las salidas son los
de la implementación oficial, sin la protección numérica de UniTextDeformer.

Después se ejecutan los Experimentos 1, 2 y 3 con `run.py --variants`:
`uni3d-vc`, `uni3d-no-vc` y `uni3d-no-vc-no-delta`. Cada experimento completa sus
20 casos en procesos independientes, uno a la vez. Sin selección, `run.py`
completa las tres variantes en ese orden, con 60 optimizaciones.
Un error detiene el lanzador correspondiente. El recorrido completo suma 80
optimizaciones. Las secciones siguientes describen las variantes Uni3D.

## Pérdidas y muestreo

LP es el coseno negativo entre el embedding de la nube actual y el texto destino.
DeltaP es el coseno negativo entre las diferencias crudas de embeddings:
`geometría_actual − geometría_original` y `texto_destino − texto_origen`.
Se normalizan las diferencias después de restarlas. LI es la media de `(J − I)²`.
VC compara features CLIP de vértices visibles en pares de cámaras; su similitud
se resta del objetivo. Los pesos de cada variante se muestran en el README.

Se fijan caras según área de la malla original y coordenadas baricéntricas;
interpolarlas sobre los vértices actuales conserva el gradiente.
Dos muestreos independientes A/B usan un generador CPU con semilla 42.
La primera iteración compara A actual con B original; las siguientes usan A/A.
La inicialización del encoder y muestreo no altera el RNG global.
Sin DeltaP, su peso es cero y su valor se registra como diagnóstico.

La secuencia de cámaras incluye un batch adicional en la primera iteración y
un sorteo `randperm` cada 100 iteraciones. Estas operaciones fijan el consumo de
RNG y no añaden pérdidas. Las variantes sin VC no crean cámaras ni contexto gráfico.

## Protección numérica

`guarded_v1`, definida en `unitextdeformer/numerics.py`, usa embeddings FP32,
normas acumuladas en FP64 y epsilon 1e−6. Una dirección casi nula o un conjunto
VC vacío contribuyen cero. Se registra cualquier término inválido.

Gradientes con norma >1 se recortan. Un backward inválido usa el gradiente analítico
de LI; un estado Adam/geometría inválido se revierte. La recuperación reduce lr
por 0,5, con mínimo 1e−5. Los umbrales geométricos están en el mismo archivo.
Por eso **iteraciones y actualizaciones Adam aceptadas son conteos distintos**.
`run.json` y `numerical_events.jsonl` registran esas intervenciones.

## Exportación

La malla final contiene los vértices calculados antes del último intento de Adam.
`checkpoint.pt` guarda Jacobianos/Adam posteriores: ambos archivos representan
momentos distintos de la optimización. No se ofrece continuación automática.

Las rutas guardadas son relativas al repositorio. Los tiempos de `run.json`
corresponden al loop de optimización de esta implementación, excluyendo carga,
exportación y evaluación. Las figuras y valores agregados del README son referencias
experimentales; los tiempos nuevos dependen del hardware y la carga de ejecución.
