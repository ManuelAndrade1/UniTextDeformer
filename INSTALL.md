# Instalación

Comandos desde la raíz `UniTextDeformer`, en Linux x86-64 con Bash.
Se necesita driver NVIDIA operativo, GPU compatible con CUDA 11.7, compilador
C++/CUDA y bibliotecas OpenGL/EGL. Las extensiones se compilan para la GPU
seleccionada mediante `CUDA_VISIBLE_DEVICES`.

## 1. Entorno

Instalar Conda/Micromamba previamente. Esta receta conserva Python 3.9,
PyTorch 1.13.1, CUDA 11.7 y las versiones usadas por la adaptación.

```bash
conda create -y -n unitextdeformer -c nvidia -c conda-forge \
  --channel-priority flexible python=3.9 pip=24.0 numpy=1.26.4 scipy=1.13.1 igl=2.6.1 \
  gcc_linux-64=11 gxx_linux-64=11 cmake make ninja pkg-config libgl-devel libegl-devel \
  cuda-nvcc=11.7 cuda-cudart-dev=11.7 cuda-cudart=11.7 cuda-cccl=11.7 \
  libcublas-dev=11.10.3.66 libcusparse-dev=11.7.4.91 libcusolver-dev=11.4.0.1
conda activate unitextdeformer
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_extensions"
export MAX_JOBS=2
export OMP_NUM_THREADS=4
mkdir -p "$TORCH_EXTENSIONS_DIR"
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements.txt
python -m pip install --no-index \
  torch-scatter==2.1.1+pt113cu117 torch-sparse==0.6.17+pt113cu117 \
  -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

Las ruedas CUDA de PyG se toman de su [índice oficial](https://data.pyg.org/whl/torch-1.13.1+cu117.html).
En el entorno `unitextdeformer`, no instalar los `requirements.txt` completos
de Uni3D/TextDeformer: contienen componentes de entrenamiento/evaluación que
esta adaptación no utiliza.

## 2. Fuentes oficiales fijadas

```bash
mkdir -p external
git clone https://github.com/threedle/TextDeformer.git external/TextDeformer
git -C external/TextDeformer checkout f5302c66c0e5cbbd2fde3ad2aac2aec8f842924c
git clone https://github.com/baaivision/Uni3D.git external/Uni3D
git -C external/Uni3D checkout 64e03c3c42c196e8cb5ed03857810af9fc9ac39c
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git external/Pointnet2_PyTorch
git -C external/Pointnet2_PyTorch checkout b5ceb6d9ca0467ea34beb81023f96ee82228f626
git clone https://github.com/NVlabs/nvdiffrast.git external/nvdiffrast_official
git -C external/nvdiffrast_official checkout 78528e683210dfaa1be57e3c65aa37d3b36c6644
```

Las fuentes y pesos exactos se describen en `configs/assets.json`.
Uni3D se usa como fuente de arquitectura; no requiere instalar su paquete ni
DeepSpeed, datasets o checkpoints de entrenamiento adicionales.

## 3. Inputs, compatibilidad y extensiones

```bash
python scripts/prepare.py
python -m pip install --no-build-isolation --no-deps ./external/nvdiffrast_official
python -m pip install --no-build-isolation --no-deps ./.build/pointnet2_ops_lib
```

`prepare.py` extrae los archivos imprescindibles de TextDeformer en
`.runtime/TextDeformer`, aplica `patches/textdeformer.patch` y verifica las siete
mallas incluidas en `meshes/` (restaura las faltantes). No rota ni repara
las entradas y no modifica los repositorios oficiales. También prepara PointNet2
para respetar la arquitectura de la GPU actual. Puede repetirse con los mismos inputs.

Compilar con la GPU destino visible para que PyTorch detecte su arquitectura.
También se puede definir `TORCH_CUDA_ARCH_LIST` antes de instalar las extensiones.
Los kernels de TextDeformer se compilan en el primer uso de VC.
Si el linker no encuentra `libcuda.so`, definir `CUDA_DRIVER_LIBRARY_DIR` con la
carpeta del driver NVIDIA. VC requiere `RasterizeGLContext` y soporte OpenGL/EGL
para renderizado sin pantalla.

## 4. Pesos preentrenados

```bash
python scripts/prepare.py --download-models
```

Descarga aproximadamente 10 GiB y verifica SHA-256. Requiere espacio adicional
para instalación/cachés y RAM para cargar la torre textual en CPU.
También se pueden descargar manualmente las URLs fijadas de `configs/assets.json`
y guardarlas en las rutas indicadas:

| Archivo | Uso |
|---|---|
| `models/uni3d/base.pt` | Uni3D-B |
| `models/uni3d/open_clip_config.json` | Configuración EVA02-E-14-plus |
| `models/uni3d/eva_clip.safetensors` | Pesos EVA-CLIP; se carga sólo `text.*` |
| `models/ViT-B-32.pt` | CLIP visual para VC |

El checkpoint Uni3D se carga estrictamente; la configuración textual debe coincidir
con EVA02-E-14-plus. No sustituir esta torre por otro CLIP. El loader acepta también
una extracción textual compatible que conserve las claves `text.*`; el descargador
verifica únicamente los archivos oficiales completos.

## 5. Verificar y correr

```bash
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
python -c 'import pointnet2_ops, nvdiffrast; print(nvdiffrast.__version__)'
python -m unittest discover -s tests -v
python run.py --cases C02 --steps 2 --output outputs/smoke
```

La prueba corta debe terminar las tres variantes Uni3D y producir `mesh_final/mesh.obj`
con `status: complete` en cada `run.json`. Conservar 10 cámaras y resolución 512
en esa prueba para comprobar los recursos de la configuración real.

Para ejecutar los 20 casos de referencia TextDeformer, preparar un entorno
`TextDeformer` según la [instalación oficial](https://github.com/threedle/TextDeformer#installation)
y usar el checkout original de `external/TextDeformer`. El lanzador invoca
directamente ese código, sin aplicar `patches/textdeformer.patch` al repositorio
original. Este entorno se instala con las dependencias de TextDeformer.
Completar esta ejecución antes de las variantes Uni3D:

```bash
conda run -n TextDeformer python scripts/run_textdeformer.py --dry-run
conda run -n TextDeformer python scripts/run_textdeformer.py --cases C02 --steps 2 --output outputs/smoke-textdeformer
conda run -n TextDeformer python scripts/run_textdeformer.py --output outputs/textdeformer
```

`--repo` y `--python` permiten usar otro checkout original o intérprete.
Las salidas conservan el formato oficial y no incluyen los registros específicos
de UniTextDeformer.

Después de completar TextDeformer, ejecutar los Experimentos 1, 2 y 3 desde el
entorno `unitextdeformer`. Esperar a que termine cada comando antes del siguiente.
Cada uno completa 20 casos y guarda sus salidas en carpetas independientes:

```bash
python run.py --variants uni3d-vc --output outputs/experiment
python run.py --variants uni3d-no-vc --output outputs/experiment
python run.py --variants uni3d-no-vc-no-delta --output outputs/experiment
```

Si falta VRAM, reducir cámaras/resolución sirve para un experimento propio,
pero cambia el protocolo. Consultar [README.md](README.md) para seleccionar casos.

En cada sesión nueva, activar el entorno y restaurar las variables del paso 1.
