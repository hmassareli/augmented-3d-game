# DA3 Depth Backend — Protótipo

Backend Python que roda **Depth Anything V3 Metric-Large** para estimar profundidade absoluta (em metros) dos pulsos do jogador.

## O que faz

```
Webcam frame → DA3-Metric-Large → depth map em metros
                                    ↓
MediaPipe fornece (x,y) do pulso → ler depth naquele pixel → Z em metros
```

Resolve o problema central: MediaPipe não detecta Z quando o braço estica pra frente. O DA3 olha a imagem inteira e gera depth real.

## Instalação

### 1. Clonar e instalar DA3

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3.git
cd depth-anything-3
pip install -e .
pip install xformers torch>=2 torchvision
```

### 2. Instalar dependências do backend

```bash
pip install -r depth_backend/requirements.txt
```

### 3. Rodar

```bash
# Windows
run-depth-backend.cmd

# Linux/Mac
python -m uvicorn depth_backend.server:app --host 127.0.0.1 --port 8001
```

O modelo (~350M params, ~1.4GB) é baixado automaticamente do HuggingFace na primeira execução.

## Endpoints

### `GET /api/health`
Verifica se o modelo carregou.

### `POST /api/depth`
**Input:**
- `image`: frame JPEG da webcam
- `points`: JSON array de `{"x": 0..1, "y": 0..1}` (posições normalizadas dos pulsos do MediaPipe)
- `focal`: focal length da câmera em pixels (default: 800)

**Output:**
```json
{
  "depths": [0.85, 0.92],  // metros, um por ponto
  "depth_image": "data:image/jpeg;base64,...",  // visualização
  "inference_ms": 45.3,
  "depth_shape": [518, 518]
}
```

### `POST /api/depth/raw`
Retorna o depth map completo como PNG 16-bit.

## Integração com o web client

O arquivo `web/src/depth-tracker.ts` exporta:

- **`DepthTracker`** — cliente que envia frames pro backend, recebe depth, aplica Kalman filter
- **`PunchDetector`** — detecta socos pela variação de depth do pulso
- **`PunchEvent`** — `{ type: "jab"|"cross"|"left_hook"|"right_hook", power: 0-1 }`

### Uso no main.ts

```typescript
import { DepthTracker, type PunchEvent } from "./depth-tracker";

const depthTracker = new DepthTracker("http://127.0.0.1:8001", 800);

depthTracker.onPunch = (event: PunchEvent) => {
  console.log(`💥 ${event.type}! Power: ${(event.power * 100).toFixed(0)}%`);
  // Aqui você aplica dano, toca som, etc.
};

depthTracker.onDepthUpdate = (depths) => {
  // depths.left e depths.right em metros
  // Pode usar pra posicionar as luvas em Z no Three.js
};

// No loop principal, depois de obter pose do MediaPipe:
await depthTracker.processFrame(
  webcam,
  leftWristPoint,   // {x: 0..1, y: 0..1} do MediaPipe
  rightWristPoint,
  performance.now(),
  leftWristVelocity,
  rightWristVelocity,
);
```

## Performance esperada

| GPU | Resolução inference | FPS estimado |
|---|---|---|
| RTX 4090 | 518×518 | 40-60 |
| RTX 3060 | 518×518 | 25-35 |
| RTX 2060 | 518×518 | 15-20 |
| CPU only | 518×518 | 2-5 |

O cliente envia frames a 320px de largura (downscale) e o backend redimensiona pra 518×518. O throttle envia a cada 2 frames pra não saturar.

## Calibração do focal length

O focal length em pixels pode ser obtido de:
1. **EXIF da câmera** (se disponível)
2. **Calibração com padrão xadrez** (opencv)
3. **Estimativa**: pra webcam típica 640×480, focal ~600-800px

Se errar o focal, o depth relativo ainda funciona pra detecção de soco. O valor absoluto em metros estará escalado, mas a variação (delta) proporcional.

## Limitações

1. **Depth na borda da mão** — o pixel do pulso pode pegar depth do fundo. O backend amostra uma janela 5×5 e tira a mediana pra mitigar.
2. **Motion blur** — soco rápido gera blur, afeta a qualidade do depth.
3. **Latência de rede** — mesmo em localhost, há overhead de encode/decode JPEG + HTTP. Pra minimizar, o frame é downscale pra 320px.
4. **Sem consistência temporal nativa** — cada frame é independente. O Kalman filter no cliente compensa.
