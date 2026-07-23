# DA3 Depth Integration — Protótipo

Protótipo de integração do **Depth Anything V3 Metric-Large** no jogo de boxe para resolver o problema do Z absoluto do punho.

## O problema

MediaPipe Pose não detecta profundidade (Z) quando o braço estica pra frente. Em nenhum quadro. O Z do MediaPipe é relativo e simplesmente não muda pra esse movimento. Kalman não conserta dado que não existe.

## A solução

Usar **Depth Anything V3 Metric-Large** (ByteDance, nov 2025) que:
- Prevê depth métrico absoluto (em metros) por pixel
- 0.35B params, Apache 2.0
- ~20 FPS em GPU boa a 768×1024, mais rápido em resolução menor
- Vê a imagem inteira, não depende de keypoints

## Arquivos criados

```
depth_backend/
  server.py              # FastAPI server com DA3 metric depth
  requirements.txt       # Dependências Python
  test_depth_video.py    # Script de teste offline (processa vídeo)
  README.md              # Documentação do backend

web/src/
  depth-tracker.ts       # Cliente TypeScript: DepthTracker + PunchDetector + Kalman
  depth-integration-example.ts  # Exemplo de como integrar no main.ts existente

run-depth-backend.cmd    # Script Windows pra rodar o backend
```

## Como funciona

```
┌─────────────────────────────────────────────────┐
│ Browser (web/)                                   │
│                                                  │
│  Webcam → MediaPipe Pose → (x,y) pulsos (30fps) │
│       │                                          │
│       │ frame JPEG + pontos                      │
│       ▼                                          │
│  DepthTracker.processFrame()                     │
│       │                                          │
│       │ HTTP POST /api/depth                     │
└───────┼──────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────┐
│ Python Backend (depth_backend/)                  │
│                                                  │
│  DA3-Metric-Large → depth map em metros          │
│  Sample depth at (x,y) dos pulsos                │
│  Return: [left_depth_m, right_depth_m]           │
└───────┬──────────────────────────────────────────┘
        │
┌───────▼──────────────────────────────────────────┐
│ Browser (de volta)                               │
│                                                  │
│  Kalman filter suaviza depth                     │
│  PunchDetector detecta socos:                    │
│    - depth diminui rápido = braço estendendo     │
│    - depth atinge mínimo = extensão completa     │
│    - depth volta a crescer = retração            │
│    - classifica: jab / cross / hook               │
│    - calcula power (0-1)                         │
└──────────────────────────────────────────────────┘
```

## Próximos passos

1. **Testar offline** — rodar `test_depth_video.py` no vídeo de teste do repo (`WhatsApp Video 2026-07-21 at 15.32.46.mp4`) pra ver se o depth do punho muda quando o braço estende
2. **Integrar no main.ts** — seguir o exemplo em `depth-integration-example.ts`
3. **Calibrar thresholds** — ajustar `DEPTH_DELTA_THRESHOLD` e `VELOCITY_THRESHOLD` no `PunchDetector` com dados reais
4. **Otimizar** — converter DA3 pra ONNX/TensorRT pra mais FPS
