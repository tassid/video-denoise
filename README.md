# Video Denoise

App local para Windows que remove ruído visual de vídeos. Arrasta o vídeo, escolhe um preset, clica em Start. Nada sai da máquina.

A interface é um script PowerShell (WinForms) em cima do `ffmpeg`; o motor de IA é um script Python com PyTorch rodando na GPU.

## Requisitos

- Windows 10/11
- [ffmpeg](https://ffmpeg.org/) no PATH: `winget install Gyan.FFmpeg`
- Para o método **AI**: placa NVIDIA (testado em RTX 5060 Ti, 8 GB), Python 3.13 ou [uv](https://docs.astral.sh/uv/), ~5 GB de disco

## Instalação

```powershell
git clone https://github.com/tassid/video-denoise.git
cd video-denoise
powershell -ExecutionPolicy Bypass -File .\setup.ps1   # só para o método AI (~3 GB)
```

Abrir com duplo clique em `Video Denoise.bat`.

Sem rodar o `setup.ps1` o app funciona, apenas sem o método AI.

## Uso

1. Solte um ou mais vídeos na janela (ou clique na área para procurar).
2. Escolha um **Preset** ou ajuste os controles.
3. **Preview** mostra antes/depois de um frame, em 100%. `Preview at` escolhe o instante (`90`, `1:30`, `0:01:30`; vazio = meio do vídeo).
4. Marque **Test: first 10 s only** para experimentar antes de processar tudo.
5. **Start**. O resultado fica ao lado do original como `nome_denoised.mp4`. O original nunca é alterado nem sobrescrito.

### Controles

| Controle | O que faz |
|---|---|
| Method | `Fast` (hqdn3d), `Best quality` (nlmeans, lento, CPU) ou `AI` (DRUNet, GPU) |
| Strength | Quanto ruído remover. Alto demais apaga textura. |
| Sharpen | Nitidez após o denoise, só na luminância. |
| Keep detail | Devolve parte do detalhe original por cima do resultado. No AI é seletivo: quase tudo em áreas com textura, quase nada em áreas lisas. |
| Soft / zoomed source | Para vídeo recortado/ampliado, com grão grosso: processa em metade do tamanho e amplia de volta. Só AI. |
| Enhance | Reconstrói detalhe com Real-ESRGAN. **Inventa detalhe**: bom em objetos, pode alterar sutilmente rostos reais. Só AI, desligado nos presets para pessoas. |
| Encoder | H.264 na CPU (compatível) ou HEVC na GPU NVIDIA (mais rápido). |

### Presets

Os presets de fábrica não podem ser alterados. **Save...** grava os ajustes atuais com um nome seu (em `presets.json`, ao lado do app); **Delete** apaga um preset seu.

## Velocidade (RTX 5060 Ti)

| Caso | Aproximado |
|---|---|
| AI, 1080p | 3 fps |
| AI + Soft source + Enhance, 1080x1920 | 6 fps |
| AI, 4K | 0,6 fps |
| Fast | perto de tempo real |

## Limites conhecidos

- Denoise não cria nitidez. Se o vídeo original já está borrado (zoom, foco), só `Enhance` melhora a aparência, e ele reconstrói, não recupera.
- Presets `Natural` e `Heavy noise` são pontos de partida, não foram calibrados em muito material.
- Saída sempre em 4:2:0 (8 ou 10 bits conforme a origem).

## Modelos

Baixados pelo `setup.ps1` dos repositórios oficiais, não redistribuídos aqui:

- DRUNet — [cszn/KAIR](https://github.com/cszn/KAIR) (MIT)
- Real-ESRGAN `realesr-general-x4v3` — [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause)

## Teste sem tela

```powershell
powershell -ExecutionPolicy Bypass -File .\VideoDenoise.ps1 -AutoRun clip.mp4 -AutoRunOpts "test,s=2,d=8"
```

Roda preview + processamento com a janela real, imprime o log e sai com código 0 em caso de sucesso. Uma captura da janela fica em `%TEMP%\VideoDenoise\ui.png`.
