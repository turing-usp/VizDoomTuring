# Comandos

Use os comandos abaixo dentro de:

```powershell
cd "C:\Users\guilh\OneDrive\Desktop\turing\vizdoom_teste - Copia\VizDoomTuring"
```

## Treino principal com render

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --actors-per-match 10 --render host --shm-obs --trainer-port 7200 --port 5200 --progress-bar --frame-skip 6 --ticrate 70 --host-start-delay 1.5 --actor-start-delay 0.02 --match "multi_duel_dm15_smart.wad|map01|5" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP02|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP11|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP16|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP27|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP28|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP29|1"
```

## Treino principal sem render

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --actors-per-match 10 --render none --shm-obs --trainer-port 7200 --port 5200 --progress-bar --frame-skip 6 --ticrate 70 --host-start-delay 1.5 --actor-start-delay 0.02 --match "multi_duel_dm15_smart.wad|map01|5" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP02|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP11|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP16|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP27|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP28|1" --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP29|1"
```

## Smart map simples

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --num-matches 8 --actors-per-match 12 --render none --shm-obs --trainer-port 7200 --port 5200 --scenario multi_duel_dm15_smart.wad --map map01 --frame-skip 9 --ticrate 105 --host-start-delay 1.5 --actor-start-delay 0.02
```

## Teste rapido com render

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --num-matches 2 --actors-per-match 12 --render host --shm-obs --trainer-port 7200 --port 5200 --scenario multi_duel_dm15_smart.wad --map map01 --frame-skip 6 --ticrate 70 --host-start-delay 1.5 --actor-start-delay 0.02
```

## Assistir modelo

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --actors-per-match 10 --render all --shm-obs --trainer-port 7200 --port 5200 --progress-bar --frame-skip 1 --ticrate 35 --host-start-delay 1.5 --actor-start-delay 0.02 --match "..\tmp_freedm\freedm-0.13.0\freedm-0.13.0\freedm.wad|MAP11|1" --play
```

## Opcoes uteis

- `--warmstart-reset-steps`: reinicia timesteps/schedules mantendo os pesos carregados.
- `--render none`: melhor para treino pesado.
- `--render host`: mostra uma janela para acompanhar o host.
- `--frame-skip 1`: visualizacao mais suave, mas bem mais pesada.
- `--scenario multi_duel_dm15_smart.wad`: troca o cenario base.
- `--wad ...`: adiciona conteudo extra; nao use para trocar o mapa principal.
