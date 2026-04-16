# Comandos

Use os comandos abaixo dentro de:

```powershell
cd "C:\Users\guilh\OneDrive\Desktop\turing\vizdoom_teste - Copia\VizDoomTuring"
```

eu uso esse:
python .\run_train.py --cfg .\stern_deathmatch.yaml --num-matches 12 --actors-per-match 12 --render host --trainer-port 7200 --port 5200 --scenario multi_duel_dm15_smart.wad --map map01 --frame-skip 9 --ticrate 105 --host-start-delay 1.5 --actor-start-delay 0.02

para reiniciar timesteps
--warmstart-reset-steps
## Smart map 8x12 treino

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --num-matches 8 --actors-per-match 12 --render none --shm-obs --trainer-port 7200 --port 5200 --scenario multi_duel_dm15_smart.wad --map map01 --host-start-delay 1.5 --actor-start-delay 0.02
```

## Smart map 8x15 treino

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --num-matches 8 --actors-per-match 15 --render none --shm-obs --trainer-port 7200 --port 5200 --scenario multi_duel_dm15_smart.wad --map map01 --host-start-delay 1.5 --actor-start-delay 0.02
```

## Smart map visualizacao suave

Use este para conferir o mapa e ver o jogo sem tanto pulo:

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --num-matches 10 --actors-per-match 15 --render host --trainer-port 7200 --port 5200 --scenario multi_duel_dm15_smart.wad --map map01 --frame-skip 9 --ticrate 105 --host-start-delay 1.5 --actor-start-delay 0.02
```

## Smart map visualizacao intermediaria

```powershell
python .\run_train.py --cfg .\stern_deathmatch.yaml --num-matches 1 --actors-per-match 15 --render host --trainer-port 7200 --port 5200 --scenario multi_duel_dm15_smart.wad --map map01 --frame-skip 2 --ticrate 35 --host-start-delay 1.5 --actor-start-delay 0.02
```

## Dicas

- `--scenario multi_duel_dm15_smart.wad` troca o cenario base de verdade.
- `--wad ...` agora fica para conteudo extra, nao para trocar o mapa principal.
- `--frame-skip 1` deixa a visualizacao mais suave, mas custa desempenho.
- `--render none` e o modo certo para treino pesado.
