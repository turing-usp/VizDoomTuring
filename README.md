<p align="center">
  <img src="banner.png" alt="banner" width="800"/>
</p>

# VizDoomTuring

Framework para treino e teste de agentes no [VizDoom](https://vizdoom.cs.put.edu.pl/), porte do DOOM voltado para Aprendizado por Reforço.

## Configurando o Projeto

Uma vez que você clonou o repositório, podemos fazer os preparativos para executá-lo! Para isto, siga os seguintes passos:

1. Crie um ambiente virtual:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. E em seguida instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Treinando o Modelo

Você pode tanto usar sua **própria Rede Neural** quanto usar uma já pronta para treinar seu agente. Cada agente pode ter seus próprios atributos customizados de acordo com o usuário deseja treinar! Para fazer essa customização, crie um arquivo .yaml como o **example_agent.yaml** e customize os atributos de Renderização, Recompensas, Policy e Treina mento!
É possível tanto treinar seu agente com outros agentes iguais a ele, quanto treinar com **multiagente**, com um .yaml customizado para cada agente. Em ambos os casos, usaremos o script **run_train.py**, executando-o pelo seguinte comando em seu navegador:

### Apenas um agente:

   ```bash
   python run_train.py --cfg [seu_agente].yaml --num-matches [X] --actors-per-match [Y]
   ```
Em que X se refere ao número de salas/partidas abertas e Y a quantidade de agentes em cada uma.

### Multiagente:

   ```bash
   python run_train.py --agent [AGENTE1].yaml:[X] --agent [AGENTE2].yaml:[Y] --num-matches Z
   ```

Em que X e Y referem-se ao número de agentes de cada tipo que são postos em Z salas diferentes.

### Parâmetros Adicionais:

Além disto, os parâmetros adicionais podem ser inseridos juntos:
| Parâmetro | Funcionalidade |
|------------|-------------|
| `--cfg` | Define o arquivo YAML do agente (Modo Single-Model). |
| `--agent` | Define `YAML:COUNT` (Modo Multi-Model). Ex: `bot.yaml:4` |
| `--num-matches` | Quantidade de janelas/partidas simultâneas. |
| `--actors-per-match` | Quantos jogadores em cada partida. |
| `--render` | Escolhe a visão: none (oculto), host (só 1) ou all (todos). |
| `--shm-obs` | Ativa Memória Compartilhada (acelera muito o treino). |
| `--map` | Nome do mapa do Doom. |
| `--wad` | Caminho para arquivo de mapa customizado. |
| `--port` | Porta inicial para a conexão do jogo. |
| `--stack` | Quantidade de frames processados por vez. |
| `--timelimit` | Tempo máximo de cada partida em minutos. |

## Treinamento via Cloud

Esse repositório também contém um script para realizar o treinamento via cloud. Para isso, abra o arquivo **TreinoColab.ipynb** no Google Colaboratory ou software de sua preferência e siga as instruções contidas nele.

## Assistindo o Agente

Se quiser apenas assistir como o seu agente se comporta, você pode usar o arquivo **Launcher.py** para fazê-lo de forma mais intuitiva. Considere:


   ```bash
   python launcher.py
   ```

