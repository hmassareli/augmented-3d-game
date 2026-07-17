# Plano: jogo de boxe 3D controlado por camera

## 1. O que vamos construir

Um jogo de navegador em terceira pessoa no qual a webcam observa a postura do jogador e controla duas luvas 3D. O adversario luta como um boxeador: procura distancia, protege a cabeca, circula, esquiva e responde aos golpes. Teclado e mouse continuam disponiveis como acessibilidade e como ferramenta de teste.

O alvo inicial e um PC com webcam, Chrome ou Edge recente e boa iluminacao frontal. Celular pode vir depois; nao deve ser o primeiro alvo, pois camera, desempenho e enquadramento variam demais.

### Resultado desejado

```text
Webcam
  |
  v
Detector de pose ------> filtro de ruido ------> intencoes do jogador
  |                                                   |
  |                                                   +--> andar / girar / defender / golpear
  v
Pontos do corpo                                      |
  |                                                   v
  +----------------------------------------------> jogo 3D (Three.js)
                                                       |
                                                       +--> luvas, personagem, ringue, oponente, audio
```

## 2. Termos basicos

| Termo                   | Em linguagem simples                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `landmark`              | Um ponto que o modelo encontra na imagem, por exemplo pulso, ombro ou quadril.                                                 |
| modelo de pose          | Uma rede neural ja treinada para localizar esses pontos no video; ela roda no aparelho do jogador.                             |
| rig                     | O esqueleto invisivel dentro de um modelo 3D, usado para mover bracos, pernas e tronco.                                        |
| IK (cinematica inversa) | Tecnica que dobra os ossos intermediarios para que uma mao chegue a um ponto pedido.                                           |
| maquina de estados      | Uma forma simples e previsivel de definir o que o oponente esta fazendo: guardar, aproximar, atacar, esquivar ou se recuperar. |
| dead zone               | Pequena faixa que ignora tremores involuntarios para evitar que o personagem ande sozinho.                                     |
| interpolacao            | Transicao gradual entre duas posicoes, em vez de um salto visual.                                                              |

## 3. Decisao tecnica recomendada

### Camera e pose: MediaPipe Pose Landmarker

Usar o **MediaPipe Pose Landmarker**, uma biblioteca do Google que detecta 33 pontos corporais em video no navegador. Ele entrega ombros, cotovelos, pulsos, quadris, joelhos e tornozelos, suficientes para locomocao, guarda e golpes.

- Rodar no cliente: o video e analisado localmente e nao precisa ser enviado a servidor.
- Processar cada quadro de video que chegar, com limite adaptativo de aproximadamente 24 a 30 analises por segundo.
- Usar `requestVideoFrameCallback` quando disponivel para sincronizar a analise ao video.
- Comecar com o modelo leve para menor atraso; permitir modo de qualidade maior em computadores rapidos.
- Mostrar um indicador de enquadramento antes da luta: tronco, punhos e quadris precisam estar visiveis.
  linacao do tronco. A primeira versao deve usar MediaPipe; so trocar se a medicao real mo
  Alternativa para comparar: **MoveNet** do TensorFlow.js. Ele e rapido, mas tem menos pontos e geralmente oferece menos margem para inferir a incstrar atraso inaceitavel.

### Graficos: Three.js + Rapier

- **Three.js** desenha o ringue, personagens, luz, sombra e particulas no navegador usando a GPU.
- **Rapier** calcula colisao: luva contra rosto/corpo, limite do ringue e empurrao leve. A fisica nao deve decidir a animacao inteira, apenas os contatos importantes.
- Modelos e texturas em `glTF/GLB`, formato proprio para carregar cenas 3D na web.
- Audio espacial simples: sino, passo, impacto, respiracao e torcida discreta.

### Personagem do jogador: modelo animado + luvas guiadas pela camera

Nao vale a pena tentar reproduzir com exatidao todas as articulacoes filmadas. Uma camera unica nao sabe a profundidade de forma confiavel e um braco pode esconder o outro. A abordagem de qualidade e hibrida:

```text
Antes: webcam tenta mandar diretamente em cada osso
       -> bracos tremem, ocorrem giros estranhos e o golpe nao parece forte

Proposto: webcam identifica a intencao
          -> animacao mantem postura, pernas e equilibrio
          -> IK leva as luvas para a direcao real das maos
          -> colisao e efeitos confirmam o golpe
```

- Pernas, tronco basal e deslocamentos usam animacoes profissionais misturadas entre si.
- Cada luva e um objeto 3D visivel, levemente a frente do personagem, preso ao alvo filtrado do pulso correspondente.
- O antebraco e o braco acompanham a luva por IK quando isso for visualmente seguro.
- Quando a camera perder um pulso, manter a ultima posicao por pouco tempo e retornar para a guarda; nunca congelar uma luva no ar indefinidamente.
- Um avatar completo pode ser desligado no modo de primeira pessoa, deixando somente luvas e sombra para leitura mais limpa.

## 4. Controles corporais

### Calibracao obrigatoria, curta e pessoal

Antes do round, o jogador fica em guarda por dois segundos e olha para a tela. Salvar apenas valores temporarios:

- centro horizontal e vertical do peito;
- largura entre os ombros, usada como regua proporcional ao proprio corpo;
- distancia normal entre pulso e ombro;
- inclinacao neutra do tronco;
- lado dominante, configuravel manualmente.

Nenhuma imagem precisa ser salva. Os dados de calibracao podem existir apenas na memoria do navegador e ser apagados ao encerrar a partida.

### Movimento pelo tronco

Em vez de reagir a todo pixel, medir o vetor entre o centro dos quadris e o centro dos ombros. Normalizar pela largura dos ombros: pessoas perto e longe da camera recebem o mesmo controle relativo.

| Gesto sustentado por cerca de 180 ms | Resultado                     |
| ------------------------------------ | ----------------------------- |
| tronco para frente                   | andar para frente             |
| tronco para tras                     | recuar                        |
| tronco para a esquerda               | circular para a esquerda      |
| tronco para a direita                | circular para a direita       |
| voltar ao centro                     | desacelerar e manter a guarda |

Implementacao recomendada:

1. Aplicar filtro exponencial nos pontos; ele suaviza o tremor dando mais peso ao movimento recente sem atrasar demais.
2. Usar uma dead zone de aproximadamente 8% da largura do ombro.
3. Exigir que a inclinacao atravesse a dead zone por 180 ms.
4. Aumentar a velocidade gradualmente em 250 a 400 ms e limitar a velocidade maxima.
5. Converter esquerda/direita da imagem para a orientacao do personagem, nao para o norte fixo do ringue.

Isso atende as duas ideias do pedido: o personagem pode andar continuamente enquanto houver inclinacao, e so comeca depois de um pequeno tempo de confirmacao. Na configuracao, oferecer dois perfis: `Sustentado` (mais preciso) e `Direto` (mais responsivo, porem sujeito a falsos movimentos).

### Socos, guarda e esquiva

Um soco nao deve ser somente “a mao esta longe”. Usar uma pontuacao que combina extensao do braco e velocidade do pulso:

```text
pulso sai da guarda rapidamente
        +
cotovelo abre na direcao do alvo
        +
mao fica a frente do ombro por alguns quadros
        = golpe candidato
```

- `jab`: mao da frente avanca quase em linha reta.
- `cross`: mao traseira avanca e o ombro/tronco gira para o lado oposto.
- `hook`: pulso descreve arco lateral com cotovelo alto.
- guarda: ambos os pulsos proximos das bochechas por uma janela curta.
- esquiva lateral: cabeca e ombros se movem lateralmente, sem transformar inclinacao de caminhada em esquiva. A primeira versao pode reservar esquiva para uma tecla; depois de medir conflitos, habilitar por corpo.
- O dano depende da velocidade filtrada, extensao e distancia correta. Golpes ao acaso cansam o jogador e tem pouco efeito.

Usar uma pequena previsao de 40 a 70 ms para a luva compensar a latencia da camera. Ela deve ser desligada quando o rastreamento estiver pouco confiante para nao criar luvas “fantasma”.

## 5. Oponente que parece boxear

Oponente bom nao significa oponente que le a pose do jogador sem limite. Ele precisa cometer erros, respeitar recuperacao e ter sinais visuais de intencao.

```text
                 distancia / angulo / stamina / perigo recebido
                                      |
                                      v
  procurar alcance -> circular -> guardar -> atacar -> recuperar
           ^                            |                    |
           +------------- esquivar <----+--------------------+
```

Regras iniciais:

- Manter uma distancia de combate, aproximar quando longe e sair quando muito perto.
- Circular para nao ficar parado no centro da linha de golpe.
- Defender alto quando detecta um golpe vindo ao rosto e defender baixo em resposta a golpes no corpo.
- Usar ataques com preparacao e recuperacao: `jab`, `cross`, `jab-cross`, `hook` e golpe ao corpo.
- Escolher combinacoes por contexto, nao por sorte pura. Por exemplo, depois que o jogador recua repetidamente, usar jab para reduzir a distancia; depois de um soco perdido, responder com contra-ataque moderado.
- Ter stamina, tempo de reacao, agressividade e habilidade como parametros por dificuldade.
- Dar ao jogador uma janela real para esquiva e bloqueio. A dificuldade sobe pela decisao e ritmo, nao por dano invisivel ou deteccao perfeita.

Para uma qualidade maior, fazer o oponente usar **utilidade**, uma pontuacao para cada acao. Exemplo: atacar ganha pontos se esta no alcance e o jogador esta recuperando; perde pontos se a propria stamina estiver baixa ou se uma luva vier rapido em sua direcao. Isso e mais natural que uma sequencia fixa de animacoes e mais simples de equilibrar que treinar uma IA por reforco.

## 6. Colisao e sensacao de impacto

- Cada luva recebe uma esfera de colisao; cabeca, mandibula, torax e figado do oponente recebem capsulas ou esferas menores.
- Aceitar um impacto somente quando a luva estiver no estado de golpe, viajando em direcao ao alvo e dentro da distancia. Isso impede dano com a mao parada.
- Ao acertar: pausar a imagem muito brevemente (hit stop), vibrar a camera com moderacao, tocar audio por material, emitir particulas pequenas e iniciar animacao de reacao.
- Impor intervalo minimo entre impactos do mesmo braco para impedir multiplos acertos no mesmo atravessamento.
- Proteger contra camera ruim: se a confianca dos pontos cair, pausar dano corporal e mostrar uma mensagem curta de reenquadramento em vez de inventar movimentos.

## 7. Assets 3D e audio

Nao baixar qualquer arquivo de resultado de busca. Antes de integrar, registrar URL, autor, licenca, data de download e se a redistribuicao e permitida. A licenca mostrada na pagina especifica do asset vence qualquer descricao generica abaixo.

| Necessidade                          | Fonte para avaliar                                   | Uso proposto                                                                | Observacao de licenca                                                                                                                                          |
| ------------------------------------ | ---------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| boxeador com rig e animacoes         | [Adobe Mixamo](https://www.mixamo.com/)              | personagem base e movimentos genericos como idle, passos, esquivas e quedas | A pagina confirma personagens rigados e animacoes; revisar os termos Adobe antes de publicar, especialmente se os arquivos forem redistribuidos.               |
| personagem humano de qualidade maior | [Sketchfab](https://sketchfab.com/features/download) | buscar `boxer rigged`, `boxing gloves rigged` e `fighter game ready`        | Filtrar por `Downloadable` e pela licenca do modelo individual, idealmente CC0 ou CC-BY com credito. Nao assumir que todo download e livre para uso comercial. |
| ringue, arquibancada e props         | [Poly Haven](https://polyhaven.com/)                 | HDRI, materiais e texturas de alta qualidade                                | Conteudo CC0; bom para materiais e iluminacao, nao para personagem humano pronto.                                                                              |
| efeitos sonoros                      | [Freesound](https://freesound.org/)                  | sino, impacto, passos e torcida                                             | Selecionar CC0 ou cumprir exatamente a atribuicao solicitada pelo autor.                                                                                       |
| alternativa estilizada gratuita      | [Quaternius](https://quaternius.com/)                | prototipo com personagens e cenarios leves                                  | Conferir a licenca exibida no pacote escolhido antes de incluir no repositorio.                                                                                |

Para a versao de alta qualidade, a melhor combinacao e: um boxeador licenciado individualmente no Sketchfab ou loja equivalente, retopologia se necessaria, rig compativel com Mixamo, luvas separadas modeladas ou ajustadas no Blender e materiais PBR (texturas que simulam como couro e tecido reagem a luz). Mixamo e otimo para a base, mas nao deve ser a unica fonte de movimentos de boxe especificos.

### Lista objetiva de buscas

1. `boxer male rigged game ready PBR` e `female boxer rigged game ready PBR` no Sketchfab, filtrando licenca antes de baixar.
2. `boxing ring game ready` e `gym interior game ready` para o cenario.
3. `boxing idle`, `boxing footwork`, `boxing dodge`, `punch combo`, `hit reaction` e `knockdown` no Mixamo.
4. Texturas de couro, borracha e madeira no Poly Haven para padronizar as luvas, saco de pancada e ringue.

## 8. Arquitetura de codigo proposta

```text
src/
  app/                 inicializacao, telas e configuracoes
  camera/              permissao, video e pose MediaPipe
  input/               calibracao, filtros e interpretacao de gestos
  game/
    player/            estado do jogador, luvas, IK e stamina
    opponent/          decisao, navegacao e ataques do adversario
    combat/            hitboxes, dano, rounds e arbitragem
    world/             ringue, camera, luz e audio
  assets/              somente manifestos e arquivos com licenca registrada
  ui/                  HUD, calibracao, pausa e acessibilidade
```

Cada modulo deve expor dados simples. Exemplo: `input` nao move objetos 3D; ele publica `moveVector`, `punch`, `guard` e `trackingConfidence`. Assim e possivel testar o jogo com teclado, video gravado ou webcam sem duplicar a logica de combate.

## 9. Fases de implementacao e criterio de pronto

### Fase 0 - Fundacao (1 a 2 dias)

- Criar projeto Vite com TypeScript, Three.js, MediaPipe e Rapier.
- Cena com ringue simples, camera, luz, medidor de FPS e controles teclado/mouse.
- Criterio: manter 60 FPS em computador alvo sem camera; carregar assets uma unica vez.

### Fase 1 - Webcam e calibracao (2 a 3 dias)

- Pedir permissao de camera apenas apos clique do usuario.
- Exibir video espelhado, esqueleto de depuracao e indicador de confianca.
- Criar captura de postura neutra e botoes de recalibrar/espelhar camera.
- Criterio: obter ombros, quadris e pulsos continuamente por 60 segundos em iluminacao normal.

### Fase 2 - Movimento corporal (2 a 4 dias)

- Implementar filtro, dead zone, confirmacao temporal e aceleracao.
- Gravar telemetria local: inclinacao, direcao escolhida, latencia e transicoes.
- Criterio: 20 tentativas de andar/parar por direcao com no maximo 1 falso acionamento por direcao; atraso percebido abaixo de 150 ms quando o hardware permitir.

### Fase 3 - Luvas e golpes (3 a 5 dias)

- Luvas flutuantes vinculadas aos pulsos, com limites de alcance e retorno a guarda.
- Deteccao de jab e cross; depois hook.
- Alvo de treino com impactos, pontuacao e repeticao em camera lenta.
- Criterio: pelo menos 85% dos jabs intencionais reconhecidos em uma sessao de teste e menos de 10% de golpes falsos quando o jogador esta parado em guarda.

### Fase 4 - Combate e oponente (4 a 7 dias)

- Adversario com locomocao, guarda, distancia, combinacoes, esquiva e stamina.
- Hitboxes, dano, rounds, nocaute tecnico e fim de luta.
- Criterio: tres dificuldades distinguiveis; oponente nao atravessa cordas nem ataca sem recuperacao visual.

### Fase 5 - Arte, polimento e acessibilidade (3 a 7 dias)

- Integrar personagem e cenario licenciados, animacoes de maior qualidade, audio e efeitos.
- Opcoes para sensibilidade, lado dominante, camera espelhada, controles tradicionais e modo sentado.
- Criterio: usuario entende como calibrar sem tutorial longo; o jogo continua utilizavel quando a camera falha, usando controles alternativos.

## 10. Riscos e como reduzir

| Risco                                            | Consequencia                                  | Mitigacao                                                                                                                    |
| ------------------------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| pouca luz, fundo confuso ou corpo fora do quadro | pontos errados e controles irritantes         | tela de qualidade de rastreamento, mensagem de reposicionamento e modo teclado.                                              |
| latencia da webcam/modelo                        | socos parecem atrasados                       | modelo leve, resolucao moderada, filtro curto, previsao limitada e medicao em milissegundos.                                 |
| uma camera nao mede profundidade com precisao    | socos para frente podem parecer curtos/longos | usar velocidade, extensao relativa e estados de golpe, nao distancia 3D bruta.                                               |
| jogador se cansa ou se machuca                   | sessao ruim e risco fisico                    | rounds curtos, pausa, area livre, aquecimento opcional e nenhum movimento que exija salto/agachamento profundo inicialmente. |
| asset sem permissao clara                        | bloqueio de publicacao                        | manifesto de licenca por asset e nenhum arquivo importado sem origem registrada.                                             |
| personagem “fantoche”                            | baixa qualidade apesar de boa camera          | animacao base forte, IK limitado e luvas independentes como prioridade visual.                                               |

## 11. Como medir qualidade de verdade

Nao avaliar apenas por “parece legal”. Registrar sessoes com consentimento e medir:

- FPS de renderizacao, tempo de inferencia de pose e atraso total camera-para-luva.
- taxa de acerto de gesto intencional e taxa de falso positivo em guarda/parado.
- perda de rastreamento por minuto.
- distancia media de combate, quantidade de ataques bloqueados/esquivados e vitorias por dificuldade.
- nota de conforto e fadiga depois de 3 rounds curtos.

Comparar sempre o mesmo cenario: mesma iluminacao, mesma webcam, mesma distancia e mesma sequencia de movimentos. Assim uma melhoria no filtro ou no modelo pode ser atribuida a algo real, e nao a uma mudanca de ambiente.

## 12. Primeiro corte jogavel

O primeiro corte nao tentara ter todos os golpes. Ele deve conter:

1. Webcam, permissao, enquadramento e calibracao.
2. Ringue simples e duas luvas flutuantes seguindo os pulsos.
3. Inclinar tronco para andar nas quatro direcoes, com teclado de reserva.
4. Jab e cross contra saco de pancada ou manequim com hitbox.
5. Um oponente de treino que apenas guarda, recebe dano e se move lentamente.

Depois que esse corte for fluido e medido, adicionar oponente completo, assets finais e modos de luta. Esse caminho diminui o maior risco do projeto: descobrir tarde que o controle por camera nao esta estavel o suficiente para o combate.

## 13. IA de boxe: self-play supervisionado

### Decisao: IA escolhe a acao; o motor executa o boxe

A IA nao controla ossos, posicoes exatas das luvas nem velocidade livremente. Ela escolhe uma acao de uma biblioteca de animacoes e o motor de combate aplica suas regras. Isso torna o treinamento muito mais rapido e impede que o modelo descubra movimentos impossiveis.

```text
estado da luta -> politica treinada -> acao desejada
             |
             v
           supervisor de combate
             |
      acao valida ----------+---------- acao invalida
          |                              |
          v                              v
     animacao / simulacao                 guarda ou recuperacao
```

O **supervisor** e codigo deterministico: com a mesma situacao, ele sempre toma a mesma decisao. Ele nao tenta vencer pelo modelo; apenas protege as regras fisicas e taticas que seriam caras para a IA redescobrir.

### Observacao e acoes

O modelo recebe numeros relativos ao proprio corpo, nunca coordenadas absolutas do mapa. Assim ele pode enfrentar jogadores em qualquer ponto do ringue.

| Grupo | Dados para a IA |
| --- | --- |
| posicao | distancia, angulo para o oponente, borda mais proxima e velocidade relativa |
| combate | guarda, ataque atual, recuperacao restante, ultimo golpe e acerto/erro recente |
| recursos | vida e stamina dos dois lutadores |
| variedade | ultimas seis acoes da propria IA e quantas vezes cada uma foi repetida |

As acoes iniciais serao: `guardar`, avancar, recuar, circular esquerda/direita, pivotar esquerda/direita, jab, cross, hook esquerdo/direito, bloquear alto, bloquear corpo e esquivar esquerda/direita. Uma acao dura uma janela curta e tem inicio, execucao e recuperacao; a IA nao pode trocar de ataque a cada quadro.

### Limites humanos e stamina

Stamina e simples de implementar e necessaria. Ela deve limitar decisao, velocidade e repeticao, em vez de ser apenas uma barra decorativa.

| Situacao | Regra inicial |
| --- | --- |
| capacidade maxima | 100 pontos de stamina |
| guarda/parado | regenera 12 pontos por segundo |
| movimentacao | custa 3 pontos por segundo |
| jab/cross/hook | custam 5 / 7 / 9 pontos ao iniciar |
| stamina abaixo de 25 | movimentos ficam 15% mais lentos; golpes recebem penalidade moderada de forca |
| stamina abaixo de 10 | supervisor bloqueia ataques pesados; permite guarda, recuo e jab lento |
| recuperacao de golpe | nenhum novo ataque antes de terminar sua janela de recuperacao |

Os valores serao parametros de configuracao, nao constantes espalhadas pelo codigo. Eles sao o primeiro palpite mensuravel e serao ajustados depois de testes de luta. O motor tambem limita aceleracao, velocidade de giro e velocidade maxima de cada deslocamento para que nenhuma politica possa virar ou atacar mais rapido que a animacao correspondente.

### Regras do supervisor

- Girar gradualmente para manter o oponente a frente; nunca teletransportar a rotacao.
- Impedir ataques fora de alcance e trocar a acao por guarda quando nao houver alternativa segura.
- Respeitar custo de stamina, recuperacao, colisao com cordas e limite do ringue.
- Aplicar intervalo minimo entre ataques do mesmo braco e entre impactos aceitos.
- Limitar repeticao: apos tres ataques iguais em uma janela curta, reduzir a prioridade da mesma acao; apos cinco, bloquear temporariamente salvo se houver justificativa defensiva explicita no estado.
- Registrar cada acao rejeitada e o motivo. Isso permite descobrir se a politica esta ruim ou se as regras estao restritivas demais.

### Treino sem overfitting

**Overfitting** e quando a IA decora um adversario ou ambiente especifico e vai mal em situacoes novas. Para reduzir isso, o treino nao sera contra uma unica versao de si mesma.

1. Criar uma fila de oponentes: politica atual, versoes antigas e bots de regras com estilos agressivo, defensivo e contra-atacador.
2. Sortear parametros a cada luta: tamanho do ringue dentro de uma faixa pequena, stamina inicial, velocidade de recuperacao, distancia inicial e estilo do adversario.
3. Avaliar uma versao candidata contra uma colecao fixa que ela nunca enfrenta enquanto aprende.
4. Promover o novo modelo somente se melhorar a taxa de vitoria e mantiver indicadores de variedade, stamina e legalidade de acao.
5. Gravar semente aleatoria, configuracao, recompensa por componente e versao do oponente em cada checkpoint.

### Recompensa de treino

O modelo recebe pontos por luta. A recompensa precisa ensinar boxe eficaz e tambem uma luta legivel:

| Evento | Pontos de recompensa |
| --- | --- |
| golpe limpo no alcance correto | positivo, proporcional ao dano |
| bloqueio ou esquiva de golpe real | positivo moderado |
| manter distancia e angulo seguros | positivo pequeno e continuo |
| vencer round/luta | positivo grande |
| golpe no ar, fora de alcance ou sem stamina | negativo |
| encostar na corda, girar excessivamente ou acao rejeitada | negativo |
| repetir ataque sem efeito | negativo crescente |
| ficar passivo por tempo demais | negativo pequeno |

A recompensa de vitoria nunca deve ser a unica metrica. Tambem vamos medir taxa de acoes invalidadas, stamina media, distribuicao de golpes, repeticao maxima de uma acao e tempo gasto em distancia de combate.

### Ordem de execucao

1. Criar um nucleo de combate deterministico e testavel: estado, acoes, stamina, recuperacao, ringue, dano e supervisor.
2. Criar o simulador Python sem renderizacao, que roda milhares de lutas usando exatamente esse contrato de acoes.
3. Construir o jogo web 3D com controles de teclado e o mesmo contrato de combate; camera e pose entram depois do loop de luta estar divertido.
4. Implementar bots de regras para validar que a luta faz sentido antes de treinar uma rede neural.
5. Rodar self-play supervisionado com uma biblioteca consolidada, inicialmente Stable-Baselines3 com PPO; PPO e um algoritmo que melhora a politica gradualmente sem permitir mudancas grandes e instaveis de uma vez.
6. Exportar o modelo aprovado para ONNX, um formato portavel de rede neural, e executar sua inferencia no navegador com ONNX Runtime Web.
7. Comparar a IA exportada contra bots e contra registros anonimizados de acoes humanas, sem treinar diretamente durante uma partida do usuario.

O primeiro marco executavel e uma luta de texto entre dois bots: nenhuma stamina negativa, nenhuma acao fora da recuperacao e nenhuma saida do ringue. So depois dessa validacao faz sentido gastar tempo em modelo neural e assets 3D finais.

## 14. Prioridade atual: avatar e movimento pela webcam

A IA fica em espera. O proximo objetivo e provar o loop que o jogador realmente sente: carregar o Kojiro, ligar a webcam com permissao explicita e reproduzir os movimentos minimos no avatar.

1. Carregar o FBX `Non-Constraint Rig` do Kojiro e validar materiais, escala, sombra e orientacao no ringue.
2. Usar MediaPipe Pose Landmarker localmente no navegador para ler ombros, pulsos e quadris.
3. Mostrar video espelhado e indicador de rastreamento; camera so inicia depois de clique do usuario.
4. Mapear cada pulso relativo ao ombro para uma luva 3D, suavizada e limitada ao alcance humano.
5. Usar o centro/inclinacao de ombros e quadris para postura e locomocao; apenas depois aplicar IK aos bracos do rig.
6. Incorporar as animacoes Mixamo como base de guarda e passos, misturando-as com as luvas rastreadas.

O primeiro teste de aceitacao e simples: com webcam ativa, levantar e mover uma mao deve mover a luva correspondente de forma estavel, sem a outra luva acompanhar por engano. Esse teste vem antes de luta, IA ou multiplayer.
