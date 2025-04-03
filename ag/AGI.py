import random
import numpy as np



# *** 1. Representação do Individuo (Cromossoma) ***
# Uma rota será representada como uma lista ordenada de cidades.
# Exemplo: [0, 1, 2, 3, 0] representa um ciclo visitando as cidades na ordem 0 -> 1 -> 2 -> 3 -> 0

def criar_individuo(num_cidades):
    """Cria um indivíduo (rota) aleatório para o PCV."""
    cidades = list(range(num_cidades))
    random.shuffle(cidades)
    return cidades + [cidades[0]] # Adiciona a cidade inicial no final para formar um ciclo

# *** 2. Função de Adaptação (Fitness Function) ***
def calcular_custo(individuo, matriz_distancias):
    """Calcula o custo total da rota percorrida pelo indivíduo."""
    custo = 0
    for i in range(len(individuo) - 1):
        cidade_atual = individuo[i]
        proxima_cidade = individuo[i+1]
        custo += matriz_distancias[cidade_atual][proxima_cidade]
    return custo

def avaliar_populacao(populacao, matriz_distancias):
    """Avalia o fitness de cada indivíduo na população."""
    fitness = []
    for individuo in populacao:
        fitness.append(calcular_custo(individuo, matriz_distancias))
    return fitness

# *** 3. Operadores Genéticos ***

# --- Seleção ---
def selecao_torneio(populacao, fitness, tamanho_torneio=3):
    """Seleciona um indivíduo usando o método do torneio."""
    melhor_individuo = None
    melhor_fitness = float('inf')
    indices_participantes = random.sample(range(len(populacao)), tamanho_torneio)
    for indice in indices_participantes:
        if fitness[indice] < melhor_fitness:
            melhor_fitness = fitness[indice]
            melhor_individuo = populacao[indice]
    return melhor_individuo

# --- Crossover ---
def crossover_OX1(pai1, pai2):
    """Realiza o crossover do tipo Order Crossover 1 (OX1)."""
    tamanho = len(pai1) - 1 # Ignora a cidade de retorno no final
    ponto1 = random.randint(0, tamanho - 1)
    ponto2 = random.randint(ponto1 + 1, tamanho)

    filho1 = [-1] * tamanho
    filho2 = [-1] * tamanho

    # Copia a seção do pai1 para o filho1
    for i in range(ponto1, ponto2):
        filho1[i] = pai1[i]

    # Preenche as posições restantes do filho1 com as cidades do pai2 na ordem em que aparecem,
    # ignorando as cidades já presentes no filho1
    indice_pai2 = ponto2 % tamanho
    indice_filho1 = ponto2 % tamanho
    while -1 in filho1:
        cidade_pai2 = pai2[indice_pai2]
        if cidade_pai2 not in filho1:
            filho1[indice_filho1] = cidade_pai2
            indice_filho1 = (indice_filho1 + 1) % tamanho
        indice_pai2 = (indice_pai2 + 1) % tamanho

    # Repete o processo trocando os pais para criar o filho2
    for i in range(ponto1, ponto2):
        filho2[i] = pai2[i]

    indice_pai1 = ponto2 % tamanho
    indice_filho2 = ponto2 % tamanho
    while -1 in filho2:
        cidade_pai1 = pai1[indice_pai1]
        if cidade_pai1 not in filho2:
            filho2[indice_filho2] = cidade_pai1
            indice_filho2 = (indice_filho2 + 1) % tamanho
        indice_pai1 = (indice_pai1 + 1) % tamanho

    return filho1 + [filho1[0]], filho2 + [filho2[0]]

# --- Mutação ---
def mutacao_troca(individuo, taxa_mutacao):
    """Realiza a mutação de troca em um indivíduo com uma dada taxa de mutação."""
    tamanho = len(individuo) - 1 # Ignora a cidade de retorno
    if random.random() < taxa_mutacao:
        i = random.randint(0, tamanho - 1)
        j = random.randint(0, tamanho - 1)
        individuo_mutado = list(individuo)
        individuo_mutado[i], individuo_mutado[j] = individuo_mutado[j], individuo_mutado[i]
        return individuo_mutado
    return individuo

# *** 4. Infecção Viral (Implementar com base nos detalhes do artigo) ***
def aplicar_infeccao_viral(populacao, fitness, taxa_infeccao, matriz_distancias):
    """Implementação do mecanismo de infecção viral."""
    nova_populacao = list(populacao)
    num_infectados = int(taxa_infeccao * len(populacao))
    indices_para_infectar = random.sample(range(len(populacao)), num_infectados)

    # *** LÓGICA DA INFECÇÃO VIRAL A SER IMPLEMENTADA COM BASE NO ARTIGO ***
    # Com base no trecho fornecido:
    # - Vírus têm infectibilidade inicial fixa.
    # - Infecção com melhora de fitness -> infectibilidade aumenta (até um limite).
    # - Infecção com piora de fitness -> infectibilidade diminui (até 0).
    # - Infectibilidade = 0 -> vírus descarta parte e copia parte do cromossoma (transdução).

    # Estrutura para representar vírus (poderá ser ajustada):
    class Virus:
        def __init__(self, segmento_rota):
            self.segmento = segmento_rota
            self.infectibilidade = 1 # Valor inicial (fixo, conforme o artigo)

    # Inicialização da população de vírus (exemplo simplificado):
    tamanho_segmento_viral = 5
    populacao_virus = [Virus(random.sample(range(matriz_distancias.shape[0]), tamanho_segmento_viral))
                       for _ in range(tamanho_populacao // 2)] # Exemplo de tamanho da população viral

    for indice in indices_para_infectar:
        individuo_infectado = list(populacao[indice])
        fitness_antes = fitness[indice]
        virus_selecionado = random.choice(populacao_virus) # Selecionar um vírus aleatoriamente

        # Tentar inserir o segmento viral no cromossoma do indivíduo (exemplo simplificado)
        ponto_insercao = random.randint(0, len(individuo_infectado) - 2)
        nova_rota = individuo_infectado[:ponto_insercao] + virus_selecionado.segmento + individuo_infectado[ponto_insercao:]
        # Remover duplicatas e garantir um ciclo válido pode ser complexo e depende da estratégia viral

        # Avaliar a nova rota (isso é um passo conceitual, a implementação real da infecção pode ser mais sutil)
        # novo_fitness = calcular_custo(nova_rota, matriz_distancias)

        # Atualizar a infectibilidade do vírus (precisa da lógica completa do artigo)
        # if novo_fitness < fitness_antes:
        #     virus_selecionado.infectibilidade = min(virus_selecionado.infectibilidade + 1, limite)
        # elif novo_fitness > fitness_antes:
        #     virus_selecionado.infectibilidade = max(virus_selecionado.infectibilidade - 1, 0)

        # Transdução (se infectibilidade == 0) - Lógica a ser implementada

        # Por enquanto, uma modificação aleatória simples para manter a estrutura:
        ponto1 = random.randint(0, len(individuo_infectado) - 2)
        ponto2 = random.randint(0, len(individuo_infectado) - 2)
        individuo_infectado[ponto1], individuo_infectado[ponto2] = individuo_infectado[ponto2], individuo_infectado[ponto1]
        nova_populacao[indice] = individuo_infectado

    return nova_populacao

# *** 5. Algoritmo Genético Principal ***
def algoritmo_genetico(matriz_distancias, num_cidades, tamanho_populacao, num_geracoes, taxa_crossover, taxa_mutacao, taxa_infeccao):
    """Executa o algoritmo genético com infecção viral para o PCV."""
    populacao = [criar_individuo(num_cidades) for _ in range(tamanho_populacao)]
    melhor_fitness_historico = []
    melhor_individuo_historico = []

    for geracao in range(num_geracoes):
        fitness = avaliar_populacao(populacao, matriz_distancias)
        melhor_indice = np.argmin(fitness)
        melhor_fitness = fitness[melhor_indice]
        melhor_individuo = populacao[melhor_indice]

        melhor_fitness_historico.append(melhor_fitness)
        melhor_individuo_historico.append(melhor_individuo)

        nova_populacao = [melhor_individuo] # Elitismo: mantém o melhor indivíduo

        while len(nova_populacao) < tamanho_populacao:
            pai1 = selecao_torneio(populacao, fitness) # Usando seleção por torneio
            pai2 = selecao_torneio(populacao, fitness)

            if random.random() < taxa_crossover:
                filho1, filho2 = crossover_OX1(pai1, pai2)
                filho1_mutado = mutacao_troca(filho1, taxa_mutacao)
                filho2_mutado = mutacao_troca(filho2, taxa_mutacao)
                nova_populacao.extend([filho1_mutado, filho2_mutado])
            else:
                filho1_mutado = mutacao_troca(list(pai1), taxa_mutacao)
                filho2_mutado = mutacao_troca(list(pai2), taxa_mutacao)
                nova_populacao.extend([filho1_mutado, filho2_mutado])

        populacao = nova_populacao[:tamanho_populacao] # Garante o tamanho da população

        populacao = aplicar_infeccao_viral(populacao, fitness, taxa_infeccao, matriz_distancias)

       

    return melhor_individuo_historico[-1], melhor_fitness_historico[-1], melhor_fitness_historico

# *** 6. Carregamento da Matriz de Distâncias (a partir da TSPLIB) ***
def carregar_distancias_tsp(nome_arquivo):
    """
    Carrega a matriz de distâncias a partir de um arquivo no formato TSPLIB.
    """
    try:
        with open(nome_arquivo, 'r') as f:
            lines = f.readlines()

        num_cidades = None
        edge_weight_type = None
        node_coords = {}
        distance_matrix = None
        reading_coords = False
        reading_edges = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("NAME"):
                pass
            elif line.startswith("TYPE"):
                problem_type = line.split(":")[1].strip()
                if problem_type != "TSP":
                    print(f"Erro: Arquivo não é do tipo TSP. Tipo encontrado: {problem_type}")
                    return None, None
            elif line.startswith("DIMENSION"):
                num_cidades = int(line.split(":")[1].strip())
                distance_matrix = np.zeros((num_cidades, num_cidades))
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip()
                if edge_weight_type not in ["GEO", "EUC_2D", "ATT", "EXPLICIT"]:
                    print(f"Aviso: Tipo de peso de aresta '{edge_weight_type}' não totalmente suportado. Assumindo cálculo euclidiano se coordenadas forem fornecidas.")
            elif line.startswith("NODE_COORD_SECTION"):
                reading_coords = True
            elif line.startswith("EDGE_WEIGHT_SECTION"):
                reading_edges = True
            elif line.startswith("DISPLAY_DATA_SECTION") or line.startswith("TOUR_SECTION") or line.startswith("EOF"):
                reading_coords = False
                reading_edges = False
            elif reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        node_id = int(parts[0]) - 1  # Ajusta para índice base 0
                        x = float(parts[1])
                        y = float(parts[2])
                        node_coords[node_id] = (x, y)
                    except ValueError:
                        print(f"Aviso: Linha de coordenadas mal formatada: '{line}'")
            elif reading_edges:
                # Para arquivos com matriz de distâncias explícita
                weights = [float(w) for w in line.split()]
                # Lógica para preencher a matriz dependerá do formato
                pass

        if node_coords and distance_matrix is not None:
            for i in range(num_cidades):
                for j in range(i + 1, num_cidades):
                    coord1 = node_coords[i]
                    coord2 = node_coords[j]
                    if edge_weight_type in ["EUC_2D", "ATT"] or edge_weight_type is None:
                        dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                        if edge_weight_type == "ATT":
                            dx = coord1[0] - coord2[0]
                            dy = coord1[1] - coord2[1]
                            dij = np.sqrt((dx**2 + dy**2) / 10.0)
                            tij = round(dij)
                            distance_matrix[i][j] = tij + 1 if tij < dij else tij
                        else:
                            distance_matrix[i][j] = round(dist)
                    elif edge_weight_type == "GEO":
                        PI = 3.141592
                        lat1 = PI * (coord1[0] // 1 + 5.0 * (coord1[0] % 1) / 3.0) / 180.0
                        lon1 = PI * (coord1[1] // 1 + 5.0 * (coord1[1] % 1) / 3.0) / 180.0
                        lat2 = PI * (coord2[0] // 1 + 5.0 * (coord2[0] % 1) / 3.0) / 180.0
                        lon2 = PI * (coord2[1] // 1 + 5.0 * (coord2[1] % 1) / 180.0)
                        RRR = 6378.388
                        q1 = np.cos(lon1 - lon2)
                        q2 = np.cos(lat1 - lat2)
                        q3 = np.cos(lat1 + lat2)
                        dij = RRR * np.arccos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
                        distance_matrix[i][j] = round(dij)

                    distance_matrix[j][i] = distance_matrix[i][j]

        elif distance_matrix is None and num_cidades is not None and reading_edges:
            # Lógica para ler a matriz de distâncias diretamente da EDGE_WEIGHT_SECTION
            index = 0
            for i in range(num_cidades):
                for j in range(i + 1, num_cidades):
                    # Adapte a forma como os pesos são lidos aqui, dependendo do formato do arquivo
                    pass # Será necessário mais detalhes sobre o formato

        elif distance_matrix is None and num_cidades is not None:
            print("Aviso: Matriz de distâncias não pôde ser completamente carregada.")

        return distance_matrix, num_cidades

    except FileNotFoundError:
        print(f"Erro: Arquivo '{nome_arquivo}' não encontrado.")
        return None, None
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo TSPLIB: {e}")
        return None, None


# *** 7. Execução Principal ***
if __name__ == "__main__":
    # *** 7. Execução Principal ***
    nome_arquivo_tsp = "data/data2.tsp"  # Substitua pelo nome do seu arquivo TSPLIB
    matriz_distancias, num_cidades = carregar_distancias_tsp(nome_arquivo_tsp)

    if matriz_distancias is not None and num_cidades is not None:
        tamanho_populacao = 50
        num_geracoes = 50
        taxa_crossover = 0.8
        taxa_mutacao = 0.05
        taxa_infeccao = 0.1  # Exemplo de taxa de infecção

        melhor_rota, melhor_custo, historico_custos = algoritmo_genetico(
            matriz_distancias,
            num_cidades,
            tamanho_populacao,
            num_geracoes,
            taxa_crossover,
            taxa_mutacao,
            taxa_infeccao
        )

        
        print("Melhor Custo Encontrado:", melhor_custo)

       
        

    else:
        print("Não foi possível carregar os dados do arquivo TSPLIB.")