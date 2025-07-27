#Importando as bibliotecas principais
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import MinMax, OneHotEncodingNames,TargetLabelEncoder
from sklearn.pipeline import Pipeline
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Carrega o dataset com os dados de obesidade
dados = pd.read_csv('./Base de Dados/dados_obesidade.csv')

#Aplicando estilo ao componente de lista (selectbox/radio) no Streamlit
st.markdown('<style>div[role="listbox"] ul{background-color: #6e42ad}; </style>', unsafe_allow_html=True)

#Título da página
st.markdown("<h1 style='text-align: center; '> Teste de Nível de Gordura Corporal 📝 </h1> ", unsafe_allow_html = True)
st.warning('Preencha o formulário com todos os seus dados pessoais e clique no botão **ENVIAR** no final da página.')

#Listas de opções utilizadas em selects
lista_dados = ['Sim', 'Não', 'As Vezes']
lista_CAEC_CALC = ['Sim', 'Não', 'As Vezes', 'Frequentemente']
transporte = 'Carro', 'Biclicleta', 'Motocicleta', 'Transporte Público', 'Caminhada'

#Coletando os dados do usuário por meio de widgets do Streamlit
st.write('### Genêro')
input_genero = st.radio('Qual seu gênero?',['Masculino','Feminino'], index=0)
input_genero_dict = {'Feminino': 1, 'Masculino': 0}
input_genero = input_genero_dict.get(input_genero)
st.write('### Qual sua idade?')
input_idade = float(st.slider('Selecione a sua idade', 0, 100))
st.write('### Qual sua altura?')
input_altura = float(st.slider('Selecione a sua altura', 1.0, 2.10))
st.write('### Qual seu peso?')
input_peso = float(st.slider('Selecione a seu peso:', 1.0, 200.0))
st.write('### Histórico Familiar')
input_historico = st.radio('Possui histórico familiar para obesidade?',['Sim','Não'], index=0)
input_historico_dict = {'Sim': 1, 'Não': 0}
input_historico = input_historico_dict.get(input_historico)
st.write('### Alimentos Calóricos')
input_alimentos_caloricos = st.radio('Você come alimentos calóricos com frequência?',['Sim','Não'], index=0)
input_alimentos_caloricos_dict = {'Sim': 1, 'Não': 0}
input_alimentos_caloricos = input_alimentos_caloricos_dict.get(input_alimentos_caloricos)
st.write('### Alimentação Vegetais')
input_alimentacao_vegetais = st.selectbox('Você come vegetais nas suas refeições?', lista_dados)
st.write('### Quantas refeições principais você faz diariamente?')
input_alimentacao_diaria = float(st.slider('Selecione a sua idade', 0, 3))
st.write('### Alimentação entre Refeições')
input_alimentacao = st.selectbox('Você come alguma coisa entre as refeições?', lista_CAEC_CALC)
st.write('### Uso de Cigarro')
input_fumar = st.radio(' Você fuma?',['Sim','Não'], index=0)
input_fumar_dict = {'Sim': 1, 'Não': 0}
input_fumar = input_fumar_dict.get(input_fumar)
st.write('### Água')
input_agua = float(st.slider('Quanta de água você bebe diariamente?', 0,3))
st.write('### Calorias')
input_calorias = st.radio(' Você monitora as calorias que ingere diariamente?',['Sim','Não'], index=0)
input_calorias_dict = {'Sim': 1, 'Não': 0}
input_calorias = input_calorias_dict.get(input_calorias)
st.write('### Com que frequência você pratica atividade física?')
input_atividadefisica = float(st.slider('Selecione a quantidade de horas semanais: ', 0, 42))
st.write('### Quanto tempo você usa dispositivos tecnológicos como celular, videogame, televisão, computador e outros?')
tempo_tela = float(st.slider('Selecione a quantidade de horas semanais:', 0, 100))
st.write('### Alcool')
input_alcool = st.selectbox('Com que frequência você bebe álcool?', lista_CAEC_CALC)
st.write('### Transporte')
input_transporte = st.selectbox('Qual meio de transporte você costuma usar?', transporte)

#Mapendo as respostas de string para números/categorias esperadas
if input_alimentacao_vegetais == 'Sim':
    alimentacao_vegetais = 1
elif input_alimentacao_vegetais == 'Não':
   alimentacao_vegetais = 2
else:
    alimentacao_vegetais = 3

if input_alimentacao == 'Sim':
    alimentacao = 'Always'
elif input_alimentacao == 'Não':
    alimentacao = 'no'
elif input_alimentacao == 'As Vezes':
    alimentacao = 'Sometimes'
else:
    alimentacao = 'Frequently'
        
if input_alcool == 'Sim':
    alcool = 'Always'
elif input_alcool == 'Não':
    alcool = 'no'
elif input_alcool == 'As Vezes':
    alcool = 'Sometimes'
else:
    alcool = 'Frequently'

if input_transporte == 'Carro':
    transporte = 'Automobile'
elif input_transporte == 'Biclicleta':
    transporte = 'Bike'
elif input_transporte == 'Motocicleta':
    transporte = 'Motorbike'
elif input_transporte == 'Transporte Público':
    transporte = 'Public_Transportation'
else:
    transporte = 'Walking'
    
#Criando dicionário com os dados do usuário no formato do dataset original
dados_cliente = {
    'Gender': int(input_genero),
    'Age': float(input_idade),
    'Height': float(input_altura),
    'Weight': float(input_peso),
    'family_history': int(input_historico),
    'FAVC': int(input_alimentos_caloricos),
    'FCVC': float(alimentacao_vegetais),
    'NCP': float(input_alimentacao_diaria),
    'CAEC': str(alimentacao),
    'SMOKE': int(input_fumar),
    'CH2O': float(input_agua),
    'SCC': int(input_calorias),
    'FAF': float(input_atividadefisica),
    'TUE': float(tempo_tela),
    'CALC': str(alcool),
    'MTRANS': str(transporte),
    'Obesity': 0  # valor fictício para manter compatibilidade
}

#Função para dividir dados em treino e teste com seed fixa para reprodutibilidade
def data_split(df, test_size):
    SEED = 1561651
    treino_df, teste_df = train_test_split(df, test_size=test_size, random_state=SEED)
    return treino_df.reset_index(drop=True), teste_df.reset_index(drop=True)

#Aplicando a divisão com 20% para teste
treino_df, teste_df = data_split(dados, 0.2)

#Criando um dataframe com o novo cliente
cliente_predict_df = pd.DataFrame([dados_cliente])

#Concatenando novo cliente ao dataframe dos dados de teste
teste_novo_cliente  = pd.concat([teste_df,cliente_predict_df],ignore_index=True)

#Função responsável por aplicar o pipeline de transformação nos dados
#Pré-processa colunas categóricas e numéricas com codificadores personalizados
def pipeline_teste(df):
    #Criando um pipeline para as features com OneHotEncoding e MinMaxScaler
    feature_pipeline = Pipeline([
        ('OneHotEncoding', OneHotEncodingNames()),
        ('min_max_scaler', MinMax())
    ])
    #Combina o pipeline de features com o transformador de target
    full_pipeline = ColumnTransformer([
        ('features', feature_pipeline, df.columns.drop('Obesity')),
        ('target', TargetLabelEncoder(), ['Obesity'])
    ])
    #Ajustando o pipeline aos dados
    full_pipeline.fit(df)

    #Aplicando as transformações nos dados de entrada
    features = full_pipeline.named_transformers_['features'].transform(df.drop(columns='Obesity'))
    target = full_pipeline.named_transformers_['target'].transform(df[['Obesity']])

    #Garantindo que a saída das features seja um DataFrame com nomes de colunas
    if not isinstance(features, pd.DataFrame):
        try:
            col_names = full_pipeline.named_transformers_['features'].named_steps['OneHotEncoding'].encoder.get_feature_names_out(
                full_pipeline.named_transformers_['features'].named_steps['OneHotEncoding'].OneHotEncoding
            ).tolist()
        except:
            col_names = [f'feature_{i}' for i in range(features.shape[1])]

        col_names += [col for col in full_pipeline.named_transformers_['features'].named_steps['min_max_scaler'].features_to_scale
                      if col not in col_names]
        col_names = list(dict.fromkeys(col_names))  # remove duplicatas mantendo ordem
        features = pd.DataFrame(features, columns=col_names, index=df.index)

     #Adicionando nomes das variáveis numéricas
    target = pd.DataFrame(target, columns=['Obesity'], index=df.index)
    treino = pd.concat([features, target], axis=1)

    return treino

#Ajustando tipos das colunas
for col in ['CAEC', 'CALC', 'MTRANS', 'Obesity']:
    teste_novo_cliente[col] = teste_novo_cliente[col].astype(str)

for col in ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
    teste_novo_cliente[col] = teste_novo_cliente[col].astype(float)

for col in ['Gender','family_history','SCC', 'FAVC', 'SMOKE',]:
    teste_novo_cliente[col] = teste_novo_cliente[col].astype(int)

#Aplicando o pipeline de transformação no novo cliente
X_cliente_transformado = pipeline_teste(teste_novo_cliente)

#Quando o botão "Enviar" for pressionado
if st.button('Enviar'):
    #Carregando o modelo treinado e as features esperadas
    model, feature_names = joblib.load('./Modelo/modelo.joblib')
    #Garantindo que todas as colunas esperadas pelo modelo estejam presentes
    for col in feature_names:
        if col not in X_cliente_transformado.columns:
            X_cliente_transformado[col] = 0

    #Reordenando colunas conforme modelo treinado        
    X_cliente_transformado = X_cliente_transformado[feature_names]
    predicao = model.predict(X_cliente_transformado)

    #st.write(X_cliente_transformado)

    #Realizando a predição com o modelo carregado
    predicao = model.predict(X_cliente_transformado)

    #Dicionário com a descrição textual dos rótulos previstos
    rotulos = {
        0: 'Abaixo do Peso Ideal',
        1: 'Peso Ideal',
        2: 'Obesidade Tipo I',
        3: 'Obesidade Tipo II',
        4: 'Obesidade Tipo III',
        5: 'Sobrepeso Nível I',
        6: 'Sobrepeso Nível II'
    }

    #Obtendo o rótulo final e sua descrição textual
    classe_obesidade = int(predicao[-1])
    descricao = rotulos.get(classe_obesidade, "Desconhecido")

    #Apresentando o resultado principal ao usuário
    st.success(f'### Seu nível de gordura corporal é de: **{descricao}**')

   #Fornece feedback adicional personalizado com base na classe prevista
    if classe_obesidade == 0:
        st.info('📉 Você está abaixo do peso. Uma avaliação nutricional pode ajudar.')
    elif classe_obesidade == 1:
        st.balloons()
        st.success('✅ Você está com peso normal. Continue cuidando da sua saúde!')
    elif classe_obesidade in [2, 3, 4]:
        st.warning('⚠️ Este resultado indica obesidade. Procure um(a) profissional de saúde.')
    elif classe_obesidade in [5, 6]:
        st.info('🟡 Você está com sobrepeso. Buscar orientação pode ser útil.')

