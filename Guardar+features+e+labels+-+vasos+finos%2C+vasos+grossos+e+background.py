
# coding: utf-8

# # TRABALHO FINAL
# ### Retinal Blood Vessel Segmentation Using Line Operators and Support Vector Classification

# # SVM

# In[1]:


# get_ipython().magic('pylab inline')


# In[2]:


from scipy import signal
import matplotlib
import numpy as np
import mahotas
import scipy.ndimage
import skimage.morphology as sm
import scipy
from scipy.signal import convolve2d
from sklearn import metrics
from sklearn.utils import random
import joblib
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.svm import SVC
from time import time as tick


# # Leitura das imagens

# In[3]:


def leitura_imagens_svm(pasta):
    lista_imagens_S = []
    lista_imagens_S0 = []
    lista_imagens_I = []
    lista_mascaras = []
    lista_mascaras_vasos = []
    lista_final = []
    if (pasta == 'test'): #imagens 1 a 20
        inicio_contagem = 1
    elif (pasta == 'training'): #imagens 21 a 40
        inicio_contagem = 21
    else:
        print('ERRO! Nenhuma pasta encontra')
        
    for i in range(inicio_contagem, inicio_contagem + 20):
        if (i < 10):
            numero = str(i)
            imagem_S = scipy.misc.imread('Imagens segmentadas/'+ pasta + '/imagens_S/0' + numero + '_imagem_S.png')
            lista_imagens_S.append(imagem_S) #adicionar imagens S, uma a uma
            imagem_S0 = scipy.misc.imread('Imagens segmentadas/'+ pasta + '/imagens_S0/0' + numero + '_imagem_S0.png')
            lista_imagens_S0.append(imagem_S0) #adicionar imagens S0, uma a uma
            imagem_I = scipy.misc.imread('Imagens segmentadas/'+ pasta + '/imagens_invertidas/0' + numero + '_imagem_invertida.png')
            lista_imagens_I.append(imagem_I) #adicionar imagens invertidas, uma a uma
            mascara = scipy.misc.imread('DRIVE/'+ pasta + '/mask/mask0' + numero + '.png')
            lista_mascaras.append(mascara) #adicionar máscaras, uma a uma
            mascara_vasos = scipy.misc.imread('DRIVE/'+ pasta + '/1st_manual/0' + numero + '_manual1.gif')     
            lista_mascaras_vasos.append(mascara_vasos) #adicionar máscaras dos vasos, uma a uma
        elif (i > 9):
            numero = str(i)
            imagem_S = scipy.misc.imread('Imagens segmentadas/'+ pasta + '/imagens_S/' + numero + '_imagem_S.png')
            lista_imagens_S.append(imagem_S) #adicionar imagens S, uma a uma
            imagem_S0 = scipy.misc.imread('Imagens segmentadas/'+ pasta + '/imagens_S0/' + numero + '_imagem_S0.png')
            lista_imagens_S0.append(imagem_S0) #adicionar imagens S0, uma a uma
            imagem_I = scipy.misc.imread('Imagens segmentadas/'+ pasta + '/imagens_invertidas/' + numero + '_imagem_invertida.png')
            lista_imagens_I.append(imagem_I) #adicionar imagens invertidas, uma a uma
            mascara = scipy.misc.imread('DRIVE/'+ pasta + '/mask/mask' + numero + '.png')
            lista_mascaras.append(mascara) #adicionar máscaras, uma a uma
            mascara_vasos = scipy.misc.imread('DRIVE/'+ pasta + '/1st_manual/' + numero + '_manual1.gif')     
            lista_mascaras_vasos.append(mascara_vasos) #adicionar máscaras dos vasos, uma a uma
    lista_final.append(lista_imagens_S) #adicionar todas as imagens S
    lista_final.append(lista_imagens_S0) #adicionar todas as imagens S0
    lista_final.append(lista_imagens_I) #adicionar todas as imagens invertidas
    lista_final.append(lista_mascaras) #adicionar todas as máscaras
    lista_final.append(lista_mascaras_vasos) #adicionar todas as máscaras dos vasos
    
    return lista_final


# # GUARDAR FEATURES

# # Guardar features de 1000 pontos de vasos finos, vasos grossos e background

# In[4]:


percentagem_vasos_finos = ['5', '5', '10', '10', '10', '15', '25', '40']
percentagem_vasos_grossos = ['5', '10', '5', '10', '15', '10', '25', '40']
percentagem_background = ['90', '85', '85', '80', '75', '75', '50', '20']


# In[5]:


def definir_vasos(imagem):
    kernel = sm.square(4) #elemento estruturante
    imagem_vasos_grossos = sm.opening(imagem, kernel) #erosão seguido de dilatação, apenas fica os vasos grossos
    imagem_vasos_finos = imagem - imagem_vasos_grossos #vasos finos
    
    return imagem_vasos_finos, imagem_vasos_grossos


# In[6]:


def guardar_features_training(percentagem_vasos_finos, percentagem_vasos_grossos, percentagem_background):      
    for i in range(len(percentagem_vasos_finos)):
        vasos_finos = percentagem_vasos_finos[i]
        vasos_grossos = percentagem_vasos_grossos[i]
        background = percentagem_background[i]
        vasos_finos_int = int(vasos_finos)
        vasos_grossos_int = int(vasos_grossos)
        background_int = int(background)
        lista_imagens = leitura_imagens_svm('training') #conjuntos de imagens necessárias
        random_state = np.random.RandomState(0) #0 é a semente
        features = np.zeros((20000,3)) #linhas = 1000 pontos x 20 imagens; colunas = feature de S, S0 e I
        labels = np.zeros(20000) #linhas = 1000 pontos x 20 imagens; colunas = label da máscara vasos
        linha = 0

        for i in range(len(lista_imagens[0])): #percorrer as 20 imagens selecionadas
            imagem_S = lista_imagens[0][i]
            imagem_S0 = lista_imagens[1][i]
            imagem_I = lista_imagens[2][i]
            imagem_mascara = lista_imagens[3][i]
            imagem_mascara_vasos = lista_imagens[4][i]
            [imagem_vasos_finos, imagem_vasos_grossos] = definir_vasos(imagem_mascara_vasos)
            contagem_pontos = 0
            lista_pontos = []
            while (contagem_pontos < vasos_finos_int * 10): #até atingir a cota introduzida pelo operadorando
                x = random.sample_without_replacement(n_population = imagem_S.shape[0], n_samples = 1, random_state = random_state) #seleção de um ponto x aleatório
                y = random.sample_without_replacement(n_population = imagem_S.shape[1], n_samples = 1, random_state = random_state) #seleção de um ponto y aleatório
                coordenadas = (x,y)
                checkpoint = coordenadas in lista_pontos
                if (checkpoint == False and imagem_mascara[x,y] == 255 and imagem_mascara_vasos[x,y] == 255 and imagem_vasos_finos[x,y] == 255): #confirmação de que o pixel ainda não foi selecionado, pertence à FOV, é um vaso e é um vaso fino
                    numero_linha = linha + contagem_pontos #0 a 19999 linhas possíveis
                    features[numero_linha, 0] = imagem_S[x,y]
                    features[numero_linha, 1] = imagem_S0[x,y]
                    features[numero_linha, 2] = imagem_I[x,y]
                    labels[numero_linha] = imagem_mascara_vasos[x,y]
                    lista_pontos.append(coordenadas)
                    contagem_pontos = contagem_pontos + 1 #adição de um ponto válido e não repetido)
                    
            contagem_pontos = 0
            lista_pontos = []
            while (contagem_pontos < vasos_grossos_int * 10): #até atingir a cota introduzida pelo operadorando
                x = random.sample_without_replacement(n_population = imagem_S.shape[0], n_samples = 1, random_state = random_state) #seleção de um ponto x aleatório
                y = random.sample_without_replacement(n_population = imagem_S.shape[1], n_samples = 1, random_state = random_state) #seleção de um ponto y aleatório
                coordenadas = (x,y)
                checkpoint = coordenadas in lista_pontos
                if (checkpoint == False and imagem_mascara[x,y] == 255 and imagem_mascara_vasos[x,y] == 255 and imagem_vasos_grossos[x,y] == 255): #confirmação de que o pixel ainda não foi selecionado, pertence à FOV, é um vaso e é um vaso grosso                  
                    numero_linha = linha + contagem_pontos + vasos_finos_int * 10 #0 a 19999 linhas possíveis
                    features[numero_linha, 0] = imagem_S[x,y]
                    features[numero_linha, 1] = imagem_S0[x,y]
                    features[numero_linha, 2] = imagem_I[x,y]
                    labels[numero_linha] = imagem_mascara_vasos[x,y]
                    lista_pontos.append(coordenadas)
                    contagem_pontos = contagem_pontos + 1 #adição de um ponto válido e não repetido)
                    
            contagem_pontos = 0
            lista_pontos = []
            while (contagem_pontos < background_int * 10): #até atingir a cota introduzida pelo operadorando
                x = random.sample_without_replacement(n_population = imagem_S.shape[0], n_samples = 1, random_state = random_state) #seleção de um ponto x aleatório
                y = random.sample_without_replacement(n_population = imagem_S.shape[1], n_samples = 1, random_state = random_state) #seleção de um ponto y aleatório
                coordenadas = (x,y)
                checkpoint = coordenadas in lista_pontos
                if (checkpoint == False and imagem_mascara[x,y] == 255 and imagem_mascara_vasos[x,y] == 0): #confirmação de que o pixel ainda não foi selecionado, pertence à FOV e é background                    
                    numero_linha = linha + contagem_pontos + vasos_finos_int * 10 + vasos_grossos_int * 10 #0 a 19999 linhas possíveis
                    features[numero_linha, 0] = imagem_S[x,y]
                    features[numero_linha, 1] = imagem_S0[x,y]
                    features[numero_linha, 2] = imagem_I[x,y]
                    labels[numero_linha] = imagem_mascara_vasos[x,y]
                    lista_pontos.append(coordenadas)
                    contagem_pontos = contagem_pontos + 1 #adição de um ponto válido e não repetido)
                    
            linha = linha + 1000 #para passar para a próxima imagem que contém os próximos 1000 pontos 

            joblib.dump(features, filename = 'Features/Vasos finos, vasos grossos e background/' + vasos_finos + '% vasos finos, ' + vasos_grossos + '% vasos grossos e ' + background + '% background/features_training') #guardar as features das imagens training
            joblib.dump(labels, filename = 'Features/Vasos finos, vasos grossos e background/' + vasos_finos + '% vasos finos, ' + vasos_grossos + '% vasos grossos e ' + background + '% background/labels_training') #guardar as features das imagens training
        #print('Features e labels training guardadas!') #debug
    
    return


# # Guardar features test e training individualmente

# In[8]:


def guardar_features(pasta):
    lista_imagens = leitura_imagens_svm(pasta) #conjuntos de imagens necessárias
    linha = 0
    
    for i in range(len(lista_imagens[0])): #percorrer as 20 imagens selecionadas
        imagem_mascara = lista_imagens[3][i]
        imagem_mascara = imagem_mascara / imagem_mascara.max()
        pontos_FOV = int(np.sum(imagem_mascara)) #número de pontos pertencentes à FOV
        features = np.zeros((pontos_FOV,3)) #linhas = número de pontos pertencentes à FOV; colunas = feature de S, S0 e I
        labels = np.zeros(pontos_FOV) #linhas = número de pontos pertencentes à FOV; colunas = label da máscara vasos
        imagem_S = lista_imagens[0][i]
        imagem_S0 = lista_imagens[1][i]
        imagem_I = lista_imagens[2][i]
        imagem_mascara_vasos = lista_imagens[4][i]
        contagem_pontos = 0
        for x in range(imagem_S.shape[0]): #adicionar todos os pontos da imagem
            for y in range(imagem_S.shape[1]):
                if (imagem_mascara[x,y] == 1): #confirmação de que o pixel pertence à FOV
                    features[contagem_pontos, 0] = imagem_S[x,y]
                    features[contagem_pontos, 1] = imagem_S0[x,y]
                    features[contagem_pontos, 2] = imagem_I[x,y]
                    labels[contagem_pontos] = imagem_mascara_vasos[x,y]
                    contagem_pontos = contagem_pontos + 1 #adição de um novo ponto
        if (pasta == 'test'):
            numero = i + 1
            numero = str(numero) #conversão para string
            joblib.dump(features, filename = 'Features/test/' + numero + '_features_test') #guardar as features das imagens training
            joblib.dump(labels, filename = 'Features/test/' + numero + '_labels_test') #guardar as features das imagens
        elif (pasta == 'training'):
            numero = i + 21
            numero = str(numero) #conversão para string
            joblib.dump(features, filename = 'Features/training/' + numero + '_features_training') #guardar as features das imagens training
            joblib.dump(labels, filename = 'Features/training/' + numero + '_labels_training') #guardar as features das imagens training
    #print('Features e labels test guardadas!') #debug
    
    return


# # TREINAR CLASSIFICADOR

# # Normalização dos dados treino

# In[11]:


def normalizacao_global(features_training):
    scaler = preprocessing.StandardScaler().fit(features_training)
    features_training = scaler.transform(features_training)
    
    return scaler, features_training


# In[12]:


def normalizacao_individual(features_training):
    for i in range(0,20000,1000):
        features_1000_pontos = features_training[i:i+1000,:]
        scaler = preprocessing.StandardScaler().fit(features_1000_pontos)
        features_training[i:i+1000,:] = scaler.transform(features_1000_pontos)
    
    return features_training


# # Treinar classificador com: vasos/background, global/individual e linear/rbf

# In[13]:


def treinar_classificador(percentagem_vasos_finos, percentagem_vasos_grossos, percentagem_background, normalizacao, kernel):
    for i in range(len(percentagem_vasos_finos)):
        vasos_finos = percentagem_vasos_finos[i]
        vasos_grossos = percentagem_vasos_grossos[i]
        background = percentagem_background[i] 
        
        #carregar as features e labels, e baralhá-las para não criar um classificador tendencioso
        features_training = joblib.load('Features/Vasos finos, vasos grossos e background/' + vasos_finos + '% vasos finos, ' + vasos_grossos + '% vasos grossos e ' + background + '% background/features_training') #carrega as features das imagens training
        labels_training = joblib.load('Features/Vasos finos, vasos grossos e background/' + vasos_finos + '% vasos finos, ' + vasos_grossos + '% vasos grossos e ' + background + '% background/labels_training') #carrega as labels das imagens training
        random_state = np.random.RandomState(0) #semente
        features_training, labels_training = shuffle(features_training, labels_training, random_state = random_state) #baralhar os dados

        #escolher o tipo de normalização: global ou individual
        if (normalizacao == 'global'):
            scaler, features_training = normalizacao_global(features_training)
            joblib.dump(scaler, filename = 'Features/Vasos finos, vasos grossos e background/' + vasos_finos + '% vasos finos, ' + vasos_grossos + '% vasos grossos e ' + background + '% background/scaler_' + kernel) #guardar classificador
        elif (normalizacao == 'individual'):
            features_training = normalizacao_individual(features_training)

        #escolher o tipo de kernel: linar ou rbf
        if (kernel == 'linear'):
            clf = SVC(kernel = 'linear', probability = True)
        elif (kernel == 'rbf'):
            clf = SVC(kernel = 'rbf', probability = True)

        #testar o modelo de acordo com os parâmetros selecionados e guardá-lo
        tin = tick()
        clf = clf.fit(features_training, labels_training)
        tout = tick()
        print('Training time: {:.3f} segundos'.format(tout - tin))
        joblib.dump(clf, filename = 'Features/Vasos finos, vasos grossos e background/' + vasos_finos + '% vasos finos, ' + vasos_grossos + '% vasos grossos e ' + background + '% background/classificador_' + normalizacao + '_' + kernel) #guardar classificador

    return


if __name__ == '__main__':
    guardar_features_training(percentagem_vasos_finos, percentagem_vasos_grossos, percentagem_background)
    guardar_features('test')
    guardar_features('training')
    treinar_classificador(percentagem_vasos_finos, percentagem_vasos_grossos, percentagem_background, 'global', 'linear')
    treinar_classificador(percentagem_vasos_finos, percentagem_vasos_grossos, percentagem_background, 'global', 'rbf')
    treinar_classificador(percentagem_vasos_finos, percentagem_vasos_grossos, percentagem_background, 'individual', 'linear')
    treinar_classificador(percentagem_vasos_finos, percentagem_vasos_grossos, percentagem_background, 'individual', 'rbf')

