
# coding: utf-8

# # RANDOMIZED SEARCH

# In[1]:


# get_ipython().magic('pylab inline')


# In[2]:


from scipy import signal
import matplotlib as np
import numpy
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
import sklearn 


# In[3]:


def normalizacao_training(features_training):
    for i in range(0,20000,1000): #normalizar uma imagem (equivalente a 1000 pontos) individualemente
        features_1000_pontos = features_training[i:i+1000,:]
        scaler = preprocessing.StandardScaler().fit(features_1000_pontos)
        features_training[i:i+1000,:] = scaler.transform(features_1000_pontos)

    return features_training


# In[4]:


def random_search():
    #carregar as melhores features e labels obtidas
    features_training = joblib.load("Features/Vasos e background/20% vasos e 80% background/features_training")
    labels_training = joblib.load("Features/Vasos e background/20% vasos e 80% background/labels_training")
    group_training = joblib.load("Features/Vasos e background/20% vasos e 80% background/group_training")
    
    tin = tick()
    print("Entrou") #debug
    param_grid = {'kernel': ('linear','rbf'), 'C': [1, 100, 500, 1000, 1500], 'gamma': [0.0001, 0.005, 0.001, 0.05, 0.01, 1]}
    svc = SVC(probability = True)
    new_clf = sklearn.model_selection.RandomizedSearchCV(estimator = svc, param_distributions = param_grid, n_iter = 30, refit = True, cv = 10)
    clf = new_clf.fit(features_training, labels_training, groups = group_training)
    tout = tick()

    print(new_clf.best_params_)
    print(new_clf.best_score_)
    print(new_clf.best_estimator_)
    print('Training time: {:.3f} s'.format(tout - tin))
    
    return


# In[5]:


# # TREINAR COM NOVO MODELO

# In[6]:


def normalizacao_individual_treino(features_training):
    for i in range(0,20000,1000): #normalizar uma imagem (equivalente a 1000 pontos) individualemente
        features_1000_pontos = features_training[i:i+1000,:]
        scaler = preprocessing.StandardScaler().fit(features_1000_pontos)
        features_training[i:i+1000,:] = scaler.transform(features_1000_pontos)
    
    return features_training


# In[7]:


def treinar_classificador():
    #carregar as features e labels, e baralhá-las para não criar um classificador tendencioso
    features_training = joblib.load('Features/Random search/features_training') #carrega as features das imagens training
    labels_training = joblib.load('Features/Random search/labels_training') #carrega as labels das imagens training
    random_state = np.random.RandomState(0) #semente
    features_training, labels_training = shuffle(features_training, labels_training, random_state = random_state) #baralhar os dados
    
    #escolher o tipo de normalização: global ou individual
    features_training = normalizacao_individual_treino(features_training)
        
    #escolher o tipo de kernel: linar ou rbf
    clf = SVC(kernel = 'rbf', C = 1500, gamma = 0.05, probability = True)
        
    #testar o modelo de acordo com os parâmetros selecionados e guardá-lo
    tin = tick()
    clf = clf.fit(features_training, labels_training)
    tout = tick()
    print('Training time: {:.3f} segundos'.format(tout - tin))
    joblib.dump(clf, filename = 'Features/Random search/classificador_hiperparametros') #guardar classificador
    
    return


# In[8]:


# # TESTAR COM NOVO MODELO

# In[9]:


def normalizacao_individual(features_test):  
    scaler = preprocessing.StandardScaler().fit(features_test)
    features_test = scaler.transform(features_test)

    return features_test


# In[10]:


def testar_classificador(clf, features_test, labels_test, features_training, labels_training):
    y_pred_train = clf.predict(features_training)
    y_pred_test = clf.predict(features_test)  
    print('Taxa de sucesso (treino): ',
          np.mean(y_pred_train == labels_training) * 100)
    print('Taxa de sucesso (teste): ',
          np.mean(y_pred_test == labels_test) * 100)
    print('Número de vectors de dados (treino/teste): {} / {}'.
          format(features_training.shape[0], features_test.shape[0]))
    print('Número de vectores de suport: ', clf.support_vectors_.shape[0])
    
    predicted_values = clf.predict_proba(features_test)
    probabilidades = predicted_values[:,1]
    
    return y_pred_test, probabilidades


# In[11]:


def reconstrucao(clf, y_pred_test, pasta, numero):
    if(numero < 10):
        numero = str(numero)
        FOV = scipy.misc.imread('DRIVE/'+ pasta + '/mask/mask0' + numero + '.png')
    else:
        numero = str(numero)
        FOV = scipy.misc.imread('DRIVE/'+ pasta + '/mask/mask' + numero + '.png')
    FOV = FOV/FOV.max()
    imagem_reconstruida = np.zeros((FOV.shape)) #imagem a ser reconstruída
    linha = 0 #y_pred_train é uma lista na horizontal com todos os pixeis, tem o tamanho = número de linhas x número de colunas da imagem
    for x in range(FOV.shape[0]):
        for y in range(FOV.shape[1]):
            if (FOV[x,y] == 1): #apenas nos interessa pontos pertencentes à FOV
                imagem_reconstruida[x,y] = y_pred_test[linha]
                linha = linha + 1 #para ir avançando ao longo da lista
    
    return imagem_reconstruida


# In[12]:


def calculo_metricas(imagem, pasta, i):
    if(i < 10):
        numero = str(i)
        mascara = scipy.misc.imread('DRIVE/'+ pasta + '/mask/mask0' + numero + '.png')
        mascara = mascara/mascara.max()
        mascara_vasos = scipy.misc.imread('DRIVE/'+ pasta + '/1st_manual/0' + numero + '_manual1.gif')
        mascara_vasos = mascara_vasos/mascara_vasos.max()
    else:
        numero = str(i)
        mascara = scipy.misc.imread('DRIVE/'+ pasta + '/mask/mask' + numero + '.png')
        mascara = mascara/mascara.max()
        mascara_vasos = scipy.misc.imread('DRIVE/'+ pasta + '/1st_manual/' + numero + '_manual1.gif')
        mascara_vasos = mascara_vasos/mascara_vasos.max()
    
    verdadeiros_positivos = 0
    verdadeiros_negativos = 0
    falsos_positivos = 0
    falsos_negativos = 0
    for x in range (imagem.shape[0]):
        for y in range (imagem.shape[1]): 
            if (mascara[x,y] == 1):
                if (imagem[x,y] == 1 and mascara_vasos[x,y] == 1): #verdadeiros positivos => imagem calculada = 1 e mascara da DRIVE = 1
                    verdadeiros_positivos = verdadeiros_positivos + 1
                elif (imagem[x,y] == 0 and mascara_vasos[x,y] == 0): #verdadeiros negativos => imagem calculada = 0 e mascara da DRIVE = 0
                    verdadeiros_negativos = verdadeiros_negativos + 1
                elif (imagem[x,y] == 1 and mascara_vasos[x,y] == 0): #falsos positivos => imagem calculada = 1 e mascara da DRIVE = 0
                    falsos_positivos = falsos_positivos + 1
                elif (imagem[x,y] == 0 and mascara_vasos[x,y] == 1): #falsos negativos => imagem calculada = 0 e mascara da DRIVE = 1
                    falsos_negativos = falsos_negativos + 1
    sensibilidade = verdadeiros_positivos / (verdadeiros_positivos + falsos_negativos) * 100
    especificidade = verdadeiros_negativos / (verdadeiros_negativos + falsos_positivos) * 100
    exatidao = (verdadeiros_positivos + verdadeiros_negativos) / (verdadeiros_positivos + verdadeiros_negativos + falsos_positivos + falsos_negativos) * 100  
    
    return sensibilidade, especificidade, exatidao


# In[13]:


def calculo_auc(probabilidades, labels_test):
    labels_test = labels_test / 255
    auc = metrics.roc_auc_score(labels_test, probabilidades) * 100
    
    return auc


# In[14]:


def calculo_roc(lista_fpr, lista_tpr):
    #guardar ROC
    [figura, ax] = plt.subplots( nrows=1, ncols=1 ) #criação da imagem e do eixo
    ax.plot(sort(lista_fpr), sort(lista_tpr), color = "green") 
    figura.suptitle('Curva ROC')
    ax.set_xlabel('Especificidade')
    ax.set_ylabel('Sensibilidade')
    plt.close(figura) #fecha a imagem e está guardada
    
    return figura


# In[15]:


def geral(vasos, background, pasta):  
    ficheiro = open('Resultados/Random search/' + pasta + '/métricas_individual_rbf_' + pasta + '.txt','w') #cria um ficheiro para guardar as métricas cálculadas
    ficheiro.write('IMAGEM' + '\t' + 'SENSIBILIDADE' + '\t' + 'ESPECIFICIDADE' + '\t' + 'EXATIDÃO' + '\t' + 'AUC \n')
    media_sens = 0
    media_espe = 0
    media_exat = 0
    media_auc = 0
    lista_fpr = []
    lista_tpr = []
    if (pasta == 'test'): #imagens 1 a 20
        inicio_contagem = 1
    elif (pasta == 'training'): #imagens 21 a 40
        inicio_contagem = 21
        
    #carregamento de dados, previamente criadas
    features_training = joblib.load('Features/Random search/features_training') #carrega as features das imagens test, uma a uma
    labels_training  = joblib.load('Features/Random search/labels_training') #carrega as labels das imagens test, uma a uma  
    classificador = joblib.load('Features/Random search/classificador_hiperparametros') #carrega o classificador
   
    #normalização das features training
    features_training = normalizacao_training(features_training)
    
    for i in range(inicio_contagem, inicio_contagem + 20):
        numero = str(i)    
        print('Imagem ' + numero)
        
        #carregamento do classificador, features e labels teste, previamente criadas
        features_test = joblib.load('Features/' + pasta + '/' + numero + '_features_' + pasta) #carrega as features das imagens test, uma a uma
        labels_test  = joblib.load('Features/' + pasta + '/' + numero + '_labels_' + pasta) #carrega as labels das imagens test, uma a uma
        
        #normalização dos dados
        features_test = normalizacao_individual(features_test)
        
        #testar classificador
        [y_pred_test, prob_linear] = testar_classificador(classificador, features_test, labels_test, features_training, labels_training)
        
        #reconstrução e save da imagem
        imagem_reconstruida = reconstrucao(classificador, y_pred_test, pasta, i)
        imagem_reconstruida = imagem_reconstruida/imagem_reconstruida.max()
        scipy.misc.toimage(imagem_reconstruida).save('Resultados/Random search/' + pasta + '/' + numero + '_imagem_reconstruida.png')
        
        #calcular e guardar as métricas
        [sensibilidade, especificidade, exatidao] = calculo_metricas(imagem_reconstruida, pasta, i)       
        auc = calculo_auc(prob_linear, labels_test)
        media_sens = media_sens + sensibilidade
        media_espe = media_espe + especificidade
        media_exat = media_exat + exatidao
        media_auc = media_auc + auc
        ficheiro.write('Imagem ' + str(numero) + '\t' + str(sensibilidade) + '\t' + str(especificidade) + '\t' + str(exatidao) + '\t' + str(auc) + '\n')
        
        #calcular a ROC
        [fpr, tpr, thresholds] = metrics.roc_curve(labels_test/255, prob_linear) #normalizar valores entre 0 e 1
        fpr = fpr.tolist() #converter fpr para uma lista
        tpr = tpr.tolist()#converter tpr para uma lista
        lista_fpr = lista_fpr + fpr #concatenar todas as imagens
        lista_tpr = lista_tpr + tpr #concatenar todas as imagens        
        print('') #parágrafo
    ficheiro.write('Médias' + '\t' + str(media_sens/20) + '\t' + str(media_espe/20) + '\t' + str(media_exat/20) + '\t' + str(media_auc/20) + '\n')  
    ROC = calculo_roc(lista_fpr, lista_tpr) #cálculo da ROC
    ROC.savefig('Resultados/Random search/' + pasta + '/ROC_individual_rbf_' + pasta + '.png')
    print("") #parágrafo
    print("") #parágrafo
    print("") #parágrafo
    
    return

if __name__ == '__main__':
    random_search()
    treinar_classificador()
    geral('20', '80', 'test')
    geral('20', '80', 'training')



