# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:50:05 2023

@author: nbouaddo
"""
import pickle

import os
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import numpy as np
# chemin = "Z:/Statistiques/Stagiaires/Stage 2023 Yuka performance/5 PROGRAMMES/ANALYSES/MODELES_LOG/"
chemin = "C:/Users/BOUADDOUCH Najia/Documents/STAGE CHANEL/CLAIMPREDICT/"

labels = ['ANTI-AGEING', 'BRIGHTENING-ILLUMINATING', 'CLEANSING', 'FIRMING',
       'MATTIFYING', 'MOISTURISING-HYDRATING', 'PLUMPING',
       'REDUCESDARKCIRCLES-PUFFINESS', 'REDUCESFINELINES-WRINKLES',
       'REDUCESREDNESS', 'REDUCESTHEAPPEARANCEOFPORES', 'TONING',
       'UVPROTECTION', 'WHITENING', 'ANTIOXIDANT']
       
def load_all_classifiers(labels):
  
  classifiers = {}
  for label in labels:
    
      # Load the model
    model_filename = f"{chemin}binary_models/{label}_model.pkl"  # Replace "label" with the actual label name
    with open(model_filename, "rb") as file:
    
       
      classifiers[label] = pickle.load(file)
  return classifiers
 

def custom_tokenizer(text):
  
  tokens = text.split(', ')
  
  return tokens

  
def load_vectorizer(directory):
  vectorizer_file = os.path.join(directory, "vectorizer.pkl")
  with open(vectorizer_file, "rb") as file:
    vectorizer = pickle.load(file)
  return vectorizer  


#predire imsur un nouveau jeu de données : 

def predict_all_classifiers(classifiers, X_test):
    predictions = {}
    for label, classifier in classifiers.items():
        predictions[label] = classifier.predict(X_test)
    return predictions
  

################  predire pour une liste d'ingrédients

# l'importance des variables pour un claim donné ,que les 50 premiers ingrédients les plus importants: 

def get_global_weights_df(vectorizer, classifiers, label):
  feature_importance = classifiers[label].coef_[0]
  words = vectorizer.get_feature_names_out() 
  #words = np.concatenate((words, ['SKINCARE']), axis=0) 
  df_words_weights = pd.DataFrame({'words': words, 'weights': feature_importance})
  df_words_weights = df_words_weights.sort_values(ascending=False, by='weights')
  return df_words_weights.iloc[0:50]

def is_vectorized_sentence_empty(vectorized_sentence):
    return vectorized_sentence.nnz == 0
def get_prediction(sentence, vectorizer, classifiers, labels):
    x_test = vectorizer.transform(pd.Series(sentence))
    #x_test=np.append(x_test,cat)
    preds = np.zeros((x_test.shape[0], len(labels)))
    if not is_vectorized_sentence_empty(x_test):
        for idx, label in enumerate(labels):
            preds[:, idx] = classifiers[label].predict_proba(x_test)[:, 1]
    return preds, x_test

def get_local_weights_df(vectorizer, test_term_doc, classifiers, label):
  words = np.array(vectorizer.get_feature_names_out())[test_term_doc.indices]
  weights = classifiers[label].coef_.ravel()[test_term_doc.indices]
  df_words_weights = pd.DataFrame({'words': words, 'weights': weights}).sort_values(ascending=False, by='weights')
  return df_words_weights

#dict de mes classifieurs
classifiers = load_all_classifiers(labels)
vectorizer = load_vectorizer(chemin + 'binary_models')

#exemple : pour le claim toning : 
label = 'TONING'
variable_importance = get_global_weights_df(vectorizer, classifiers, label)



#Avant de prédire une nouvelle liste, il faut passer les ingrédients dans la fonction R qui permet de trouver les fuzzy duplicates : new_list_ing_prepro.R
#
#==> Afin d'assurer que l'ingrédients est écrit de la même façon que notre base vectorizer


#https://www.gnpd.com/sinatra/recordpage/10824094/from_search/2QSb5kMJtI/?page=1
new_list2 = "AQUA, ETHYLHEXYL METHOXYCINNAMATE, BUTYLENE GLYCOL, GLYCERIN, SD ALCOHOL 1, CYCLOPENTASILOXANE, C12-15 ALKYL BENZOATE, NIACINAMIDE, DIPHENYLSILOXY PHENYL TRIMETHICONE, CYCLOHEXASILOXANE, TITANIUM DIOXIDE, BIS-ETHYLHEXYLOXYPHENOL METHOXYPHENYL TRIAZINE, 1,2-HEXANEDIOL, DIETHYLAMINO HYDROXYBENZOYL HEXYL BENZOATE, C14-22 ALCOHOLS, GLYCERYL STEARATE, PEG-100 STEARATE, SILICA, DIMETHICONE, CETEARYL ALCOHOL, POLYACRYLATE-13, C12-20 ALKYL GLUCOSIDE, AMMONIUM ACRYLOYLDIMETHYLTAURATE/VP COPOLYMER, POLYSILICONE-11, POLYISOBUTENE, PARFUM, CAPRYLYL GLYCOL, CI 77492, DISODIUM EDTA, ETHYLHEXYLGLYCERIN, ALUMINUM HYDROXIDE, ADENOSINE, CARBOMER, TRIETHOXYCAPRYLYLSILANE, POLYSORBATE 20, CI 77491, SORBITAN ISOSTEARATE, BHT, GOSSYPIUM HERBACEUM EXTRACT, TOCOPHEROL"
#new_list =  "AQUA, GLYCERIN, PROPANEDIOL, LACTOBACILLUS FERMENT LYSATE FILTRATE, PAEONIA SUFFRUTICOSA BRANCH/FLOWER/LEAF EXTRACT, BUTYLENE GLYCOL, PENTAERYTHRITYL TETRAETHYLHEXANOATE, HYDROXYETHYLPIPERAZINE ETHANE SULFONIC ACID, NIACINAMIDE, COCO-CAPRYLATE/CAPRATE, C10-18 TRIGLYCERIDES, DICAPRYLYL ETHER, SQUALANE, BENTONITE, 1,2-HEXANEDIOL, AMMONIUM ACRYLOYLDIMETHYLTAURATE/VP COPOLYMER, CETEARYL ALCOHOL, PROPYLENE GLYCOL DICAPRYLATE/DICAPRATE, SIMMONDSIA CHINENSIS SEED OIL, HYDROXYACETOPHENONE, CETYL PALMITATE, SODIUM METHYL STEAROYL TAURATE, SORBITAN PALMITATE, BATYL ALCOHOL, HYDROXYPROPYL TETRAHYDROPYRANTRIOL, SODIUM HYALURONATE, SUCROSE PALMITATE, CARBOMER, ARGININE, GLYCERYL ACRYLATE/ACRYLIC ACID COPOLYMER, SOLUBLE COLLAGEN, PALMITOYL TRIPEPTIDE-1, PALMITOYL TETRAPEPTIDE-7, ACETYL DIPEPTIDE-1 CETYL ESTER, HYDROXYPINACOLONE RETINOATE, DECARBOXY CARNOSINE HCL, EPIGALLOCATECHIN GALLATYL GLUCOSIDE, PAEONIA SUFFRUTICOSA SEED OIL, PAEONIA SUFFRUTICOSA ROOT EXTRACT, PAEONIA SUFFRUTICOSA EXTRACT, ASTAXANTHIN, BIFIDA FERMENT FILTRATE, UBIQUINONE, TOCOPHEROL, CENTELLA ASIATICA ROOT EXTRACT, PROPOLIS EXTRACT, JASMINUM SAMBAC FLOWER EXTRACT, POLIANTHES TUBEROSA EXTRACT, GLYCYRRHIZA INFLATA ROOT EXTRACT, MAGNOLIA OFFICINALIS BARK EXTRACT, LEPTOSPERMUM SCOPARIUM LEAF EXTRACT, HABERLEA RHODOPENSIS LEAF EXTRACT, CITRUS RETICULATA PEEL EXTRACT, ROSMARINUS OFFICINALIS LEAF EXTRACT, HELIANTHUS ANNUUS SEED FLOUR, ROSA DAMASCENA FLOWER EXTRACT, SODIUM LACTATE, COCO-GLUCOSIDE, HEXANEDIOL, PENTYLENE GLYCOL, PELARGONIUM GRAVEOLENS OIL, DIISOPROPYL ADIPATE, GLYCERYL COCOATE, PROPYLENE CARBONATE, HYDROXYETHYLCELLULOSE, CANANGA ODORATA FLOWER OIL, CITRUS UNSHIU PEEL OIL, ANTHEMIS NOBILIS FLOWER OIL, ROSMARINUS OFFICINALIS LEAF OIL, POGOSTEMON CABLIN OIL, LITSEA CUBEBA FRUIT OIL, ACRYLATES/C10-30 ALKYL ACRYLATE CROSSPOLYMER, SORBITAN LAURATE, STEARALKONIUM HECTORITE, SORBITAN OLEATE, XANTHAN GUM, STEARYL GLYCYRRHETINATE"
#new_list = "AQUA, C9-12 ALKANE, UNDECANE, GLYCERIN, CETEARYL ISONONANOATE, ISONONYL ISONONANOATE, PENTYLENE GLYCOL, TRIDECANE, SQUALANE, ETHYLENE/PROPYLENE/STYRENE COPOLYMER, BUTYLENE GLYCOL, PARFUM, DIPSACUS SYLVESTRIS EXTRACT, PHENOXYETHANOL, PROPANEDIOL, SILYBUM MARIANUM SEED OIL, CAPRYLIC/CAPRIC TRIGLYCERIDE, TROMETHAMINE, TOCOPHERYL ACETATE, ESCIN, CARBOMER, CHENOPODIUM QUINOA SEED EXTRACT, ETHYLHEXYLGLYCERIN, AVENA SATIVA KERNEL EXTRACT, XANTHAN GUM, BUTYLENE/ETHYLENE/STYRENE COPOLYMER, LEONTOPODIUM ALPINUM FLOWER/LEAF EXTRACT, PERSEA GRATISSIMA OIL UNSAPONIFIABLES, CARAMEL, CURCUMA LONGA (TURMERIC) ROOT EXTRACT, THEOBROMA CACAO EXTRACT, DISODIUM EDTA, MUSA SAPIENTUM FRUIT EXTRACT, ACTINIDIA CHINENSIS FRUIT EXTRACT, KALANCHOE PINNATA LEAF EXTRACT, MYROTHAMNUS FLABELLIFOLIA LEAF/STEM EXTRACT, CITRIC ACID, LYCIUM BARBARUM FRUIT EXTRACT, SODIUM BENZOATE, MALTODEXTRIN, ORTHOSIPHON STAMINEUS EXTRACT, HEDYCHIUM CORONARIUM ROOT EXTRACT, SALICORNIA HERBACEA EXTRACT, MANGIFERA INDICA LEAF EXTRACT, JANIA RUBENS EXTRACT, PENTAERYTHRITYL TETRA-DI-T-BUTYL HYDROXYHYDROCINNAMATE, ASCORBIC ACID, SODIUM CITRATE, TOCOPHEROL, POTASSIUM SORBATE, ENGELHARDTIA CHRYSOLEPIS LEAF EXTRACT, CI 14700, CALLICARPA JAPONICA FRUIT EXTRACT"
#new_list = 'ZINC OXIDE, AQUA, GLYCERIN'
#new_list2 = ['AQUA', 'GLYCERIN']
#new_list2 = ['ZINC OXIDE']
#new_list2 = ', '.join(new_list2)


list2= [new_list]
vectorized_sentence = vectorizer.transform([new_list])
predicted_probabilities=get_prediction(new_list2, vectorizer, classifiers, labels)


#local feature importance

local_vi = get_local_weights_df(vectorizer, vectorized_sentence, classifiers, 'MOISTURISING-HYDRATING')

# Create a dataframe with the labels and probabilities
data = {'Labels': labels, 'Probabilities': predicted_probabilities[0].tolist()[0]}
df = pd.DataFrame(data)
# Specify the desired order of the labels
result =df.groupby(["Labels"])['Probabilities'].median().reset_index().sort_values('Probabilities')
# Plot the bar chart with the specified order
sns.barplot(x='Labels', y='Probabilities', data=df, order=result['Labels'])

plt.xlabel('Labels')
plt.ylabel('Probabilities')
plt.title('Predicted Probabilities (Reordered)')
plt.xticks(rotation=90)
plt.show()

