# """
#
# File: Data Preprocessing
#
# Summary: In this file we preprocess the data: standardize casing, lemmatization, tokenization
#
# Input: Log formatted file with raw conversations
#
# Output: Log formatted file but with preprocessed data
#
# """

# Core Libraries
from itertools import islice
import numexpr as ne
import numpy as np 
import pandas as pd  
import string  
from tqdm import tqdm  # Progress bar
import unicodedata

# ------------------
# NLP Related Libraries
import re  
import spacy  
from spacy.language import Language
import stanza
from symspellpy import SymSpell, Verbosity, editdistance

ne.set_vml_num_threads(8)

# Pandas options to display whole information of dataframes
# pd.set_option(
#     "display.max_rows", None
# )  # Default value of display.max_rows is 10 i.e. at max 10 rows will be printed. Set it None to display all rows in the dataframe
# pd.set_option(
#     "display.max_columns", None
# )  # Set it to None to display all columns in the dataframe
# pd.set_option("display.width", None)
# pd.set_option("display.max_colwidth", None)


def HacerPreproceso(
    Departamento="TODOS",
    Emisor="AGENTE", 
    file_name = "tbfMensajes",
    Filtro=True, 
    Juntar=False, 
    SegundosDelta=0, 
    NLP=None,
    stanzaNLP=None,
    AutoCorrect=False,
    sym_spell=None,
    use_stanza=True,
):
    """This function controls the core sequence of events to preprocess data. Modifiable options are Departamento, Emisor, Filtro, Juntar, SegundosDelta

    Args:
        Departamento (str): Filters the conversations by department. Options: TODOS, Admision_Preparatoria, Admision_Profesional, SOAE, SOAD.
        Emisor (str): Options: AGENTE, PROSPECTO, TECBOT.
        Filtro (bool): If True, filters the dataset given Emisor.
        Juntar (bool): Works along with SegundosDelta where if there are multiple lines from a single Emisor, those multi-lines are merged withing SegundosDelta time frame.
        SegundosDelta (int): Time frame to merge multi-lines.
        NLP (object): spacy language model. Is None the first iteration, then it's initialized on further processes.
        stanzaNLP (object): stanza language model. Is None the first iteration, then it's initialized on further processes.
        AutoCorrect (bool): sets if autocorrection process is going to be done. This produces inconsistent data, needs more work.
        sym_spell (bool): Is None the first iteration, then it's initialized on further processes.
        use_stanza (bool): Indicates that stanza will be used on lemmatization. STANZA LEMMATIZATION IS MORE COMPLETE ON FOR THIS FILE.

    Returns:
        DF, NLP, stanzaNLP, sym_spell: DF being the complete and preprocessed data. The rest are for reusability during the process.
    """

    # If autocorrect is set to true, we have 7 sub-processes, else 6. For ProgressBar.
    if AutoCorrect == True:
        total = 7
    else:
        total = 6

    # ProgressBar initialization
    with tqdm(
        total=total, bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}", leave=False
    ) as pbar:

        # 1) Read Logs
        pbar.set_description(f"Leyendo Logs...")
        DF = LeerLogs('./data/{}.csv'.format(file_name))
        pbar.write(f"Done: Leyendo Logs.")
        pbar.update(1)

        # 2) Filter Logs
        pbar.set_description(f"Filtrando Logs...")
        DF = FiltrarEmisor(DF, Emisor, Filtro)
        pbar.write(f"Done: Filtrando Logs.")
        pbar.update(1)

        # 3) Read Names Dataset
        pbar.set_description(f"Leyendo Nombres...")
        NombresNPArray = LeerNombresTemp()
        pbar.write(f"Done: Leyendo Nombres.")
        pbar.update(1)

        # Loads if Language Models are not loaded
        if NLP == None:
            # 4.1) Initialize Spacy/Stanza
            pbar.set_description(f"Inicializando Spacy...")
            NLP, stanzaNLP = CargarSpacy(use_stanza)
            pbar.write(f"Done: Inicializando Spacy.")
            pbar.update(1)
        else:
            # 4.2) Language model is already initialized
            pbar.write(f"Done: Spacy ya se encuentra inicializado.")
            pbar.update(1)

        # Check if autocorrection is going to be done
        if AutoCorrect == True:
            # If autocorrect object is not initialized
            if sym_spell == None:
                # 5) Initialize Dictionary
                pbar.set_description(f"Inicializando Diccionario...")
                sym_spell = CargaSymSpell()
                pbar.write(f"Done: Inicializando Diccionario.")
                pbar.update(1)

        # 5|6) Lemmatize
        pbar.set_description(f"Lematizando...")
        DF = TokenizaYLematiza(
            DF, NombresNPArray, NLP, stanzaNLP, AutoCorrect, sym_spell
        )
        pbar.write(f"Done: Lematizando.")
        pbar.update(1)

        # Column is normalized
        DF["OracionLematizada"] = (
            DF["OracionLematizada"]
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )

        # 6|7) Write Files
        pbar.set_description(f"Escribiendo Logs Preprocesados...")
        # Checks if conversations are merged to indicate in filename
        if Juntar == True:
            DF.to_csv(
                "./data/processed/internal/admisiones/chat_reports/clean_logs/logs_"
                + (Departamento + "_" + file_name).capitalize()
                + "_sec_"
                + str(SegundosDelta)
                + ".csv",
                encoding="utf-8",
                index=False,
            )
        else:
            DF.to_csv(
                "./data/processed/internal/admisiones/chat_reports/clean_logs/logs_"
                + (Departamento + "_" + file_name).capitalize()
                + "_sec_No.csv",
                encoding="utf-8",
                index=False,
            )
        pbar.write(f"Done: Escribiendo Logs Preprocesados.")
        pbar.update(1)

    return DF, NLP, stanzaNLP, sym_spell


def LeerLogs(full_file_name):
    """Returns a DF of the specified file

    Args:
        full_file_name: file name with the path to read from and extension
    """
    DF = pd.read_csv(
        full_file_name,
        encoding="utf-8",
        keep_default_na=False,
        na_values=["", " "],
        )
    return DF

def FiltrarEmisor(DF, Emisor, Filtro):
    """Filter the dataset by Emisor.

    Args:
        DF (dataset): Dataset from conversations.
        Emisor (str): Options: AGENTE, PROSPECTO, TECBOT.
        Filtro (bool): Describes if dataset will be filtered.
    Returns:
        DF: Dataset from conversations.
    """

    if Filtro:
        if Emisor in ["AGENTE", "PROSPECTO"]:
            DF = DF[DF["emisor"] == Emisor]
        else:
            DF = DF[DF["emisor"] == "TECBOT"]
    return DF

# Global variable for names array
NombresNPArray_g = None

def LeerNombresTemp():
    return np.array(["Ana", "Pedro"])

def LeerNombres():
    """Reads names dataset and stores them in global variable NombresNPArray_g.
    
    Data origin: Muestra de Nombres y Apellidos Comunes en Mexico. http://www.datamx.io/dataset/muestra-de-nombres-y-apellidos-comunes-en-mexico 

    Returns:
        NombresNPArray: Arrray containing common names in Mexico.
    """

    # Reads male names
    NombresHombres = pd.read_csv(
        "../data/raw/external/nombres_personas/hombres.csv", encoding="utf-8"
    )
    NombresHombres = NombresHombres.drop_duplicates(subset=["nombre"], keep="first")
    NombresHombres = NombresHombres[NombresHombres["nombre"].str.len() >= 4]
    NombresHombres["Nombre"] = NombresHombres["nombre"].str.lower()
    NombresHombres["Nombre"] = (
        NombresHombres["Nombre"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    NombresHombres = NombresHombres["Nombre"]

    # Reads female names
    NombresMujeres = pd.read_csv(
        "../data/raw/external/nombres_personas/mujeres.csv", encoding="utf-8"
    )
    NombresMujeres = NombresMujeres.drop_duplicates(subset=["nombre"])
    NombresMujeres = NombresMujeres[NombresMujeres["nombre"].str.len() >= 4]
    NombresMujeres["Nombre"] = NombresMujeres["nombre"].str.lower()
    NombresMujeres["Nombre"] = (
        NombresMujeres["Nombre"]
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
    )
    NombresMujeres = NombresMujeres["Nombre"]

    # Concatenates names and stores as array
    Nombres = pd.concat([NombresHombres, NombresMujeres], axis=0)
    Nombres = Nombres.to_list()
    NombresNPArray = np.asarray(Nombres)

    return NombresNPArray


def CargarSpacy(use_stanza):
    """Loads language models to be used. It can return both Spacy and Stanza since both can be used.

    Returns:
        NLP, stanzaNLP: Spacy and Stanza language models.
    """

    # Loads es_core_news_lg from spacy, without ner and parser
    NLP = spacy.load("es_core_news_lg", disable=["ner", "parser"])
    # Pipe when using spacy (they are both technically the same, but is like this to allow custom options). The other is disabled at runtime.
    NLP.add_pipe("Lematiza_spacy", name="Lematiza_spacy")
    # Pipe when using stanza (they are both technically the same, but is like this to allow custom options). The other is disabled at runtime.
    NLP.add_pipe("Lematiza_stanza", name="Lematiza_stanza", last=True)
    # If Stanza will be used, it is loaded, None if not used
    if use_stanza == True:
        stanzaNLP = stanza.Pipeline(
            "es",
            processors="tokenize,mwt,pos,lemma",
            tokenize_no_ssplit=True,
            verbose=False,
        )
    else:
        stanzaNLP = None

    # Some preprositions not covered in raw language models
    preposiciones = [
        "a",
        "e",
        "u",
        "durante",
        "según",
        "ante",
        "en",
        "sin",
        "bajo",
        "entre",
        "so",
        "cabe",
        "hacia",
        "sobre",
        "con",
        "hasta",
        "tras",
        "contra",
        "mediante",
        "versus",
        "de",
        "para",
        "via",
        "desde",
        "por",
        "y",
        "o",
    ]

    # Prepositions are added as stop words
    for p in preposiciones:
        NLP.vocab[p].is_stop = True

    return NLP, stanzaNLP


def CargaSymSpell():
    """Load dictionary for autocorrect process. Since inconsistencies happen when using this, parameters such as max_dictionary_edit_distance and prefix_length are to be modified.

    Returns:
        sym_spell: dictionary
    """

    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=14)
    dictionary_path = "utils\\dict\\es_full.txt"
    sym_spell.load_dictionary(
        dictionary_path, term_index=0, count_index=1, encoding="utf8"
    )
    sym_spell._distance_algorithm = editdistance.DistanceAlgorithm.DAMERUAUOSA

    return sym_spell

# A list of words that are not to be replaced when lemmatizing
lista_no_reemplaza_g = [
    "hola",
    "gracias",
    "campo",
    "liga",
    "bloque",
    "dato",
    "archivo",
    "papa",
    "mama",
    "folio",
    "fecha",
    "correo",
    "linea",
    "apoyo",
    "sede",
    "matricula",
    "nomina",
    "tesoreria",
    "pagina",
    "paa",
    "paep",
    "prepa",
    "secu",
    "preparatoria",
    "universidad",
    "campus",
    "tec21",
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
    "sinaloa",
    "guadalajara",
    "sonora",
    "hidalgo",
    "tampico",
    "irapuato",
    "toluca",
    "aguascalientes",
    "laguna",
    "zacatecas",
    "chiapas",
    "monterrey",
    "chihuahua",
    "puebla",
    "mexico",
    "queretaro",
    "juarez",
    "saltillo",
    "morelia",
    "leon",
    "veracruz",
    "obregon",
    "potosi",
    "cuernavaca",
    "santa",
    "fe",
]

# Pipe when using spacy (they are both technically the same, but is like this to allow custom options). The other is disabled at runtime.
@Language.component("Lematiza_spacy")
def Lematiza_pipe_spacy(doc):
    HuboPregunta = False
    LemmasList = []
    for Token in doc:
        Temporal = ""
        if Token.text == "?":
            HuboPregunta = True
        if not Token.is_stop:
            if Token.text in lista_no_reemplaza_g:
                Token.lemma_ == Token.text
                Temporal = ReemplazaToken(
                    Token, NombresNPArray_g, 0, lista_no_reemplaza_g
                )
            else:
                Temporal = ReemplazaToken(
                    Token, NombresNPArray_g, 0, lista_no_reemplaza_g
                )
        if ((Temporal not in string.punctuation)) and (
            re.match("([a-zA-Z0-9]+)", Temporal) != None
        ):
            LemmasList.append(Temporal)

    doc = " ".join(LemmasList)

    return doc, HuboPregunta

# Pipe when using stanza (they are both technically the same, but is like this to allow custom options). The other is disabled at runtime.
@Language.component("Lematiza_stanza")
def Lematiza_pipe_stanza(doc):
    HuboPregunta = False
    LemmasList = []
    for Token in doc:
        Temporal = ""
        if Token.text == "?":
            HuboPregunta = True
        if not Token.is_stop:
            if Token.text in lista_no_reemplaza_g:
                Token.lemma_ == Token.text
                Temporal = ReemplazaToken(
                    Token, NombresNPArray_g, 1, lista_no_reemplaza_g
                )
            else:
                Temporal = ReemplazaToken(
                    Token, NombresNPArray_g, 1, lista_no_reemplaza_g
                )
        if ((Temporal not in string.punctuation)) and (
            re.match("([a-zA-Z0-9]+)", Temporal) != None
        ):
            LemmasList.append(Temporal)

    doc = str(" ".join(LemmasList))

    return doc, HuboPregunta


def ReemplazaToken(Token, NombresNPArray, stanza_flag, lista_no_reemplaza):
    """Function to replace words if they match a specific pattern, and then flagged accordingly.

    Args:
        Token (str): spacy token
        NombresNPArray (array): array of common names in Mexico. See LeerNombres().
        stanza_flag (bool): If stanza is used (value = 1), we look for the text member, not the lemma. If not used (value = 0), lemma is looked after.
        lista_no_reemplaza (list): contains words not to be replaced.

    Returns:
        [type]: [description]
    """

    # Patron de una matircula o nomina. Ej. A00829621 o l00829621
    PatronMatricula = "([A|a][0-9]{5,8})"
    PatronNomina = "([L|l][0-9]{5,8})"

    # Matricula
    if bool(re.search(PatronMatricula, str(Token.lemma_))):
        Temporal = "zzzmatricula"
    # Negation
    elif str(Token.lemma).lower() == "no":
        Temporal = "zzzneg"
    # If not replaced
    elif (lista_no_reemplaza != None) and (str(Token.text) in lista_no_reemplaza):
        Temporal = Token.text
    # Some word-specific standardization (mty > monterrey)
    elif str(Token.lemma).lower() == "mty":
        Temporal = "monterrey"
    # Some word-specific standardization (gdl > guadalajara)
    elif str(Token.lemma).lower() == "gdl":
        Temporal = "guadalajara"
    # Some word-specific standardization (mex > mexico)
    elif str(Token.lemma).lower() == "mex":
        Temporal = "mexico"
    # Some word-specific standardization (prepa > preparatoria)
    elif str(Token.text).lower() == "prepa":
        Temporal = "preparatoria"
    # Some word-specific standardization (secu > secundaria)
    elif str(Token.text).lower() == "secu":
        Temporal = "secundaria"
    # Nomina
    elif bool(re.search(PatronNomina, str(Token.lemma_))):
        Temporal = "zzznomina"
    # Email
    elif Token.like_email:
        Temporal = "zzzemail"
    # URL
    elif Token.like_url:
        Temporal = "zzzurl"
    # Number
    elif (Token.pos_ == "NUM" or Token.like_num) or (
        re.search(r"(^\d+$)", str(Token.lemma_))
    ):
        Temporal = "zzznumero"
    # Is a common name
    elif (
        str(Token.text)
        in NombresNPArray[
            0:,
        ]
    ):  # (token.text in nombres):
        Temporal = "zzznombre"
    # Is not to be replaced by flag, meaning will be lemmatized (if not already by stanza)
    else:
        # Checks if stanza will be used for lemmatization (such case value = 1). Value = 0 to lemmatize with spacy.
        if stanza_flag == 0:
            Temporal = str(Token.lemma_)
        else:
            Temporal = str(Token.text)
        # word normalization
        Temporal = Temporal.lower()
        Temporal_NFKD = unicodedata.normalize("NFKD", Temporal)
        Temporal = "".join([c for c in Temporal_NFKD if not unicodedata.combining(c)])
        if re.search(r"zzz", Temporal):
            Temporal = Temporal
        else:
            Temporal = re.sub(r"[^a-zA-Z]+", "", Temporal)

    return Temporal

# For successful stanza handling, docs are read in chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def TokenizaYLematiza(DF, NombresNPArray, NLP, stanzaNLP, AutoCorrect, sym_spell):
    """Core process for lemmatization

    Args:
        DF (dataset): Dataset from conversations.
        NombresNPArray (array): Arrray containing common names in Mexico.
        NLP (object): spacy language model.
        stanzaNLP (object): stanza language model.
        AutoCorrect (bool): states if autocorrect is going to be applied.
        sym_spell (object): dictionary for autocorrect.

    Returns:
        DF: Dataset from conversations.
    """
    DFLength = len(DF)

    # Checks if autocorrection will be done
    if AutoCorrect == True:
        # Start reading dictionary values from 5th element and on
        dict_list = list(islice(sym_spell.words.items(), 5))
        # Verifies if dict_list is loaded correctly
        if len(dict_list) > 0:
            print("Diccionario cargado exitosamente!")
        else:
            print("ERROR AL CARGAR EL DICCIONARIO, INTERRUMPE EL KERNEL INMEDIATAMENTE")
        # Start performing autocorrection
        with tqdm(
            total=DFLength,
            bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}",
            leave=False,
        ) as pbar3:
            pbar3.set_description(f"Corrigiendo errores...")
            CuerpoCorrectoList = []
            # n^2 process where each word from each line is read
            for row in DF.itertuples():
                OracionCorrecta = []
                for item in str(row.Cuerpo).split(" "):
                    # Create an ignore regex from words that are not to be replaced
                    regex_pattern = "(" + "|".join(lista_no_reemplaza_g) + ")"
                    # Gather word suggestions
                    suggestions = sym_spell.lookup(
                        item,
                        Verbosity.CLOSEST,
                        max_edit_distance=2,
                        include_unknown=True,
                        ignore_token=regex_pattern,
                    )
                    # Select first suggestion
                    suggestion = str(suggestions[0]).split(",")[0]
                    # Add word to sentence
                    OracionCorrecta.append(suggestion)
                # Add sentence to body
                CuerpoCorrectoList.append(" ".join(OracionCorrecta))
                pbar3.update(1)
            # String normalization
            DF["CuerpoCorregido"] = CuerpoCorrectoList
            DF["CuerpoCorregido"] = DF["CuerpoCorregido"].str.lower()
            DF["CuerpoCorregido"] = (
                DF["CuerpoCorregido"]
                .str.normalize("NFKD")
                .str.encode("ascii", errors="ignore")
                .str.decode("utf-8")
            )
        pbar3.close()

    DFLength = len(DF)

    # Load names if not already loaded
    global NombresNPArray_g
    if NombresNPArray_g == None:
        NombresNPArray_g = NombresNPArray

    # Preprocessing stage
    with tqdm(
        total=3, bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}", leave=False
    ) as pbar2:
        pbar2.set_description(f"Preprocesando Logs...")
        OracionLematizadaList = []
        HuboPreguntaList = []
        # Column changes if autocorrect is done
        if AutoCorrect == True:
            idx = "CuerpoCorregido"
        else:
            idx = "Cuerpo"

        # Normalize empty cases
        DF[idx] = [re.sub(" +", " ", str(x)) for x in DF[idx]]
        DF[idx] = [re.sub("\n+", " ", str(x)) for x in DF[idx]]

        # If stanza is not used
        if stanzaNLP == None:
            for doc, hubo_pregunta in NLP.pipe(
                iter(DF[idx]), disable=["Lematiza_stanza"]
            ):
                OracionLematizadaList.append(doc)
                HuboPreguntaList.append(hubo_pregunta)
                pbar2.update(1)
        # Stanza is used. This code is more complete as is the main procedure for thesis.
        else:
            list_doc_text = []
            pbar2.set_description(f"Tokenizando...")

            # Runs the pipe and states for empty cases as zzzignore
            for Doc, HuboPregunta in NLP.pipe(
                iter(DF[idx]), disable=["tagger", "ner", "parser", "Lematiza_spacy"]
            ):
                joined = [token for token in Doc.split()]
                if len(joined) > 0:
                    list_doc_text.append(str(" ".join(joined)) + "\n\n")
                else:
                    list_doc_text.append("zzzignore" + "\n\n")
                HuboPreguntaList.append(HuboPregunta)

            # Chunk size is 500
            n = 500
            # List of batches
            doc_batches = list(chunks(list_doc_text, n))

            list_sentences = []
            pbar2.set_description(f"Stanza NLP...")
            # Stanza process
            with tqdm(
                total=len(list_doc_text),
                bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}",
                leave=False,
            ) as pbar3:
                # Read each batch
                for batch in doc_batches:
                    pbar3.set_description(f"Stanza batches...")
                    # Read each sentence in the batch
                    for sent in stanzaNLP(batch).sentences:
                        list_doc = []
                        # Read each word in the sentence
                        for word in sent.words:
                            # If not stop word
                            if not NLP.vocab[word.text].is_stop:
                                # If is not to be replaced word
                                if (str(word.text)) in lista_no_reemplaza_g:
                                    list_doc.append(word.text)
                                else:
                                    # If flagged with 'zzz'
                                    if "zzz" in word.text:
                                        temporal_token = (
                                            "zzz" + word.text[3:].upper() + "zzz"
                                        )
                                        list_doc.append(temporal_token)
                                    # Lemmatize word
                                    else:
                                        list_doc.append(word.lemma)
                        if len(list_doc) > 0:
                            list_sentences.append(" ".join(list_doc))
                        else:
                            list_sentences.append(" ")
                        pbar3.update(1)
            OracionLematizadaList = list_sentences

        # Result is stored in OracionLematizada
        DF["OracionLematizada"] = OracionLematizadaList
        DF["HuboPregunta"] = HuboPreguntaList

        pbar2.write(f"Done: Preprocesando Logs.")

    pbar2.close()
    return DF

HacerPreproceso()