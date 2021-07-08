# """
#
# File: Reports to Logs
#
# Summary: In this file we process the raw report from admissions department and wrangle the data in order to
#             separate each converation into separate dialogues to form a log.
#
# Input: Raw report as received from the admissions department where each whole conversation is a row.
#
# Output: Log formatted CSV file for each of the departments contained in the reports.
#
# """

# Libraries
import datetime  # Manejo de datetimes
import io  # Lectura y escritura de archivos
import pandas as pd  # Pandas para manejo de las dataframes
import numpy as np  # Numpy
import numexpr as ne
import string  # Manipulacion de strings
import unicodedata

# ------------------
import spacy  # Procesos de NLP
from spacy.language import Language
import stanza
import re  # Regular expressions
from sklearn.feature_extraction.text import CountVectorizer  # Document Term Matrix
from itertools import islice
from symspellpy import SymSpell, Verbosity, editdistance
import time

# ------------------
from tqdm import tqdm  # Progress bar

ne.set_vml_num_threads(8)

# Pandas options to display whole information of dataframes
pd.set_option(
    "display.max_rows", None
)  # Default value of display.max_rows is 10 i.e. at max 10 rows will be printed. Set it None to display all rows in the dataframe
pd.set_option(
    "display.max_columns", None
)  # Set it to None to display all columns in the dataframe
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def HacerPreproceso(
    Departamento="TODOS",
    Emisor="AGENTE",
    Filtro=True,
    Juntar=False,
    SegundosDelta=0,
    NLP=None,
    stanzaNLP=None,
    AutoCorrect=False,
    sym_spell=None,
    use_stanza=True,
):
    """[summary]

    Args:
        Departamento (str, optional): [description]. Defaults to "TODOS".
        Emisor (str, optional): [description]. Defaults to "AGENTE".
        Filtro (bool, optional): [description]. Defaults to True.
        SegundosDelta (int, optional): [description]. Defaults to 0.

    Returns:
        [type]: [description]
    """

    if AutoCorrect == True:
        total = 7
    else:
        total = 6

    with tqdm(
        total=total, bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}", leave=False
    ) as pbar:

        pbar.set_description(f"Leyendo Logs...")
        DF = LeerLogs(Departamento, Juntar, SegundosDelta)
        time.sleep(1)
        pbar.write(f"Done: Leyendo Logs.")
        time.sleep(1)
        pbar.update(1)

        pbar.set_description(f"Filtrando Logs...")
        DF = FiltrarEmisor(DF, Emisor, Filtro)
        time.sleep(1)
        pbar.write(f"Done: Filtrando Logs.")
        time.sleep(1)
        pbar.update(1)

        pbar.set_description(f"Leyendo Nombres...")
        NombresNPArray = LeerNombres()
        time.sleep(1)
        pbar.write(f"Done: Leyendo Nombres.")
        time.sleep(1)
        pbar.update(1)

        if NLP == None:
            pbar.set_description(f"Inicializando Spacy...")
            NLP, stanzaNLP = CargarSpacy(use_stanza)
            time.sleep(1)
            pbar.write(f"Done: Inicializando Spacy.")
            time.sleep(1)
            pbar.update(1)
        else:
            time.sleep(1)
            pbar.write(f"Done: Spacy ya se encuentra inicializado.")
            time.sleep(1)
            pbar.update(1)

        if AutoCorrect == True:
            if sym_spell == None:
                pbar.set_description(f"Inicializando Diccionario...")
                sym_spell = CargaSymSpell()
                time.sleep(1)
                pbar.write(f"Done: Inicializando Diccionario.")
                time.sleep(1)
                pbar.update(1)

        pbar.set_description(f"Lematizando...")
        DF = TokenizaYLematiza(
            DF, NombresNPArray, NLP, stanzaNLP, AutoCorrect, sym_spell
        )
        time.sleep(1)
        pbar.write(f"Done: Lematizando.")
        time.sleep(1)
        pbar.update(1)

        DF["OracionLematizada"] = (
            DF["OracionLematizada"]
            .str.normalize("NFKD")
            .str.encode("ascii", errors="ignore")
            .str.decode("utf-8")
        )

        pbar.set_description(f"Escribiendo Logs Preprocesados...")
        if Juntar == True:
            DF.to_csv(
                "../data/processed/internal/admisiones/chat_reports/clean_logs/logs_"
                + str(Departamento).capitalize()
                + "_sec_"
                + str(SegundosDelta)
                + ".csv",
                encoding="utf-8",
                index=False,
            )
        else:
            DF.to_csv(
                "../data/processed/internal/admisiones/chat_reports/clean_logs/logs_"
                + str(Departamento).capitalize()
                + "_sec_No.csv",
                encoding="utf-8",
                index=False,
            )

        time.sleep(1)
        pbar.write(f"Done: Escribiendo Logs Preprocesados.")
        time.sleep(1)
        pbar.update(1)

    return DF, NLP, stanzaNLP, sym_spell


def LeerLogs(Departamento, Juntar, SegundosDelta):
    """[summary]

    Args:
        Departamento ([type]): [description]
        SegundosDelta ([type]): [description]

    Returns:
        [type]: [description]
    """

    # departamento = 'TODOS' # TODOS, Admision_Preparatoria, Admision_Profesional, SOAE, SOAD
    if Juntar == True:
        DF = pd.read_csv(
            f"../data/processed/internal/admisiones/chat_reports/logs/logs_{str(Departamento).capitalize()}_sec_{SegundosDelta}.csv",
            encoding="utf-8",
            keep_default_na=False,
            na_values=["", " "],
        )
    else:
        DF = pd.read_csv(
            f"../data/processed/internal/admisiones/chat_reports/logs/logs_{str(Departamento).capitalize()}_sec_No.csv",
            encoding="utf-8",
            keep_default_na=False,
            na_values=["", " "],
        )
    return DF


def FiltrarEmisor(DF, Emisor, Filtro):
    """[summary]

    Args:
        DF ([type]): [description]
        Emisor ([type]): [description]
        Filtro ([type]): [description]

    Returns:
        [type]: [description]
    """

    if (Filtro) == False:
        DF = DF
    elif (Filtro == True) and Emisor == "AGENTE":
        DF = DF[DF["Emisor"] == "AGENTE"]
    elif (Filtro == True) and Emisor == "PROSPECTO":
        DF = DF[DF["Emisor"] == "PROSPECTO"]
    else:
        DF = DF[DF["Emisor"] == "TECBOT"]

    return DF


NombresNPArray_g = None


def LeerNombres():
    """[summary]

    Returns:
        [type]: [description]
    """

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

    Nombres = pd.concat([NombresHombres, NombresMujeres], axis=0)
    Nombres = Nombres.to_list()
    NombresNPArray = np.asarray(Nombres)

    return NombresNPArray


def CargarSpacy(use_stanza):
    """[summary]

    Returns:
        [type]: [description]
    """
    NLP = spacy.load("es_core_news_lg", disable=["ner", "parser"])
    NLP.add_pipe("Lematiza_spacy", name="Lematiza_spacy")
    NLP.add_pipe("Lematiza_stanza", name="Lematiza_stanza", last=True)
    if use_stanza == True:
        stanzaNLP = stanza.Pipeline(
            "es",
            processors="tokenize,mwt,pos,lemma",
            tokenize_no_ssplit=True,
            verbose=False,
        )
    else:
        stanzaNLP = None

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

    for p in preposiciones:
        NLP.vocab[p].is_stop = True

    return NLP, stanzaNLP


def CargaSymSpell():
    """[summary]

    Returns:
        [type]: [description]
    """
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=14)
    dictionary_path = "utils\\dict\\es_full.txt"
    sym_spell.load_dictionary(
        dictionary_path, term_index=0, count_index=1, encoding="utf8"
    )
    sym_spell._distance_algorithm = editdistance.DistanceAlgorithm.DAMERUAUOSA

    return sym_spell


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
    """[summary]

    Args:
        Token ([type]): [description]
        NombresNPArray ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Patron de una matircula o nomina. Ej. A00829621 o l00829621
    PatronMatricula = "([A|a][0-9]{5,8})"
    PatronNomina = "([L|l][0-9]{5,8})"

    if bool(re.search(PatronMatricula, str(Token.lemma_))):
        Temporal = "zzzmatricula"
    elif str(Token.lemma).lower() == "no":
        Temporal = "zzzneg"
    elif (lista_no_reemplaza != None) and (str(Token.text) in lista_no_reemplaza):
        Temporal = Token.text
    elif str(Token.lemma).lower() == "mty":
        Temporal = "monterrey"
    elif str(Token.lemma).lower() == "gdl":
        Temporal = "guadalajara"
    elif str(Token.lemma).lower() == "mex":
        Temporal = "mexico"
    elif str(Token.text).lower() == "prepa":
        Temporal = "preparatoria"
    elif str(Token.text).lower() == "secu":
        Temporal = "secundaria"
    elif bool(re.search(PatronNomina, str(Token.lemma_))):
        Temporal = "zzznomina"
    elif Token.like_email:
        Temporal = "zzzemail"
    elif Token.like_url:
        Temporal = "zzzurl"
    elif (Token.pos_ == "NUM" or Token.like_num) or (
        re.search(r"(^\d+$)", str(Token.lemma_))
    ):
        Temporal = "zzznumero"
    elif (
        str(Token.text)
        in NombresNPArray[
            0:,
        ]
    ):  # (token.text in nombres):
        Temporal = "zzznombre"
    else:
        if stanza_flag == 0:
            Temporal = str(Token.lemma_)
        else:
            Temporal = str(Token.text)
        Temporal = Temporal.lower()
        Temporal_NFKD = unicodedata.normalize("NFKD", Temporal)
        Temporal = "".join([c for c in Temporal_NFKD if not unicodedata.combining(c)])
        if re.search(r"zzz", Temporal):
            Temporal = Temporal
        else:
            Temporal = re.sub(r"[^a-zA-Z]+", "", Temporal)

    return Temporal


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def TokenizaYLematiza(DF, NombresNPArray, NLP, stanzaNLP, AutoCorrect, sym_spell):
    """[summary]

    Args:
        DF ([type]): [description]
        NombresNPArray ([type]): [description]
        NLP ([type]): [description]

    Returns:
        [type]: [description]
    """
    DFLength = len(DF)

    if AutoCorrect == True:
        dict_list = list(islice(sym_spell.words.items(), 5))
        if len(dict_list) > 0:
            print("Diccionario cargado exitosamente!")
        else:
            print("ERROR AL CARGAR EL DICCIONARIO, INTERRUMPE EL KERNEL INMEDIATAMENTE")
        with tqdm(
            total=DFLength,
            bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}",
            leave=False,
        ) as pbar3:
            pbar3.set_description(f"Corrigiendo errores...")
            CuerpoCorrectoList = []
            for row in DF.itertuples():
                OracionCorrecta = []
                for item in str(row.Cuerpo).split(" "):
                    regex_pattern = "(" + "|".join(lista_no_reemplaza_g) + ")"
                    suggestions = sym_spell.lookup(
                        item,
                        Verbosity.CLOSEST,
                        max_edit_distance=2,
                        include_unknown=True,
                        ignore_token=regex_pattern,
                    )
                    suggestion = str(suggestions[0]).split(",")[0]
                    OracionCorrecta.append(suggestion)
                CuerpoCorrectoList.append(" ".join(OracionCorrecta))
                pbar3.update(1)
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

    global NombresNPArray_g
    if NombresNPArray_g == None:
        NombresNPArray_g = NombresNPArray

    with tqdm(
        total=3, bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}", leave=False
    ) as pbar2:
        pbar2.set_description(f"Preprocesando Logs...")
        OracionLematizadaList = []
        HuboPreguntaList = []
        # for i, Row in enumerate(DF.itertuples(), 1):
        if AutoCorrect == True:
            idx = "CuerpoCorregido"
        else:
            idx = "Cuerpo"

        DF[idx] = [re.sub(" +", " ", str(x)) for x in DF[idx]]
        DF[idx] = [re.sub("\n+", " ", str(x)) for x in DF[idx]]

        if stanzaNLP == None:
            for doc, hubo_pregunta in NLP.pipe(
                iter(DF[idx]), disable=["Lematiza_stanza"]
            ):
                OracionLematizadaList.append(doc)
                HuboPreguntaList.append(hubo_pregunta)
                pbar2.update(1)
        else:
            list_doc_text = []
            pbar2.set_description(f"Tokenizando...")

            for Doc, HuboPregunta in NLP.pipe(
                iter(DF[idx]), disable=["tagger", "ner", "parser", "Lematiza_spacy"]
            ):
                joined = [token for token in Doc.split()]
                if len(joined) > 0:
                    list_doc_text.append(str(" ".join(joined)) + "\n\n")
                else:
                    list_doc_text.append("zzzignore" + "\n\n")
                HuboPreguntaList.append(HuboPregunta)

            n = 500
            doc_batches = list(chunks(list_doc_text, n))

            list_sentences = []
            pbar2.set_description(f"Stanza NLP...")
            with tqdm(
                total=len(list_doc_text),
                bar_format="{bar}|{desc}{percentage:3.0f}% {r_bar}",
                leave=False,
            ) as pbar3:
                for batch in doc_batches:
                    pbar3.set_description(f"Stanza batches...")
                    for sent in stanzaNLP(batch).sentences:
                        list_doc = []
                        for word in sent.words:
                            if not NLP.vocab[word.text].is_stop:
                                if (str(word.text)) in lista_no_reemplaza_g:
                                    list_doc.append(word.text)
                                else:
                                    if "zzz" in word.text:
                                        temporal_token = (
                                            "zzz" + word.text[3:].upper() + "zzz"
                                        )
                                        list_doc.append(temporal_token)
                                    else:
                                        list_doc.append(word.lemma)
                        if len(list_doc) > 0:
                            list_sentences.append(" ".join(list_doc))
                        else:
                            list_sentences.append(" ")
                        pbar3.update(1)
            OracionLematizadaList = list_sentences

        DF["OracionLematizada"] = OracionLematizadaList
        DF["HuboPregunta"] = HuboPreguntaList

        pbar2.write(f"Done: Preprocesando Logs.")

    pbar2.close()
    return DF