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
import datetime # Manejo de datetimes
import io # Lectura y escritura de archivos
import pandas as pd # Pandas para manejo de las dataframes
import string # Manipulacion de strings
from tqdm import tqdm # Progress bar
import re # Regex
import unicodedata # unicodedata

# Pandas options to display whole information of dataframes
pd.set_option('display.max_rows', None) # Default value of display.max_rows is 10 i.e. at max 10 rows will be printed. Set it None to display all rows in the dataframe
pd.set_option('display.max_columns', None) # Set it None to display all columns in the dataframe
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

def Logs(Filtro = False, Departamento = "TODOS", Juntar = False, SegundosDelta = 43200, ExcelLeido = False, Excel_Original = None, Agents_Original = None):
    """
    
    Main function that calls the rest of the functions to parse a report to a log

    Args:
        Filtro (bool, optional):        Indicates if only a department set in Departamento will be filtered. Defaults to False.
        Departamento (str, optional):   Name of a department to filter. Defaults to "TODOS".
        Juntar (bool, optional):        Indicates if consecutive dialogues from the same sender will be joined and taken as one 
                                        dialogue gien a time treshold between messages set in SegundosDelta. Defaults to False.
        SegundosDelta (int, optional):  Indicates the amount of seconds that will be taken into account as threshold for 
                                        determining consecutive messages from the same sender. Defaults to 43200.

    Returns:
        DataFrame: a dataframe in the format of a log.
    """
    
    print(f"Departamento: {Departamento}\nFiltro: {Filtro}")
    DF, Excel_Original, Agents_Original = HacerLogs(Filtro, Departamento, Juntar, SegundosDelta, ExcelLeido, Excel_Original, Agents_Original)
    return DF, Excel_Original, Agents_Original

def HacerLogs(Filtro, Departamento, Juntar, SegundosDelta, ExcelLeido, Excel_Original, Agents_Original):    
    """
    
    Function that contains the general pipeline of reading excels, parsing to log format and return such log formatted
    DataFrame.

    Args:
        Filtro (bool):          Indicates if only a department set in Departamento will be filtered.
        Departamento (string):  Name of a department to filter.
        Juntar (bool):          Indicates if consecutive dialogues from the same sender will be joined and
                                taken as one dialogue gien a time treshold between messages set in SegundosDelta. 
        SegundosDelta (int):    Indicates the amount of seconds that will be taken into account as threshold for 
                                determining consecutive messages from the same sender.

    Returns:
        DataFrame: a dataframe in the format of a log.
    """
    
    # Variables para funcion de lectura
    PathToExcel = '../data/raw/internal/admisiones/chat_reports/'
    ExcelFile = 'full_year.xlsx'
    Sheets = None
    ColumnsNames = ['CasoID', 'ConversacionID', 'Departamento', 'Cuerpo']
    Columns = 'A, B, J, W'

    Section = 0
    
    if (Juntar == True):
        Total = 6
    else:
        Total = 5
    
    with tqdm(total = Total, bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar:
        # DF sera el dataframe conteniendo los chats
        pbar.set_description(f"Leyendo Excel")
        
        if ExcelLeido == False:
            DF = LeerExcels(PathToExcel, ExcelFile, Sheets, ColumnsNames, Columns)
            Excel_Original = DF.copy()
            ExcelLeido = True
        else:
            DF = Excel_Original.copy()
        
        # quitamos donde haya NA en caso y en cuerpo
        DF = DF.dropna(axis = 0, how = 'all', subset=['Cuerpo'])
        pbar.write(f'Done: Leyendo Excel')
        pbar.update(1)

        # ---------------------------------------------

        pbar.set_description(f"Filtrando Departamento")
        DF = FiltrarDepartamento(DF, Filtro, Departamento)
        pbar.write(f'Done: Filtrando Departamento')
        pbar.update(1)
        
        pbar.set_description(f"Leyendo Agents")
        if Agents_Original == None:
            Agentes = LeerAgentes()
            Agents_Original = Agentes.copy()
        else:
            Agentes = Agents_Original.copy()
        pbar.write(f'Done: Leyendo Agents')
        pbar.update(1)
        
        pbar.set_description(f"Ajustando DF")
        DF = ProcesaDF(DF, Agentes, pbar, Departamento)
        pbar.write(f'Done: Ajustando DF')
        pbar.update(1)
        
        # pbar.set_description(f"Procesando: Secuencia Conversaciones")
        # DF = EstableceSecuenciaConversaciones(DF)
        # pbar.update(1)
        
        # FILTRAR CONVERSACIONES BASURA
        if (Departamento == 'SOAD'):
            DF = DF[DF['ConversacionID'] != 124331]
            DF = DF[DF['ConversacionID'] != 126164]
            DF = DF[DF['ConversacionID'] != 129982]
            DF = DF[DF['ConversacionID'] != 137285]
            DF = DF[DF['ConversacionID'] != 126004]
        
        if (Juntar == True):
            pbar.set_description(f"Procesando: Junta Conversaciones")
            DF = JuntaConversacionesEmisorSeguido(DF, SegundosDelta)
            pbar.update(1)
        
        pbar.set_description(f"Escribiendo CSV")
        
        if (Juntar == True):
            DF.to_csv('../data/processed/internal/admisiones/chat_reports/logs/logs_' + str(Departamento).capitalize() + '_sec_' + str(SegundosDelta) + '.csv', encoding='utf-8', index = False)
        else:
            DF.to_csv('../data/processed/internal/admisiones/chat_reports/logs/logs_' + str(Departamento).capitalize() + '_sec_No.csv', encoding='utf-8', index = False)
        pbar.write(f'Done: Escribiendo CSV')
        pbar.update(1)
        
    return DF, Excel_Original, Agents_Original

def LeerExcels(PathToExcel, ExcelFile, Sheets, ColumnsNames, Columns, Excel=None):
    """
    
    Function to read an excel containing the reports and return it as a DataFrame.

    Args:
        PathToExcel (string):   directory where the excel to be read is located.
        ExcelFile (string):     name of the file where the reports are stored. In this case the reports from January
                                to December, 2019 are contained in a file called all.xlsx where a sheet indicated the
                                month, being 1, 2, 3, ..., 10, 11, 12 sheets.
        Sheets (array):         serves to indicate if the search on the excel file will be for specifically a determined
                                sheet or if set to None, all sheets will be used.
        ColumnsNames (array):   an array of string indicating the columns to be set for the returned DataFrame.
        Columns (array):        an array of the indices of the columns of the excel file (in this case we call the columns
                                by their excel nomenclature, that being A, B, C...).

    Returns:
        DataFrame: a concatenation of each of the sheets in the file that is being read.
    """
    
    PathToExcel = PathToExcel + ExcelFile
    
    return pd.concat(pd.read_excel(PathToExcel, sheet_name = None, names = ColumnsNames, dtype = object, usecols = Columns, keep_default_na = False, na_values=['', ' ']))
    

def FiltrarDepartamento(DF, Filtro, Departamento):
    """[summary]

    Args:
        DF ([type]):            [description]
        Filtro ([type]):        [description]
        Departamento ([type]):  [description]

    Returns:
        [type]: [description]
    """
    
    if (Filtro):
        return DF[DF['Departamento'] ==  Departamento]
    else:
        return DF

def LeerAgentes():
    """[summary]

    Returns:
        [type]: [description]
    """
    
    F = io.open("../data/processed/internal/admisiones/agents_names/agentes_sh.txt", mode = "r")
    Agentes_F = F.read().splitlines()
    F.close()
    
    G = io.open("../data/processed/internal/admisiones/agents_names/agentes.txt", mode = "r")
    Agentes_G = G.read().splitlines()
    G.close()
    
    Agentes = Agentes_F + Agentes_G
    
    return Agentes

def CalculaSecuencia(DF, Secuencia, ConversacionFixedID):
    """[summary]

    Args:
        DF ([type]):                    [description]
        Secuencia ([type]):             [description]
        ConversacionFixedID ([type]):   [description]

    Returns:
        [type]: [description]
    """
    
    SecuenciaList = []
    with tqdm(total = len(DF), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar:
        pbar.set_description(f"Calcuando Secuencias")
        for Index, Row in enumerate(DF.itertuples(), 1):
            ConversacionID = int(Row.ConversacionID)
            if (Secuencia == -1) and (ConversacionFixedID == -1):
                Secuencia = 0
                ConversacionFixedID = int(ConversacionID)
            elif ConversacionID == ConversacionFixedID:
                Secuencia += 1
            else:
                Secuencia = 0
                ConversacionFixedID = int(ConversacionID)

            SecuenciaList.append(Secuencia)
            
            pbar.update(1)
        
        pbar.write(f"Done: Calculando Secuencias")
        
    return SecuenciaList

def CalculaTimespan(DF, Timespan, TimespanList, ConversacionFixedID):
    """[summary]

    Args:
        DF ([type]):                    [description]
        Timespan ([type]):              [description]
        TimespanList ([type]):          [description]
        ConversacionFixedID ([type]):   [description]

    Returns:
        [type]: [description]
    """
    
    HuboErrorPasteEnConversacionBoolList = []
    with tqdm(total = len(DF), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar:
        pbar.set_description(f"Calcuando Timespans")
        for Index, Row in enumerate(DF.itertuples(), 1):
            HuboErrorPasteEnConversacionBool = False
            ConversacionID = int(Row.ConversacionID)
            if (Timespan == -1) and (ConversacionFixedID == -1):
                Timespan = pd.to_timedelta(0)
                ConversacionFixedID = int(ConversacionID)
                TimespanPrevia = pd.to_timedelta(0)
                Secuencia = int(Row.Secuencia)
            elif (ConversacionID == ConversacionFixedID) and (Row.Secuencia == (Secuencia + 1)):
                if pd.to_timedelta(Row.Timespan) >= TimespanPrevia:
                    Timespan = pd.to_timedelta(Row.Timespan)
                elif (pd.to_timedelta(Row.Timespan) < TimespanPrevia) or (pd.to_timedelta(Row.Timespan) == pd.to_timedelta(pd.offsets.Second(1))) :
                    Timespan = pd.to_timedelta(TimespanPrevia) + pd.to_timedelta(Row.Timespan)
                    HuboErrorPasteEnConversacionBool = True
                elif pd.to_timedelta(Row.Timespan) == None:
                    Timespan = pd.to_timedelta(0)
                Secuencia = int(Row.Secuencia)
                TimespanPrevia = pd.to_timedelta(Row.Timespan)
            else:
                Timespan = pd.to_timedelta(0)
                ConversacionFixedID = int(ConversacionID)
                TimespanPrevia = pd.to_timedelta(0)
                Secuencia = int(Row.Secuencia)
            HuboErrorPasteEnConversacionBoolList.append(HuboErrorPasteEnConversacionBool)
            TimespanList.append(Timespan)
            pbar.update(1)
            
        pbar.write(f"Done: Calcuando Timespans")
        
    return (TimespanList, HuboErrorPasteEnConversacionBoolList)

def NormalizaCuerpo(DF):
    """[summary]

    Args:
        DF ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    CuerpoNormalizadoList = []
    HuboTransferenciaBoolList = [] # transfer_bool_list = []
    with tqdm(total = len(DF), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar:
        pbar.set_description(f"Normalizando Cuerpo")
        for Index, Row in enumerate(DF.itertuples(), 1):
            HuboTransferenciaBool = False
            Texto = str(Row.Cuerpo)
            RegularExpression00 = re.compile(r'(Ha comenzado el chat)')
            RegularExpression01 = re.compile(r'(Chat Started)')
            RegularExpression02 = re.compile(r'(Chat transferido desde\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\s{0,1}A\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
            RegularExpression03 = re.compile(r'(Chat transferred From\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\s{0,1}To\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
            RegularExpression04 = re.compile(r'(Agente\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}transfirió correctamente la plática de chat al botón\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
            RegularExpression05 = re.compile(r'(Lo siento no te entiendo, aún sigo aprendiendo)')
            RegularExpression08 = re.compile(r'(Agent Chatbot failed to transfer the chat to button ButtonId\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
            RegularExpression09 = re.compile(r'(Agent Chatbot successfully transferred the chat to button ButtonId\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
            RegularExpression10 = re.compile(r'(Este servicio está disponible de lunes a viernes de 8:00 a 22:00 horas y sábados de 10:00 a 13:00 horas, tiempo del centro de México \(exceptuando vacaciones y asuetos\); puedes contáctarnos también a través de la cuenta admisiones@servicios.itesm.mx)')
            RegularExpression11 = re.compile(r'Origen de chat')
            RegularExpression12 = re.compile(r'Agente Chatbot no transfirió')
            
            if (RegularExpression00.search(Texto)) or (RegularExpression01.search(Texto)):
                RegularExpression = "zzzinicio"
            elif (RegularExpression11.search(Texto)):
                RegularExpression = 'zzzorigen'
            elif (RegularExpression02.search(Texto)) or (RegularExpression03.search(Texto)):
                RegularExpression = "zzztransfer"
                HuboTransferenciaBool = True
            elif (RegularExpression05.search(Texto)) and (str(Row.Emisor) == 'TECBOT'):
                RegularExpression = 'zzzbotfail'
            elif (RegularExpression08.search(Texto) and (str(Row.Emisor) == 'SYS')) or (RegularExpression12.search(Texto) and (str(Row.Emisor) == 'SYS')):
                RegularExpression = 'zzztransferfail'
            elif (RegularExpression09.search(Texto) and (str(Row.Emisor) == 'SYS')) or (RegularExpression04.search(Texto) and (str(Row.Emisor) == 'SYS')):
                RegularExpression = 'zzztransfersuccess'
            elif (RegularExpression10.search(Texto)) and ((str(Row.Emisor) == 'SYS') or (str(Row.Emisor) == 'TECBOT')):
                RegularExpression = 'zzzinfochat'
            else:
                #Temporal01 = RegularExpression06.search(Texto)
                #if Temporal01 != None:
                #   Match = Temporal01.group(1)
                #Temporal02 = RegularExpression07.search(str(Texto))
                #print(Temporal02, Texto)
                #if Temporal02 != None:
                #    RegularExpression = Temporal02.group(1)
                #else:
                #    RegularExpression = None
                if len(str(Texto)) > 0:
                    RegularExpression = Texto
                else:
                    RegularExpression = ''

            CuerpoNormalizado = RegularExpression
            CuerpoNormalizado = str(CuerpoNormalizado).strip()
            CuerpoNormalizadoList.append(CuerpoNormalizado)
            HuboTransferenciaBoolList.append(HuboTransferenciaBool)

            pbar.update(1)
        
        pbar.write(f"Done: Normalizando Cuerpo")
        
    return (CuerpoNormalizadoList, HuboTransferenciaBoolList)

def NormalizaMenuYRespuestas(DF, Departamento):
    """[summary]

    Args:
        DF ([type]):            [description]
        Departamento ([type]):  [description]

    Returns:
        [type]: [description]
    """
    
    Departamentos_ATR = ['Admision_Profesional', 'Admision_Preparatoria', 'Admision_Profesional_Ingles', 'SOAD_Ingles', 'Admision_Preparatoria_Ingles', 'AdmisionAnaly', 'Tec_Bot_ATR']
    
    Departamentos_ADyAE = ['SOAD', 'SOAE', 'Tec_Bot_ADyAE']
    
    Bot = ''
    if Departamento in Departamentos_ATR:
        Bot = 'ATR'
    elif Departamento in Departamentos_ADyAE:
        Bot = 'ADyAE'
    
    PathToExcel = '../data/processed/internal/admisiones/chatbot_structure/'
    ExcelFile = 'menus.xlsx'
    PathToExcel = PathToExcel + ExcelFile
    SheetName = "Menus"
    ColumnsNames = ['ID', 'Menu']
    Columns = 'A, B'
    MenuDF = pd.read_excel(io = PathToExcel, sheet_name = SheetName, names = ColumnsNames, dtype = object, usecols = Columns)
    
    MenuDF['ID'] = MenuDF['ID'].str.strip()
    MenuDF['Menu'] = MenuDF['Menu'].str.strip()
    MenuDF['Menu'] = MenuDF['Menu'].str.lower()
    MenuDF['Menu'] = MenuDF['Menu'].str.normalize('NFKD')\
           .str.encode('ascii', errors='ignore')\
           .str.decode('utf-8')
    MenuDictionary = dict(zip(MenuDF.Menu, MenuDF.ID))
                        
    PathToExcel = '../data/processed/internal/admisiones/chatbot_structure/'
    ExcelFile = 'menus.xlsx'
    PathToExcel = PathToExcel + ExcelFile
    SheetName = "Answers" + Bot
    ColumnsNames = ['ID', 'Answer']
    Columns = 'B, C'
    RespuestaDF = pd.read_excel(io = PathToExcel, sheet_name = SheetName, names = ColumnsNames, dtype = object, usecols = Columns)
    
    RespuestaDF['ID'] = RespuestaDF['ID'].str.strip()
    RespuestaDF['Answer'] = RespuestaDF['Answer'].str.strip()
    RespuestaDF['Answer'] = RespuestaDF['Answer'].str.lower()
    RespuestaDF['Answer'] = RespuestaDF['Answer'].str.normalize('NFKD')\
           .str.encode('ascii', errors='ignore')\
           .str.decode('utf-8')
    RespuestaDictionary = dict(zip(RespuestaDF.Answer, RespuestaDF.ID))
    
    CuerpoMenuList = []
    CuerpoRespuestaList = []
    
    DF['Cuerpo'] = DF['Cuerpo'].str.strip()
    DF['Cuerpo'] = DF['Cuerpo'].str.lower()
    DF['CuerpoNorm'] = DF['Cuerpo'].str.normalize('NFKD')\
           .str.encode('ascii', errors='ignore')\
           .str.decode('utf-8')
    
    CuerpoMenuList = DF['CuerpoNorm'].map(MenuDictionary)
    CuerpoRespuestaList = DF['CuerpoNorm'].map(RespuestaDictionary)
    DF['CuerpoMenu'] = CuerpoMenuList
    DF['CuerpoRespuesta'] = CuerpoRespuestaList
    CuerpoList = []
    
    with tqdm(total = len(DF), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar3:
        pbar3.set_description(f"Normalizando Menus y Respuestas")
        for Index, Row in enumerate(DF.itertuples(), 1):
            
            # if (Row.Cuerpo == 'soy el bot del tec de monterrey, en este momento solo he aprendido a contestar temas relacionados con los procesos de admision y becas, en esta etapa no puedo responder preguntas directas, te ofrezco un menu de opciones?'):
                # print(f'R: {Row.CuerpoRespuesta}')
            
            if (Row.IniciaTecBotBool == True):
                if (pd.isna(Row.CuerpoMenu) == True) and (pd.isna(Row.CuerpoRespuesta) == True):
                    CuerpoList.append(str(Row.CuerpoNorm).lower())
                elif (pd.isna(Row.CuerpoMenu) == False) and (pd.isna(Row.CuerpoRespuesta) == True):
                    CuerpoList.append(str(Row.CuerpoMenu).lower())
                elif (pd.isna(Row.CuerpoMenu) == True) and (pd.isna(Row.CuerpoRespuesta) == False):
                    CuerpoList.append(str(Row.CuerpoRespuesta).lower())
                else:
                    CuerpoList.append("zzzerror")
            else:
                CuerpoList.append(str(Row.CuerpoNorm).lower())
            
            pbar3.update(1)
        
        pbar3.write(f"Done: Encontrando Transferencias En Chats De TecBot")
            
    DF = DF.drop('Cuerpo', 1)
    DF = DF.drop('CuerpoNorm', 1)
    DF = DF.drop('CuerpoMenu', 1)
    DF = DF.drop('CuerpoRespuesta', 1)
    DF['Cuerpo'] = CuerpoList
    
    return DF

def EncontrarTransferenciasEnChatsDeTecBot(DF):
    """[summary]

    Args:
        DF ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    TransferenciaList = []
    ConversacionID = -1
    Sigue = False
    EsTecBot = False
    with tqdm(total = len(DF), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar3:
        pbar3.set_description(f"Encontrando Transferencias En Chats De TecBot")
        for Index, Row in enumerate(DF.itertuples(), 1):
            if (int(Row.Secuencia) == 1) and (str(Row.Emisor) == "TECBOT"):
                EsTecBot = True
            elif (int(Row.Secuencia) == 1) and (str(Row.Emisor) != "TECBOT"):
                EsTecBot = False
                
            if bool(EsTecBot):
                if (int(Row.Secuencia) != 0) and ((str(Row.Cuerpo) == "zzztransfer") or (str(Row.Cuerpo) == "zzzinicio")):
                    ConversacionID = int(Row.ConversacionID)
                    Sigue = True

                if bool(Sigue) and (int(Row.ConversacionID) != ConversacionID):
                    ConversacionID = -1
                    Sigue = False
                    EsTecBot = False
            else:
                ConversacionID = -1
                Sigue = False

            pbar3.update(1)
            TransferenciaList.append(Sigue)
            
        pbar3.write(f"Done: Encontrando Transferencias En Chats De TecBot")
        
    return TransferenciaList

def EstableceSecuenciaConversaciones(DF):
    """[summary]

    Args:
        DF ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    Secuencia = -1
    ConversacionFixedID = -1
    SecuenciaList = []
    SecuenciaList = CalculaSecuencia(DF, Secuencia, ConversacionFixedID)    
    DF['Secuencia'] = SecuenciaList  
    
    return DF

def JuntaConversacionesEmisorSeguido(DF, SegundosDelta):
    """[summary]

    Args:
        DF ([type]):            [description]
        SegundosDelta ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    DFNew = pd.DataFrame()
    
    Delta = pd.Timedelta(pd.offsets.Second(int(SegundosDelta)))

    Cuerpo = str
    EmisorAnterior = str
    ConversacionID = int

    Cuerpo = ''
    EmisorAnterior = ''
    ConversacionID = 0
    i = 0
    
    ListaDepartamento = []
    ListaCaso = []
    ListaConversacion = []
    ListaSecuencia = []
    ListaHuboTransferBool = []
    ListaFechaAjustada = []
    ListaHuboErrorPasteEnConversacionBool = []
    ListaZonaHoraria = []
    ListaCuerpo = []
    ListaEmisor = []
    ListaHuboTransferenciaEnChatTecBotBool = []
    ListaTecBotMarcaMensajeNoEntendidoBool = []
    ListaMensajeNoEntiendeTecBotBool = []
    ListaIniciaTecBotBool = []

    for Index, Row in enumerate(DF.itertuples(), 1):
        # inicializacion
        if EmisorAnterior == '':
            Departamento = str(Row.Departamento)
            CasoID = str(Row.CasoID)
            ConversacionID = int(Row.ConversacionID)
            Secuencia = 0
            HuboTransferBool = bool(Row.HuboTransferBool)
            FechaAjustada = pd.to_datetime(Row.FechaAjustada)
            HuboErrorPasteEnConversacionBool = bool(Row.HuboErrorPasteEnConversacionBool)
            ZonaHoraria = str(Row.ZonaHoraria)
            Cuerpo = str(Row.Cuerpo)
            EmisorAnterior = str(Row.Emisor)
            HuboTransferenciaEnChatTecBotBool = bool(Row.HuboTransferenciaEnChatTecBotBool)
            TecBotMarcaMensajeNoEntendidoBool = bool(Row.TecBotMarcaMensajeNoEntendidoBool)
            MensajeNoEntiendeTecBotBool = bool(Row.MensajeNoEntiendeTecBotBool)
            IniciaTecBotBool = bool(Row.IniciaTecBotBool)
            DeltaFechaAjustada = pd.to_datetime(Row.FechaAjustada)
            
            if (Index == (len(DF))):
                ListaDepartamento.append(str(Departamento))
                ListaCaso.append(str(CasoID))
                ListaConversacion.append(int(ConversacionID))
                ListaSecuencia.append(int(Secuencia))
                ListaHuboTransferBool.append(bool(HuboTransferBool))
                ListaFechaAjustada.append(FechaAjustada)
                ListaHuboErrorPasteEnConversacionBool.append(bool(HuboErrorPasteEnConversacionBool))
                ListaZonaHoraria.append(str(ZonaHoraria))
                ListaCuerpo.append(str(Cuerpo))
                ListaEmisor.append(str(EmisorAnterior))
                ListaHuboTransferenciaEnChatTecBotBool.append(bool(HuboTransferenciaEnChatTecBotBool))
                ListaTecBotMarcaMensajeNoEntendidoBool.append(bool(TecBotMarcaMensajeNoEntendidoBool))
                ListaMensajeNoEntiendeTecBotBool.append(bool(MensajeNoEntiendeTecBotBool))
                ListaIniciaTecBotBool.append(bool(IniciaTecBotBool))
                
            continue

        if ConversacionID != int(Row.ConversacionID):
            ListaDepartamento.append(str(Departamento))
            ListaCaso.append(str(CasoID))
            ListaConversacion.append(int(ConversacionID))
            ListaSecuencia.append(int(Secuencia))
            ListaHuboTransferBool.append(bool(HuboTransferBool))
            ListaFechaAjustada.append(FechaAjustada)
            ListaHuboErrorPasteEnConversacionBool.append(bool(HuboErrorPasteEnConversacionBool))
            ListaZonaHoraria.append(str(ZonaHoraria))
            ListaCuerpo.append(str(Cuerpo))
            ListaEmisor.append(str(EmisorAnterior))
            ListaHuboTransferenciaEnChatTecBotBool.append(bool(HuboTransferenciaEnChatTecBotBool))
            ListaTecBotMarcaMensajeNoEntendidoBool.append(bool(TecBotMarcaMensajeNoEntendidoBool))
            ListaMensajeNoEntiendeTecBotBool.append(bool(MensajeNoEntiendeTecBotBool))
            ListaIniciaTecBotBool.append(bool(IniciaTecBotBool))

            Departamento = str(Row.Departamento)
            CasoID = str(Row.CasoID)
            ConversacionID = int(Row.ConversacionID)
            Secuencia = 0
            HuboTransferBool = bool(Row.HuboTransferBool)
            FechaAjustada = pd.to_datetime(Row.FechaAjustada)
            HuboErrorPasteEnConversacionBool = bool(Row.HuboErrorPasteEnConversacionBool)
            ZonaHoraria = str(Row.ZonaHoraria)
            Cuerpo = str(Row.Cuerpo)
            EmisorAnterior = str(Row.Emisor)
            HuboTransferenciaEnChatTecBotBool = bool(Row.HuboTransferenciaEnChatTecBotBool)
            TecBotMarcaMensajeNoEntendidoBool = bool(Row.TecBotMarcaMensajeNoEntendidoBool)
            MensajeNoEntiendeTecBotBool = bool(Row.MensajeNoEntiendeTecBotBool)
            IniciaTecBotBool = bool(Row.IniciaTecBotBool)
            DeltaFechaAjustada = pd.to_datetime(Row.FechaAjustada)
            
            if (Index == (len(DF))):
                ListaDepartamento.append(str(Departamento))
                ListaCaso.append(str(CasoID))
                ListaConversacion.append(int(ConversacionID))
                ListaSecuencia.append(int(Secuencia))
                ListaHuboTransferBool.append(bool(HuboTransferBool))
                ListaFechaAjustada.append(FechaAjustada)
                ListaHuboErrorPasteEnConversacionBool.append(bool(HuboErrorPasteEnConversacionBool))
                ListaZonaHoraria.append(str(ZonaHoraria))
                ListaCuerpo.append(str(Cuerpo))
                ListaEmisor.append(str(EmisorAnterior))
                ListaHuboTransferenciaEnChatTecBotBool.append(bool(HuboTransferenciaEnChatTecBotBool))
                ListaTecBotMarcaMensajeNoEntendidoBool.append(bool(TecBotMarcaMensajeNoEntendidoBool))
                ListaMensajeNoEntiendeTecBotBool.append(bool(MensajeNoEntiendeTecBotBool))
                ListaIniciaTecBotBool.append(bool(IniciaTecBotBool))
                
            continue

        # se mantiene el mismo emisor que el anterior
        if (EmisorAnterior == Row.Emisor) and (pd.to_timedelta(pd.to_datetime(Row.FechaAjustada) - DeltaFechaAjustada) < Delta):
            ListaTemporal = []
            ListaTemporal.append(Cuerpo)
            ListaTemporal.append(str(Row.Cuerpo))
            Cuerpo = " ".join(ListaTemporal)
            
            if (HuboTransferBool) or (bool(Row.HuboTransferBool)):
                HuboTransferBool = True
            if (HuboErrorPasteEnConversacionBool) or (bool(Row.HuboErrorPasteEnConversacionBool)):
                HuboErrorPasteEnConversacionBool = True
            if (HuboTransferenciaEnChatTecBotBool) or (bool(Row.HuboTransferenciaEnChatTecBotBool)):
                HuboTransferenciaEnChatTecBotBool = True
            if (TecBotMarcaMensajeNoEntendidoBool) or (bool(Row.TecBotMarcaMensajeNoEntendidoBool)):
                TecBotMarcaMensajeNoEntendidoBool = True
            if (MensajeNoEntiendeTecBotBool) or (bool(Row.MensajeNoEntiendeTecBotBool)):
                MensajeNoEntiendeTecBotBool = True
            if (IniciaTecBotBool) or (bool(Row.IniciaTecBotBool)):
                IniciaTecBotBool = True
                
            DeltaFechaAjustada = pd.to_datetime(Row.FechaAjustada)
            
            if (Index == (len(DF))):
                ListaDepartamento.append(str(Departamento))
                ListaCaso.append(str(CasoID))
                ListaConversacion.append(int(ConversacionID))
                ListaSecuencia.append(int(Secuencia))
                ListaHuboTransferBool.append(bool(HuboTransferBool))
                ListaFechaAjustada.append(FechaAjustada)
                ListaHuboErrorPasteEnConversacionBool.append(bool(HuboErrorPasteEnConversacionBool))
                ListaZonaHoraria.append(str(ZonaHoraria))
                ListaCuerpo.append(str(Cuerpo))
                ListaEmisor.append(str(EmisorAnterior))
                ListaHuboTransferenciaEnChatTecBotBool.append(bool(HuboTransferenciaEnChatTecBotBool))
                ListaTecBotMarcaMensajeNoEntendidoBool.append(bool(TecBotMarcaMensajeNoEntendidoBool))
                ListaMensajeNoEntiendeTecBotBool.append(bool(MensajeNoEntiendeTecBotBool))
                ListaIniciaTecBotBool.append(bool(IniciaTecBotBool))
            
            continue

        # cambia de emisor
        else:
            ListaDepartamento.append(str(Departamento))
            ListaCaso.append(str(CasoID))
            ListaConversacion.append(int(ConversacionID))
            ListaSecuencia.append(int(Secuencia))
            ListaHuboTransferBool.append(bool(HuboTransferBool))
            ListaFechaAjustada.append(FechaAjustada)
            ListaHuboErrorPasteEnConversacionBool.append(bool(HuboErrorPasteEnConversacionBool))
            ListaZonaHoraria.append(str(ZonaHoraria))
            ListaCuerpo.append(str(Cuerpo))
            ListaEmisor.append(str(EmisorAnterior))
            ListaHuboTransferenciaEnChatTecBotBool.append(bool(HuboTransferenciaEnChatTecBotBool))
            ListaTecBotMarcaMensajeNoEntendidoBool.append(bool(TecBotMarcaMensajeNoEntendidoBool))
            ListaMensajeNoEntiendeTecBotBool.append(bool(MensajeNoEntiendeTecBotBool))
            ListaIniciaTecBotBool.append(bool(IniciaTecBotBool))

            Departamento = str(Row.Departamento)
            CasoID = str(Row.CasoID)
            ConversacionID = int(Row.ConversacionID)
            Secuencia += 1
            HuboTransferBool = bool(Row.HuboTransferBool)
            FechaAjustada = pd.to_datetime(Row.FechaAjustada)
            HuboErrorPasteEnConversacionBool = bool(Row.HuboErrorPasteEnConversacionBool)
            ZonaHoraria = str(Row.ZonaHoraria)
            Cuerpo = str(Row.Cuerpo)
            EmisorAnterior = str(Row.Emisor)
            HuboTransferenciaEnChatTecBotBool = bool(Row.HuboTransferenciaEnChatTecBotBool)
            TecBotMarcaMensajeNoEntendidoBool = bool(Row.TecBotMarcaMensajeNoEntendidoBool)
            MensajeNoEntiendeTecBotBool = bool(Row.MensajeNoEntiendeTecBotBool)
            IniciaTecBotBool = bool(Row.IniciaTecBotBool)
            DeltaFechaAjustada = pd.to_datetime(Row.FechaAjustada)
            
            if (Index == (len(DF))):
                ListaDepartamento.append(str(Departamento))
                ListaCaso.append(str(CasoID))
                ListaConversacion.append(int(ConversacionID))
                ListaSecuencia.append(int(Secuencia))
                ListaHuboTransferBool.append(bool(HuboTransferBool))
                ListaFechaAjustada.append(FechaAjustada)
                ListaHuboErrorPasteEnConversacionBool.append(bool(HuboErrorPasteEnConversacionBool))
                ListaZonaHoraria.append(str(ZonaHoraria))
                ListaCuerpo.append(str(Cuerpo))
                ListaEmisor.append(str(EmisorAnterior))
                ListaHuboTransferenciaEnChatTecBotBool.append(bool(HuboTransferenciaEnChatTecBotBool))
                ListaTecBotMarcaMensajeNoEntendidoBool.append(bool(TecBotMarcaMensajeNoEntendidoBool))
                ListaMensajeNoEntiendeTecBotBool.append(bool(MensajeNoEntiendeTecBotBool))
                ListaIniciaTecBotBool.append(bool(IniciaTecBotBool))

            continue
            
    DFNew['Departamento'] = ListaDepartamento
    DFNew['CasoID'] = ListaCaso
    DFNew['ConversacionID'] = ListaConversacion
    DFNew['Secuencia'] = ListaSecuencia
    DFNew['HuboTransferBool'] = ListaHuboTransferBool
    DFNew['FechaAjustada'] = ListaFechaAjustada
    DFNew['HuboErrorPasteEnConversacionBool'] = ListaHuboErrorPasteEnConversacionBool
    DFNew['ZonaHoraria'] = ListaZonaHoraria
    DFNew['Cuerpo'] = ListaCuerpo
    DFNew['Emisor'] = ListaEmisor
    DFNew['HuboTransferenciaEnChatTecBotBool'] = ListaHuboTransferenciaEnChatTecBotBool
    DFNew['TecBotMarcaMensajeNoEntendidoBool'] = ListaTecBotMarcaMensajeNoEntendidoBool
    DFNew['MensajeNoEntiendeTecBotBool'] = ListaMensajeNoEntiendeTecBotBool
    DFNew['IniciaTecBotBool'] = ListaIniciaTecBotBool
            
    DFNew['ConversacionID'] = DFNew['ConversacionID'].astype(int)
    
    return DFNew

def MarcarTecBotNoEntendio(DF):
    """[summary]

    Args:
        DF ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    Lista = []
    ListaBad = []
    Fail = False
    with tqdm(total = len(DF), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar3:
        pbar3.set_description(f"Buscando Mensajes No Entendidos Por Bot")
        for Index, Row in enumerate(DF.itertuples(), 1):
            if (str(Row.Cuerpo) == "zzzbotfail") and (str(Row.Emisor) == 'TECBOT'):
                Fail = True
                ListaBad[len(ListaBad) - 1] = Fail
            else:
                Fail = False

            pbar3.update(1)
            ListaBad.append(Fail)
            Lista.append(Fail)
            
        pbar3.write(f"Done: Buscando Mensajes No Entendidos Por Bot")
    
    return (Lista, ListaBad)

def MarcarTecBotInicia(DF):
    """[summary]

    Args:
        DF ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    IniciaTecBotLista = []
    Inicia = False
    with tqdm(total = len(DF), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar3:
        pbar3.set_description(f"Buscando Conversaciones Iniciadas Por Bot")
        Control = 0
        for Index, Row in enumerate(DF.itertuples(), 1):
            Secuencia = int(Row.Secuencia)
            if (Secuencia == 0):
                Control = 0
                Inicia = False
            else:
                Control = Control + 1
            Emisor = str(Row.Emisor)
            Cuerpo = str(Row.Cuerpo)
            if ((Cuerpo == "Hola, soy Tecbot :)") or (Cuerpo == "Hola, soy Tecbot")) and (Emisor == "TECBOT"):
                Inicia = True
                for i in range(Control):
                    IniciaTecBotLista[len(IniciaTecBotLista) - (i + 1)] = Inicia

            IniciaTecBotLista.append(Inicia)
            pbar3.update(1)
            
        pbar3.write(f"Done: Buscando Conversaciones Iniciadas Por Bot")

    return IniciaTecBotLista

def ProcesaDF(DF, Agentes, pbar, Departamento):
    """[summary]

    Args:
        DF ([type]):            [description]
        Agentes ([type]):       [description]
        pbar ([type]):          [description]
        Departamento ([type]):  [description]

    Returns:
        [type]: [description]
    """
    
    pbar.set_description(f"Cuestiones de fechas")
    
    # Regex que extrae el "Ha comenzado el chat:..."
    DF['Fecha'] = DF['Cuerpo'].str.extract('(\\w+\\s\\d{2},\\s\\d{4}.\\s\\d{2}:\\d{2}:\\d{2})', expand = True)

    # Regex que extrae el timezone de la conversacion y lo guarda en su respectiva columna
    DF['ZonaHoraria'] = DF['Cuerpo'].str.extract('([+-][0-9]{4})', expand = True)

    # Date a datetime formato
    DF['Fecha'] = pd.to_datetime(DF['Fecha'])
    
    pbar.set_description(f"Temporales 0")
    DF['Cuerpo'] = DF.Cuerpo.str.strip()
    Temporal00 = pd.DataFrame(DF.Cuerpo.str.split(r'(?=\( \ds \))').tolist(), index = DF.ConversacionID).stack()
    Temporal00 = Temporal00.reset_index()[[0, 'ConversacionID']]
    Temporal00.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal00['Cuerpo'] = Temporal00.Cuerpo.str.strip()
    
    pbar.set_description(f"Temporales 1")
    Temporal01 = pd.DataFrame(Temporal00.Cuerpo.str.split(r'[ ](?=\(\s\d{0,2}h{0,1}\s{0,1}\d{0,2}m{0,1}\s{0,1}\d{1,2}s{1}\s\))').tolist(), index = Temporal00.ConversacionID).stack()
    Temporal01 = Temporal01.reset_index()[[0, 'ConversacionID']] 
    Temporal01.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal01['Cuerpo'] = Temporal01.Cuerpo.str.strip()
    
    pbar.set_description(f"Temporales 2")
    Temporal02 = pd.DataFrame(Temporal01.Cuerpo.str.split(r'[ ](?=Chat transferido desde\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\s{0,1}A\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}?$)').tolist(), index = Temporal01.ConversacionID).stack()
    Temporal02 = Temporal02.reset_index()[[0, 'ConversacionID']] 
    Temporal02.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal02['Cuerpo'] = Temporal02.Cuerpo.str.strip()
    
    pbar.set_description(f"Temporales 3")
    Temporal03 = pd.DataFrame(Temporal02.Cuerpo.str.split(r'[ ](?=Chat transferred From\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\s{0,1}To\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}?$)').tolist(), index = Temporal02.ConversacionID).stack()
    Temporal03 = Temporal03.reset_index()[[0, 'ConversacionID']] 
    Temporal03.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal03['Cuerpo'] = Temporal03.Cuerpo.str.strip()    
    
    pbar.set_description(f"Temporales 4")
    Temporal04 = pd.DataFrame(Temporal03.Cuerpo.str.split(r'[ ](?=Origen de chat: \w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30} Agente \w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}?$)').tolist(), index = Temporal03.ConversacionID).stack()
    Temporal04 = Temporal04.reset_index()[[0, 'ConversacionID']] 
    Temporal04.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal04['Cuerpo'] = Temporal04.Cuerpo.str.strip()
        
    pbar.set_description(f"Temporales 5")
    Temporal05 = pd.DataFrame(Temporal04.Cuerpo.str.split(r'[ ](?=Chat Started:\s{0,1}\w+, \w+\s\d{2},\s\d{4}.\s\d{2}:\d{2}:\d{2} \(.\d{0,1}\d{0,1}\d{0,1}\d{0,1}\)?$)').tolist(), index = Temporal04.ConversacionID).stack()
    Temporal05 = Temporal05.reset_index()[[0, 'ConversacionID']]
    Temporal05.columns = ['Cuerpo', 'ConversacionID']
    Temporal05['Cuerpo'] = Temporal05.Cuerpo.str.strip()
    
    pbar.set_description(f"Temporales 6")
    Temporal06 = pd.DataFrame(Temporal05.Cuerpo.str.split(r'[ ](?=Ha comenzado el chat: \w+, \w+\s\d{2},\s\d{4}.\s\d{2}:\d{2}:\d{2} \(.\d{0,1}\d{0,1}\d{0,1}\d{0,1}\)?$)').tolist(), index = Temporal05.ConversacionID).stack()
    Temporal06 = Temporal06.reset_index()[[0, 'ConversacionID']] 
    Temporal06.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal06['Cuerpo'] = Temporal06.Cuerpo.str.strip()
    
    pbar.set_description(f"Temporales 7")
    Temporal07 = pd.DataFrame(Temporal06.Cuerpo.str.split(r'[ ](?=Agente \w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30} transfirió correctamente la plática de chat al botón \w{0,30}\s{0,1}\w{0,30}?$)').tolist(), index = Temporal06.ConversacionID).stack()
    Temporal07 = Temporal07.reset_index()[[0, 'ConversacionID']] 
    Temporal07.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal07['Cuerpo'] = Temporal07.Cuerpo.str.strip()
    
    pbar.set_description(f"Temporales 8")
    Temporal08 = pd.DataFrame(Temporal07.Cuerpo.str.split(r'[ ](?=Agente \w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30} transfirió correctamente la plática de chat al botón ButtonId \w{0,30}\s{0,1}\w{0,30}?$)').tolist(), index = Temporal07.ConversacionID).stack()
    Temporal08 = Temporal08.reset_index()[[0, 'ConversacionID']] 
    Temporal08.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal08['Cuerpo'] = Temporal08.Cuerpo.str.strip()
        
    pbar.set_description(f"Temporales 9")
    Temporal09 = pd.DataFrame(Temporal08.Cuerpo.str.split(r'[ ](?=Agent Chatbot failed to transfer the chat to button ButtonId\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}?$)').tolist(), index = Temporal08.ConversacionID).stack()
    Temporal09 = Temporal09.reset_index()[[0, 'ConversacionID']] 
    Temporal09.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal09['Cuerpo'] = Temporal09.Cuerpo.str.strip()
    
    pbar.set_description(f"Temporales 10")
    Temporal10 = pd.DataFrame(Temporal09.Cuerpo.str.split(r'[ ](?=Agent Chatbot successfully transferred the chat to button ButtonId\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}?$)').tolist(), index = Temporal09.ConversacionID).stack()
    Temporal10 = Temporal10.reset_index()[[0, 'ConversacionID']] 
    Temporal10.columns = ['Cuerpo', 'ConversacionID'] 
    Temporal10['Cuerpo'] = Temporal10.Cuerpo.str.strip()

    # se quita el cuerpo del df original
    DF = DF.drop('Cuerpo', 1)
    
    #DF.to_csv("testDF.csv", index=True)
    #Temporal07.to_csv("testTemporal07.csv", index=True)

    pbar.set_description(f"Timespan")

    # se hace merge donde las conversaciones por mensaje se juntan con la informacion en dataframe original
    DF = pd.merge(DF, Temporal10, on = 'ConversacionID',  how = 'left')
    
    DF['Timespan'] = DF['Cuerpo'].str.extract('(\(\s\d{0,2}h{0,1}\s{0,1}\d{0,2}m{0,1}\s{0,1}\d{1,2}s{1}\s\))', expand = True)
    DF['Timespan'] = DF['Timespan'].str[2:]
    DF['Timespan'] = DF['Timespan'].str[:-1]
    DF['Timespan'] = DF.Timespan.str.strip()
    
    DF['Cuerpo'] = DF['Cuerpo'].str.extract('(?:\\(\\s\\d{0,2}h{0,1}\\s{0,1}\\d{0,2}m{0,1}\\s{0,1}\\d{1,2}s{1}\\s\\)){0,1}\\s{0,1}(.*)', expand = True)
    DF['Emisor'] = DF['Cuerpo'].str.extract('(\A\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}):.*', expand = True)
    DF['Emisor'] = [ y if not pd.isna(x) else 'SYS' for x, y in zip(DF['Timespan'], DF['Emisor']) ]
    DF['Emisor'] = ['SYS' if x == 'SYS' else 'TECBOT' if x == 'TecBot' else 'AGENTE' if x in Agentes else 'PROSPECTO' for x in DF['Emisor'] ]
    
    DF['Cuerpo2'] = DF['Cuerpo'].str.extract(':\s(.*)', expand = True)
    DF['Cuerpo3'] = [ y if z != "SYS" else x for x, y, z in zip(DF['Cuerpo'], DF['Cuerpo2'], DF['Emisor']) ]
    DF['Cuerpo'] = DF['Cuerpo3']
    del DF['Cuerpo2']
    del DF['Cuerpo3']
    
    DF['Timespan'] = [ x if not pd.isna(x) else '0s' for x in DF['Timespan'] ]

    # ListaTimespanX = []
    # ListaCuerpoX = []
    # 
    # for index, row in enumerate(DF.itertuples(),1):
    #     regex = re.compile(r'(\(\s\d{0,2}h{0,1}\s{0,1}\d{0,2}m{0,1}\s{0,1}\d{1,2}s{1}\s\))')
    #     text = str(row.Cuerpo)
    #     partido = text.partition(")")
    #     checa = partido[0] + ")"
    #     if regex.search(checa):
    #         ListaTimespanX.append(partido[0])
    #         ListaCuerpoX.append(partido[-1].lstrip())
    #     else:
    #         ListaTimespanX.append("( 0s )")
    #         ListaCuerpoX.append(text)
    # 
    # # Regex que extrae el timespan de la conversacion y lo guarda en la columna timespan
    # #DF['Timespan'] = DF['Cuerpo'].str.extract('(\(.*?\))', expand = True)
    # DF['Timespan'] = ListaTimespanX
    # DF['Cuerpo'] = ListaCuerpoX
# 
    #DF.to_csv("testDF.csv")
    
    # Se eliminan los parentesis izquierdos y derechos del timespan
    #DF['Timespan'] = DF['Timespan'].str[2:]
    #DF['Timespan'] = DF['Timespan'].str[:-1]
    
    #DF.to_csv("testDF2.csv")
    
    pbar.set_description(f"Procesando Secuencias")
    
    # Establece una secuencia en las conversaciones
    Secuencia = -1
    ConversacionFixedID = -1
    SecuenciaList = []
    SecuenciaList = CalculaSecuencia(DF, Secuencia, ConversacionFixedID)
    DF['Secuencia'] = SecuenciaList
    
    pbar.set_description(f"Procesando Timestamps")
    
    # Timestpan se formatea a tipo de dato legible por maquina de diferencia en tiempo 
    Timespan = -1
    ConversacionFixedID = -1
    TimespanList = []
    HuboErrorPasteEnConversacionBoolList = []
    (timespan_list, HuboErrorPasteEnConversacionBoolList) = CalculaTimespan(DF, Timespan, TimespanList, ConversacionFixedID)
    
    DF['Timespan'] = TimespanList 
    DF['Timespan'] = pd.to_timedelta(DF['Timespan'])
    DF['HuboErrorPasteEnConversacionBool'] = HuboErrorPasteEnConversacionBoolList 
    
    ListaFechaAjustada = []
    for index, row in enumerate(DF.itertuples(), 1):
        if (len(str(row.Timespan)) < 1):
            timespan = pasadoTimespan
        else:
            timespan = row.Timespan
            
        ListaFechaAjustada.append(row.Fecha + timespan)
        
        pasadoTimespan = row.Timespan
        
    #DF.to_csv("testDF3.csv")
    
    # Se crea columna donde se va sumando la diferencia de tiempo a la hora inicial de la conversacion  
    # DF['FechaAjustada'] = DF['Fecha'] + DF['Timespan']
    DF['FechaAjustada'] = ListaFechaAjustada

    # Se elimina la columna timespan
    DF = DF.drop('Timespan', 1)

    # Se elimina la columna date
    DF = DF.drop('Fecha', 1)

    # Regex que crea la columna emisor dada el Agente/Prospecto que inicia el mensaje en cuerpo
    # DF['Emisor'] = DF['Cuerpo'].str.extract('(.*?):.*', expand = True)
    # DF['Emisor'] = DF['Emisor'].str.extract('\(.*?\)\s(.*)')
    
    # Se crea la columna de emisor que indica quien es el que envio el mensaje
    
    #DF['Emisor'] = ['SYS' if x == 'SYS' else 'TECBOT' if x == 'TecBot' else 'AGENTE' if x in Agentes else 'PROSPECTO' for x in DF['Emisor']]
    
    #ListaEmisores = []
    
    #regex1 = re.compile(r'(Chat Started:\s{0,1}\w+,\s{0,1}\w+\s\d{2},\s\d{4}.\s\d{2}:\d{2}:\d{2}\s{0,1}\(\D\d\d\d\d\))')
    #regex2 = re.compile(r'(Agent Chatbot successfully transferred the chat to button ButtonId\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
    #regex3 = re.compile(r'(Agent Chatbot failed to transfer the chat to button ButtonId\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
    #regex4 = re.compile(r'(Chat transferred From\s\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\sTo\s\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
    #regex5 = re.compile(r'(Agente\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}transfirió correctamente la plática de chat al botón\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
    #regex6 = re.compile(r'(Chat transferido desde\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\sA\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30}\s{0,1}\w{0,30})')
    #regex7 = re.compile(r'(\sEste servicio está disponible de lunes a viernes de 8:00 a 22:00 horas y sábados de 10:00 a 13:00 horas, tiempo del centro de México \(exceptuando vacaciones y asuetos\); puedes contáctarnos también a través de la cuenta admisiones@servicios.itesm.mx)')
    
    #for index, row in enumerate(DF.itertuples(), 1):
    #    Texto = str(row.Cuerpo)
    #    
    #    if ( regex1.search(Texto) or regex2.search(Texto) or regex3.search(Texto) or regex4.search(Texto) or regex5.search(Texto) or regex6.search(Texto) or regex7.search(Texto) ):
    #        ListaEmisores.append("SYS")
    #    else:
    #        ListaEmisores.append(str(row.Emisor))
    #        
    #DF['Emisor'] = ListaEmisores
    
    pbar.set_description(f"Normalizando Cuerpo")
    
    # Normalizando Cuerpo
    CuerpoNormalizadoList = []
    HuboTransferenciaBoolList = []
    (CuerpoNormalizadoList, HuboTransferenciaBoolList) = NormalizaCuerpo(DF)
    # print(DF['Cuerpo'].head())
    DF = DF.drop('Cuerpo', 1)
    DF['Cuerpo'] = CuerpoNormalizadoList
    DF['HuboTransferBool'] = HuboTransferenciaBoolList

    # Se normaliza tengo a a-z,A-Z,1-9. Sin acentos
    DF['Cuerpo'] = DF['Cuerpo'].str.normalize('NFKD')\
           .str.encode('ascii', errors='ignore')\
           .str.decode('utf-8')
    
    # Seleccion de columnas importantes
    DF = DF[['Departamento', 'CasoID', 'ConversacionID', 'Secuencia', 'HuboTransferBool', 'FechaAjustada', 'HuboErrorPasteEnConversacionBool', 'ZonaHoraria', 'Emisor', 'Cuerpo']]
    
    DF = DF.dropna(axis = 0, how = 'all', subset=['Cuerpo'])
    
    #pbar.set_description(f"Procesando SYS")
    
    #with tqdm(total = len(DF), bar_format='{bar}|{desc}{percentage:3.0f}% {r_bar}', leave=False) as pbar2:
    #    pbar2.set_description(f"Normalizando Inicios y Transfers")
    #    EmisoresList = []
    #    for Index, Row in enumerate(DF.itertuples(), 1):
    #        if (Row.Cuerpo == 'zzzinicio') or (Row.Cuerpo == 'zzztransfer'):
    #            EmisoresList.append("SYS")
    #        else:
    #            EmisoresList.append(str(Row.Emisor))
    #        pbar2.update(1)
    #    pbar2.write(f"Done: Normalizando Inicios y Transfers")
    #       
    #DF = DF.drop('Emisor', 1)
    #DF['Emisor'] = EmisoresList
            
    #DF = DF[DF['Cuerpo'] != '']
    #DF = DF[DF['Cuerpo'] != ' ']
    #DF = DF[DF['Cuerpo'] != None]
    
    TransferenciaList = []
    TransferenciaList = EncontrarTransferenciasEnChatsDeTecBot(DF)
    DF['HuboTransferenciaEnChatTecBotBool'] = TransferenciaList
    
    Lista = []
    ListaBad = []
    Lista, ListaBad = MarcarTecBotNoEntendio(DF)
    DF['TecBotMarcaMensajeNoEntendidoBool'] = Lista
    DF['MensajeNoEntiendeTecBotBool'] = ListaBad
    
    IniciaTecBotLista = []
    IniciaTecBotLista = MarcarTecBotInicia(DF)
    DF['IniciaTecBotBool'] = IniciaTecBotLista   
    
    DF = NormalizaMenuYRespuestas(DF, Departamento)
    
    pbar.set_description(f"Procesando Secuencias")
    
    # Establece una secuencia en las conversaciones
    Secuencia = -1
    ConversacionFixedID = -1
    SecuenciaList = []
    SecuenciaList = CalculaSecuencia(DF, Secuencia, ConversacionFixedID)    
    DF['Secuencia'] = SecuenciaList  
    
    DF = DF[['Secuencia', 'CasoID', 'ConversacionID', 'Departamento', 'FechaAjustada', 'ZonaHoraria', 'Emisor', 'Cuerpo', 'IniciaTecBotBool', 'HuboTransferBool', 'HuboTransferenciaEnChatTecBotBool', 'TecBotMarcaMensajeNoEntendidoBool', 'MensajeNoEntiendeTecBotBool', 'HuboErrorPasteEnConversacionBool']]
    
    return DF