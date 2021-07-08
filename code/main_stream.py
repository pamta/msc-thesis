import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import spacy
import es_core_news_lg
import matplotlib.pyplot as plt
import unicodedata
from yellowbrick.text import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer
import webbrowser
import os

from data_acquisition.scripts import ReportsToLogs, agentes_extract
from data_processing.scripts import DataPreprocessing

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def write_tree(filename, sentences, type_tree, word, format_ = 'implicit'):
    file_full = filename + '.html'
    f = open(file_full, "w")

    f.write('<html>\n')
    f.write('<head>\n')
    f.write('\t<script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>\n')
    f.write('\t<script type="text/javascript">\n')
    f.write('\tgoogle.charts.load(\'current\', {packages:[\'wordtree\']});\n')
    f.write('\tgoogle.charts.setOnLoadCallback(drawChart);\n')
    f.write('\n')
    f.write('\tfunction drawChart() {\n')
    f.write('\t\tvar data = google.visualization.arrayToDataTable(\n')
    f.write('\t\t[ \n')
    f.write('\t\t[\'Phrases\'],\n')
    for sentence in sentences:
        f.write(f'\t\t[\'{sentence}\'],\n')
    f.write('\t\t]\n')
    f.write('\t\t);\n')
    f.write('\n')
    f.write('\t\tvar options = {\n')
    f.write('\t\twordtree: {\n')
    f.write(f'\t\t\ttype: \'{type_tree}\',\n')
    f.write(f'\t\t\tformat: \'{format_}\',\n')
    f.write(f'\t\t\tword: \'{word}\'\n')
    f.write('\t\t}\n')
    f.write('\t\t};\n')
    f.write('\n')
    f.write('\t\tvar chart = new google.visualization.WordTree(document.getElementById(\'wordtree_basic\'));\n')
    f.write('\t\tchart.draw(data, options);\n')
    f.write('\t}\n')
    f.write('\t</script>\n')
    f.write('</head>\n')
    f.write('<body>\n')
    f.write('\t<div id="wordtree_basic" style="width: 900px; height: 500px;"></div>\n')
    f.write('</body>\n')
    f.write('</html>\n')

    f.close()


class FileUpload(object):
    
    def __init__(self):
        self.fileTypes = ['csv']
        
    def run(self):
        """
        Upload class
        """
        file = st.sidebar.file_uploader("Upload file", type=['csv'])
        show_file = st.sidebar.empty()
        if not file:
            show_file.info(f"Please upload a file: {['csv']}")
            return
        
        df = pd.read_csv(file, encoding='utf8')
        
        try:
            df['text']
        except:
            st.sidebar.error("CSV File does not contain 'text' column")
            return None
        
        st.dataframe(df.head(10))
        return df
    
def runAnalytics(Departamento, Emisor, check_wordcloud, check_frequencies):
    st.write(f"Exploring Text Analytics on {str(Departamento)}")
    df = pd.read_csv(f"../data/processed/internal/admisiones/chat_reports/clean_logs/logs_{str(Departamento).capitalize()}_sec_No.csv", encoding='utf8')
    df = df[((df['IniciaTecBotBool'] == False) & (df['HuboTransferBool'] == False) & (df['Emisor'] == Emisor))]
    df = df.dropna()
    df['OracionLematizada'] = [str(text) for text in df['OracionLematizada']]
    if check_wordcloud == True:
        # Parse all the rows into a single string
        body_text = " ".join(str(text) for text in df["OracionLematizada"])
        st.write(f"There are {len(body_text):,d} words in texts.")
        st.markdown("## Wordcloud")
        # Generate a word cloud image:
        wordcloud = WordCloud(width=1080, height=720).generate(body_text)
        # Display the generated image:
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title("WordCloud texts")
        plt.show()
        plt.tight_layout(pad=0)
        st.pyplot(plt)
        plt.close()
    if check_frequencies == True:
        st.markdown("## Frequencies")
        # Load the text data
        corpus = df["OracionLematizada"]
        NLP = es_core_news_lg.load()
        SpacyStopwords = NLP.Defaults.stop_words
        # StopwordsOriginal = SpacyStopwords.words("spanish") # stopwords de spacy o nltk cuales son mejor?
        StopwordsSP = []
        for Word in SpacyStopwords:
            Stop = Word.lower()
            Stop_NFKD = unicodedata.normalize('NFKD', Stop)
            Stop = u"".join([c for c in Stop_NFKD if not unicodedata.combining(c)])
            StopwordsSP.append(Stop)
        preposiciones = ['tu', 'a', 'e', 'u', 'durante', 'según', 'ante', 'en', 'sin', 'bajo', 'entre', 'so', 'cabe', 'hacia', 'sobre', 'con', 'hasta', 'tras', 'contra', 'mediante', 'versus', 'de', 'para', 'via', 'desde', 'por', 'y', 'o']
        StopwordsSP = StopwordsSP + preposiciones
        vectorizer = CountVectorizer(stop_words=StopwordsSP, ngram_range=(1,2))
        docs       = vectorizer.fit_transform(text for text in corpus)
        features   = vectorizer.get_feature_names()
        visualizer = FreqDistVisualizer(
            features=features, size=(1080, 1500), n=100
        )
        visualizer.fit(docs)
        visualizer.show()
        st.pyplot(plt)
        plt.close()

def main():
    """
    Run this function to display Strealit app
    """
    
    st.title('txtviz')
    st.write("""
             ## Explore text features.
             
             
             ---
             """)
    st.sidebar.header("SETTINGS")
    st.sidebar.markdown("Here you can edit and run your pipeline settings.")
    st.sidebar.markdown("## PROCESSING")
    st.sidebar.markdown("### Agents")
    
    agents_button = st.sidebar.button('Process Agents')
    if agents_button:
        agentes_extract.process_agents()
    st.sidebar.markdown("### Report To Logs")
    #data = FileUpload()
    #data.run()
    
    Departamento = st.sidebar.selectbox(
        "Departamento",
        (
            'Admision_Profesional', 'SOAD', 'Admision_Preparatoria', 'Admision_Profesional_Ingles', 'SOAD_Ingles', 'Admision_Preparatoria_Ingles', 'AdmisionAnaly', 'SOAE', 'Tec_Bot_ATR', 'Tec_Bot_ADyAE'
        )
    )
    
    check_filter = st.sidebar.checkbox('filter', True)
    check_juntar = st.sidebar.checkbox('juntar', False)
    segundos_delta = st.sidebar.slider('segundos delta', 0, 60, 0)
    
    ExcelLeido = False
    Excel_Original = None
    Agents_Original = None
    
    transform_button = st.sidebar.button('Run Report to Logs')
    if transform_button:
        DF, Excel_Original, Agents_Original = ReportsToLogs.Logs(Departamento = Departamento, Filtro = check_filter, Juntar = check_juntar, SegundosDelta = segundos_delta, ExcelLeido = ExcelLeido, Excel_Original = Excel_Original, Agents_Original = Agents_Original)
        ExcelLeido = True
        
    st.sidebar.markdown("### Preprocess")
    
    Emisor = st.sidebar.selectbox(
        "Emisor",
        (
            'AGENTE', 'PROSPECTO', 'TECBOT'
        )
    )
    check_autocorrect = st.sidebar.checkbox('autocorrect', True)
    check_stanza = st.sidebar.checkbox('use stanza', True)
       
    NLP = None
    stanzaNLP = None
    sym_spell = None
    
    preprocess_button =  st.sidebar.button('Run Preprocess')
    if preprocess_button:
        DF, NLP, stanzaNLP, sym_spell = DataPreprocessing.HacerPreproceso(Departamento = Departamento, Emisor = Emisor, Filtro = check_filter, Juntar = check_juntar, SegundosDelta = segundos_delta, NLP = NLP, stanzaNLP = stanzaNLP, AutoCorrect = check_autocorrect, sym_spell = sym_spell, use_stanza = check_stanza)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ANALYSIS")
    
    st.sidebar.markdown("### Text Analytics")
    check_wordcloud = st.sidebar.checkbox('wordcloud', True)
    check_frequencies = st.sidebar.checkbox('frequencies', True)
    analytics_button =  st.sidebar.button('Run Text Analytics')
    if analytics_button:
        runAnalytics(Departamento, Emisor, check_wordcloud, check_frequencies)
        
    st.sidebar.markdown("### Wordtree")
    text_tree = st.sidebar.text_input("Tree Text")
    type_tree = st.sidebar.selectbox(
        "Tree Type",
        (
            'prefix', 'suffix', 'double'
        )
    )
    format_tree = st.sidebar.selectbox(
        "Tree Format",
        (
            'implicit', 'explicit'
        )
    )
    word_tree_button =  st.sidebar.button('Run Wordtree')
    if word_tree_button:
        filename = 'demo_tree'
        arbol = ['i live in a nation that provides', 'my nation is cool', 'i miss my nation in the good times']
        write_tree(filename, arbol, type_tree, text_tree, format_tree)
        path = os.getcwd()
        url = 'file:///' + path + '/' + filename + '.html'
        url = url.replace('\\', '/')
        webbrowser.get('windows-default').open(url, new=2)
        # st.write(url)
        st.info("done writing wordtree")
    
if __name__ == "__main__":
    main()