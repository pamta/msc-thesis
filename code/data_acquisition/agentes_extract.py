import re
import pandas as pd

def process_agents():
    agentes = []
    for i in range(0, 12, 1):
        df = pd.read_csv(f"../data/raw/internal/admisiones/chat_reports/{i+1}.csv", encoding='utf8')
        for item in df['Propietario: Nombre completo'].unique().tolist():
            if str(item) not in agentes:
                if str(item) != "nan":
                    agentes.append(item)

    idx = 0                
    for item in agentes:
        if idx == 0:
            agentes[idx] = str(item)
        else:
            agentes[idx] = "\n" + str(item)
        idx = idx + 1

    f = open("../data/processed/internal/admisiones/agents_names/agentes.txt", "w")
    f.writelines(agentes)
    f.close()

    # --------------------------------------------------------------------

    agentes_sh = []
    idx = 0
    for item in agentes:
        if len(item.split()) == 3:
            if idx > 0:
                agentes_sh.append('\n' + item.split()[0])
            else:
                agentes_sh.append(item.split()[0])
        elif len(item.split()) == 4:
            if idx > 0:
                agentes_sh.append('\n' + ' '.join(item.split()[:2]))
            else:
                agentes_sh.append(' '.join(item.split()[:2]))
        idx = idx + 1

    f = open("../data/processed/internal/admisiones/agents_names/agentes_sh.txt", "w")
    f.writelines(agentes_sh)
    f.close()