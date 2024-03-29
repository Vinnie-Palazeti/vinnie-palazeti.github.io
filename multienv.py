import os
import subprocess

envs = {
    'base':"/Users/mvp/.pyenv/versions/quarto_base/bin/python",
    'NGBoost':"/Users/mvp/.pyenv/versions/NGBoost/bin/python",
    'llm':"/Users/mvp/.pyenv/versions/llm/bin/python",
    'consumer_sent':"/Users/mvp/.pyenv/versions/consumer_sent/bin/python",
    'trees':"/Users/mvp/.pyenv/versions/trees/bin/python",
}

dir = os.path.join(os.getcwd(),'posts') 
paths = [i for i in os.listdir(dir) if '.' not in i]

for path in paths:
    if path in envs.keys():
        python_v = envs[path]
    else:
        python_v = envs['base']
    print(f'rendering...')
    process = f'QUARTO_PYTHON={python_v} quarto render posts/{path}/index.qmd'
    print(f'cmd: {process}')
    result = subprocess.run(process, shell=True)

subprocess.run("quarto render", shell=True)