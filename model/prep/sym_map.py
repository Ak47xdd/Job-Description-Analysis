"""
sym_map.py - All synonyms for the TF-IDF Vectorizer
"""

SYNONYM_MAP = {
    # docker 
    'containeriz':              'docker',
    'containerize':             'docker',

    # git 
    'version control':          'git',
    'source control':           'git',

    # github 
    'gitlab':                   'github',
    'bitbucket':                'github',
    'pull request':             'github',

    # ci/cd 
    'infrastructure-as-code':   'ci/cd',
    'continuous integration':   'ci/cd',
    'continuous deployment':    'ci/cd',
    'continuous delivery':      'ci/cd',
    'github actions':           'ci/cd',
    'gitlab ci':                'ci/cd',
    'jenkins':                  'ci/cd',
    'devops':                   'ci/cd',
    'cicd':                     'ci/cd',

    # genai 
    'generative ai':            'genai',
    'frontier model':           'genai',
    'foundation model':         'genai',
    'gen ai':                   'genai',
    'gemini':                   'genai',
    'claude':                   'genai',
    'gpt':                      'genai',

    # agents 
    'multi-agent':              'agents',
    'orchestrat':               'agents',
    'agentic':                  'agents',
    'copilot':                  'agents',

    # mcp
    'model context protocol':   'mcp',
    'protocol wrapper':         'mcp',
    'function calling':         'mcp',
    'tool call':                'mcp',

    # prompt engineering 
    'prompt engineer':          'prompt engineering',
    'chain-of-thought':         'prompt engineering',
    'system prompt':            'prompt engineering',
    'few-shot':                 'prompt engineering',
    'zero-shot':                'prompt engineering',
    'prompt design':            'prompt engineering',

    # mlops 
    'experiment track':         'mlops',
    'model monitor':            'mlops',
    'model deploy':             'mlops',
    'model registry':           'mlops',
    'ml ops':                   'mlops',

    # kubernetes   
    'k8s':                      'kubernetes',
    'helm':                     'kubernetes',
    'eks':                      'kubernetes',
    'aks':                      'kubernetes',
    'gke':                      'kubernetes',

    # sql 
    'postgresql':               'sql',
    'postgres':                 'sql',
    'pgvector':                 'sql',
    'snowflake':                'sql',
    'databricks':               'sql',
    'mysql':                    'sql',
    'nosql':                    'sql',

    # vectordb 
    'azure ai search':          'vectordb',
    'vector database':          'vectordb',
    'vector store':             'vectordb',
    'embedding store':          'vectordb',
    'pinecone':                 'vectordb',
    'chromadb':                 'vectordb',
    'weaviate':                 'vectordb',
    'qdrant':                   'vectordb',
    'faiss':                    'vectordb',
    'milvus':                   'vectordb',

    # nlp 
    'natural language':         'nlp',
    'named entity':             'nlp',
    'text classification':      'nlp',

    # aws/azure 
    'amazon web services':      'aws/azure',
    'azure openai':             'aws/azure',
    'google cloud':             'aws/azure',
    'ai foundry':               'aws/azure',
    'sagemaker':                'aws/azure',
    'bedrock':                  'aws/azure',
    'gcp':                      'aws/azure',
    'lambda':                   'aws/azure',
    'ec2':                      'aws/azure',
    's3':                       'aws/azure',
}