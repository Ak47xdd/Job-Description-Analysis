"""
sym_map.py - All synonyms for the TF-IDF Vectorizer
"""

TAXONOMY_VERSION = "v3-canonical-buckets"

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
    'large language models':    'llms',
    'large language model':     'llms',
    'llm':                       'llms',
    'ai tools':                 'ai',

    # backend / full-stack taxonomy
    'backend services':         'backend',
    'backend engineering':      'backend',
    'full stack':                'full-stack',

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
    # Keep the full phrase intact for both SBERT semantic context and
    # deterministic lexical recovery. The previous map only handled
    # "natural language", so an explicit "Natural Language Processing (NLP)"
    # requirement could remain invisible to the output label.
    'natural language processing': 'nlp',
    'natural-language processing': 'nlp',
    'natural language':         'nlp',
    'nlp':                      'nlp',
    'named entity':             'nlp',
    'text classification':      'nlp',

    # aws/azure -- cloud-provider aliases share one canonical bucket
    'microsoft azure':          'aws/azure',
    'amazon web services':      'aws/azure',
    'azure openai':             'aws/azure',
    'google cloud platform':    'aws/azure',
    'google cloud':             'aws/azure',
    'ai foundry':               'aws/azure',
    'sagemaker':                'aws/azure',
    'bedrock':                  'aws/azure',
    'gcp':                      'aws/azure',
    'lambda':                   'aws/azure',
    'ec2':                      'aws/azure',
    's3':                       'aws/azure',
    
    'node js':                  'node.js',
    'nodejs':                   'node.js',
    'golang':                   'go',
    'rustlang':                 'rust',
    'bash scripting':           'bash',
    'shell scripting':          'bash',
    'ts':                       'typescript',
    'mongo':                    'mongodb',
    'mongo db':                 'mongodb',
    'rest api':                 'rest',
    'restful':                  'rest',
    'oauth':                    'oauth 2.0',
    'oauth2':                   'oauth 2.0',
}

# Canonical label buckets used by data preparation and inference.
# Aliases map into labels that already exist in the v2 vocabulary. The goal is
# to avoid spending separate output neurons on the same underlying skill.
CANONICAL_LABEL_MAP = {
    # AI / GenAI / LLMs
    "ai tools": "ai",
    "artificial intelligence": "ai",
    "generative ai": "genai",
    "gen ai": "genai",
    "frontier model": "genai",
    "foundation model": "genai",
    "gemini": "genai",
    "claude": "genai",
    "gpt": "genai",
    "large language models": "llms",
    "large language model": "llms",
    "llm": "llms",
    # Agents / agent frameworks
    "ai agents": "agents",
    "multi-agent": "agents",
    "multi agent": "agents",
    "agentic": "agents",
    "autogen": "agents",
    "crewai": "agents",
    "langchain": "agents",
    "langgraph": "agents",
    "llamaindex": "agents",
    "copilot": "agents",
    # Cloud
    "aws": "aws/azure",
    "azure": "aws/azure",
    "microsoft azure": "aws/azure",
    "amazon web services": "aws/azure",
    "gcp": "aws/azure",
    "google cloud": "aws/azure",
    "google cloud platform": "aws/azure",
    "azure openai": "aws/azure",
    "ai foundry": "aws/azure",
    "sagemaker": "aws/azure",
    "bedrock": "aws/azure",
    "lambda": "aws/azure",
    "ec2": "aws/azure",
    "s3": "aws/azure",
    "cloud computing": "cloud",
    "cloud environments": "cloud",
    "cloud platforms": "cloud",
    "public cloud": "cloud",
    "cloud infrastructure": "cloud",
    "cloud-native": "cloud",
    # Data / analytics
    "data analysis": "data analytics",
    "analytics": "data analytics",
    "reporting": "data analytics",
    "dashboards": "data analytics",
    "data processing": "data engineering",
    "data pipelines": "data engineering",
    "ingestion pipelines": "data engineering",
    "etl": "data engineering",
    "elt": "data engineering",
    # Data platforms / SQL ecosystem
    "postgres": "sql",
    "postgresql": "sql",
    "mysql": "sql",
    "oracle": "sql",
    "snowflake": "sql",
    "databricks": "sql",
    "trino": "sql",
    "nosql": "databases",
    "mongo": "mongodb",
    "mongo db": "mongodb",
    # Spark ecosystem
    "apache spark": "spark",
    # Kubernetes ecosystem
    "k8s": "kubernetes",
    "helm": "kubernetes",
    "eks": "kubernetes",
    "aks": "kubernetes",
    "gke": "kubernetes",
    # CI/CD ecosystem
    "continuous integration": "ci/cd",
    "continuous deployment": "ci/cd",
    "continuous delivery": "ci/cd",
    "cicd": "ci/cd",
    "github actions": "ci/cd",
    "gitlab ci": "ci/cd",
    "jenkins": "ci/cd",
    "azure devops": "ci/cd",
    "gitops": "ci/cd",
    # Version-control aliases
    "version control": "git",
    "source control": "git",
    "gitlab": "git",
    "github": "git",
    "bitbucket": "git",
    # Containers
    "containerization": "docker",
    "containerized": "docker",
    "containerize": "docker",
    "containeriz": "docker",
    # API / service terminology
    "api": "apis",
    "rest": "apis",
    "rest api": "apis",
    "rest apis": "apis",
    "restful": "apis",
    "web services": "apis",
    "cloud apis": "apis",
    "cloud provider apis": "apis",
    "sdks": "apis",
    # Frontend framework aliases
    "react.js": "react",
    "react native": "react",
    "node": "node.js",
    "nodejs": "node.js",
    "node js": "node.js",
    "nextjs": "next.js",
    "vue.js": "vue",
    # Power BI spelling variants
    "powerbi": "power bi",
    # Language spelling variants
    "golang": "go",
    "rustlang": "rust",
    "ts": "typescript",
    "csharp": "c#",
    "dotnet": ".net",
    # Shell aliases
    "shell scripting": "bash",
    "bash scripting": "bash",
    "powershell scripting": "powershell",
    # ML aliases
    "machine-learning": "machine learning",
    "ml": "machine learning",
    "deep neural networks": "deep learning",
    "tensorflow/pytorch": "deep learning",
    # Vector DB aliases
    "vector databases": "vectordb",
    "vector database": "vectordb",
    "vector store": "vectordb",
    "embedding store": "vectordb",
    # Backend / architecture aliases
    "backend services": "backend",
    "backend engineering": "backend",
    "full stack": "full-stack",
    "micro-service": "microservices",
    "micro-services": "microservices",
    # Security aliases
    "security telemetry": "telemetry",
    # Observability aliases
    "monitoring": "observability",
    "audit logs": "observability",
}



def canonicalize_skill_text(text: str) -> str:
    """Normalize aliases and canonical skill buckets in free-form text."""
    import re

    text = str(text or "").lower()
    synonym_items = sorted(SYNONYM_MAP.items(), key=lambda x: -len(x[0]))
    if synonym_items:
        pattern = re.compile(
            r"(?<![\\w])(?:"
            + "|".join(re.escape(phrase) for phrase, _ in synonym_items)
            + r")(?![\\w])"
        )
        lookup = dict(synonym_items)
        text = pattern.sub(lambda match: lookup[match.group(0)], text)

    bucket_items = sorted(CANONICAL_LABEL_MAP.items(), key=lambda x: -len(x[0]))
    if bucket_items:
        pattern = re.compile(
            r"(?<![\\w])(?:"
            + "|".join(re.escape(phrase) for phrase, _ in bucket_items)
            + r")(?![\\w])"
        )
        lookup = dict(bucket_items)
        text = pattern.sub(lambda match: lookup[match.group(0)], text)
    return text
