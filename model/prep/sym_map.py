"""
sym_map.py - All synonyms for the TF-IDF Vectorizer
"""

TAXONOMY_VERSION = "v9-clean-umbrella-labels"

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
    'infrastructure-as-code':   'platform engineering',
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
    'ai tools':                 'ai/ml',

    # Consolidated parent buckets: normalize framework variants to one label.
    '.net core':                '.net',
    'asp.net core':              '.net',
    'asp.net':                   '.net',
    'asp.net mvc':               '.net',
    'asp.net web api':            '.net',
    '.net framework':            '.net',
    'dotnet core':               '.net',
    'artificial intelligence':  'ai/ml',
    'machine learning':         'ai/ml',
    'machine-learning':         'ai/ml',
    'ai/ml':                    'ai/ml',
    'ml':                       'ai/ml',

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
    # AI/ML parent bucket pools sparse synonymous labels together.
    "ai": "ai/ml",
    "ai tools": "ai/ml",
    "ai/ml": "ai/ml",
    "artificial intelligence": "ai/ml",
    "machine learning": "ai/ml",
    "machine-learning": "ai/ml",
    "ml": "ai/ml",
    # .NET ecosystem variants share one canonical parent label.
    ".net core": ".net",
    "asp.net core": ".net",
    "asp.net": ".net",
    "asp.net mvc": ".net",
    "asp.net web api": ".net",
    ".net framework": ".net",
    "dotnet core": ".net",
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
    # ML-specific child concepts remain separate; their parent is ai/ml.
    "machine-learning": "ai/ml",
    "ml": "ai/ml",
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

    # Excel aliases and proficiency wording.
    "advanced excel": "excel",
    "microsoft excel": "excel",
    "ms excel": "excel",

    # SEO specializations and wording variants share one broad SEO target.
    # SEO tools/products are intentionally kept separate where they identify
    # a concrete product rather than the general SEO competency.
    "search engine optimization": "seo",
    "on-page seo": "seo",
    "off-page seo": "seo",
    "technical seo": "seo",
    "local seo": "seo",
    "international seo": "seo",
    "seo fundamentals": "seo",
    "seo strategy": "seo",
    "seo best practices implementation": "seo",
    "seo performance tracking": "seo",
    "seo reporting": "seo",
    "seo reporting tools": "seo",
    "seo tools": "seo",
    "seo audits": "seo",
    "technical seo audits": "seo",
    "ai-powered seo": "seo",
    "ai seo tools": "seo",

    # Analytics product naming variants.
    "ga4": "google analytics",
    "google analytics (ga4)": "google analytics",
    "google analytics 4": "google analytics",
    "search console": "google search console",

    # Office suite naming variants. Individual Word/PowerPoint skills remain
    # separate from the broader Microsoft Office competency.
    "ms office": "microsoft office",
    "microsoft office suite": "microsoft office",
    "office 365": "microsoft 365",
    "ms word": "microsoft word",

    # Google productivity-suite naming variants.
    "g suite": "google workspace",
    "gsuite": "google workspace",
    "google suite": "google workspace",

    # CRM naming variants.
    "customer relationship management": "crm",
    "crm systems": "crm",
    "crm tools": "crm",

    # Identity and access management wording variants.
    "identity & access management": "identity and access management",
    "iam": "identity and access management",

    # Audit/auditing are grammatical variants; retain domain-specific audit
    # labels such as IT audit, internal audit and financial auditing.
    "auditing": "audit",
    "auditing methodologies": "audit methodologies",
    "audit methodologies": "audit methodologies",
    "auditing procedures": "audit procedures",
    "internal auditing": "internal audit",
    "it auditing": "it audit",

    # Communication wording variants.
    "communication skills": "communication",

    # Common spelling and formatting variants.
    "corel draw": "coreldraw",
    "full-stack development": "full-stack",
    "full stack development": "full-stack",
    "nodejs": "node.js",
    "node js": "node.js",

    # Aggressive office-productivity consolidation. Individual application
    # distinctions are intentionally pooled into one competency-family label.
    "microsoft office": "office productivity",
    "microsoft office suite": "office productivity",
    "ms office": "office productivity",
    "microsoft 365": "office productivity",
    "office 365": "office productivity",
    "excel": "office productivity",
    "advanced excel": "office productivity",
    "microsoft excel": "office productivity",
    "ms excel": "office productivity",
    "word": "office productivity",
    "microsoft word": "office productivity",
    "ms word": "office productivity",
    "powerpoint": "office productivity",
    "microsoft powerpoint": "office productivity",
    "sheets": "office productivity",
    "google sheets": "office productivity",
    "google docs": "office productivity",
    "google slides": "office productivity",
    "google workspace": "office productivity",
    "g suite": "office productivity",
    "gsuite": "office productivity",
    "google suite": "office productivity",
    "spreadsheets": "office productivity",
    "outlook": "office productivity",
    "visio": "office productivity",

    # SEO competency family: specializations and commonly named SEO tools.
    "ahrefs": "seo",
    "semrush": "seo",
    "moz": "seo",
    "moz pro": "seo",
    "screaming frog": "seo",
    "screaming frog seo spider": "seo",
    "yoast seo": "seo",
    "google search console": "seo",
    "search console": "seo",
    "on-page seo": "seo",
    "off-page seo": "seo",
    "technical seo": "seo",
    "local seo": "seo",
    "international seo": "seo",
    "seo audits": "seo",
    "technical seo audits": "seo",
    "seo reporting": "seo",
    "seo tools": "seo",

    # Web analytics spelling variants.
    "website analytics": "web analytics",
    "digital analytics": "web analytics",
    "google analytics 4": "google analytics",
    "google analytics (ga4)": "google analytics",
    "ga4": "google analytics",

    # Product-name and spelling aliases.
    "adobe premiere": "adobe premiere pro",
    "after effects": "adobe after effects",
    "mssql": "sql server",
    "microsoft sql server": "sql server",
    "t-sql": "sql",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "mongo db": "mongodb",
    "nodejs": "node.js",
    "node js": "node.js",
    "react.js": "react",
    "vue.js": "vue",
    "nextjs": "next.js",
    "argocd": "argo cd",

    # Software/product aliases: preserve the best-supported canonical label.
    "photoshop": "adobe photoshop",

    # REST Assured spelling variants.
    "restassured": "rest assured",

    # STAAD.Pro product-name variants.
    "staad pro": "staad",
    "staad.pro": "staad",
    "staadpro": "staad",

    # UI/UX competency wording variants.
    "ui/ux design": "ui/ux",

    # Remove noisy fragments by pooling them into existing parent competencies.
    # These aliases are applied to the source labels before support counting.
    "porstgresql": "sql",
    "postgresql": "sql",
    "postgres": "sql",
    "pgvector": "sql",
    "database": "databases",
    "db": "databases",
    "models": "ai/ml",
    "model": "ai/ml",
    "deployment": "devops",
    "deployments": "devops",
    "model deployment": "mlops",
    "model deployment pipelines": "mlops",
    "platform": "platform engineering",
    "platforms": "platform engineering",
    "fastapi": "backend",
    "flask": "backend",
    "django": "backend",
    "express": "backend",
    "express.js": "backend",
    "spring boot": "backend",
    "search engine": "ai search",
    "search engines": "ai search",
    "information retrieval": "ai search",

    # Clean-up of noisy / overly granular labels. Pool these into existing
    # parent competencies so they do not consume separate classifier outputs.
    # Infrastructure-as-code and automation belong under Platform Engineering.
    "infrastructure-as-code": "platform engineering",
    "infrastructure as code": "platform engineering",
    "infrastructure automation": "platform engineering",
    "iac": "platform engineering",

    # AI search / retrieval / vector-search terminology shares the existing
    # Vector DB umbrella label.
    "ai search": "vectordb",
    "vector search": "vectordb",
    "vector retrieval": "vectordb",
    "information retrieval": "vectordb",
    "search engine": "vectordb",
    "search engines": "vectordb",
    "azure ai search": "vectordb",
    "vector database": "vectordb",
    "vector databases": "vectordb",
    "vector store": "vectordb",
    "embedding store": "vectordb",

    # Code review is a software-engineering practice, not a standalone skill.
    "code review": "software engineering",
    "code reviews": "software engineering",
    "code quality": "software engineering",

    # Security specializations are pooled into the existing Security label.
    "ai security": "security",
    "api security": "security",
    "application security": "security",
    "cloud security": "security",
    "cloud security tools": "security",
    "container security": "security",
    "cybersecurity": "security",
    "data security": "security",
    "endpoint security": "security",
    "information security": "security",
    "mobile security": "security",
    "network security": "security",
    "security analysis": "security",
    "security architecture": "security",
    "security assessments": "security",
    "security automation": "security",
    "security best practices": "security",
    "security controls": "security",
    "security engineering": "security",
    "security frameworks": "security",
    "security incident response": "security",
    "security monitoring": "security",
    "security operations": "security",
    "security platform": "security",
    "security policies": "security",
    "security testing": "security",
    "security tooling": "security",

    # CI/CD and DevOps variants use the existing CI/CD / DevOps labels.
    "ci/cd pipelines": "ci/cd",
    "github pipelines": "ci/cd",
    "devops monitoring": "devops",

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
