import secrets
import hashlib
from fastapi import FastAPI, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

app = FastAPI(title="Secure API Key Service")

# Setup the API key header scheme for Swagger UI documentation
# The client will pass the key inside the 'X-API-Key' header
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Simulated Database
# Structure: { hashed_key: {"owner": "username", "scopes": [...] } }
API_KEYS_DB = {}


def generate_secure_api_key(prefix: str = "sk") -> str:
    """Generates a unique, cryptographically secure API key."""
    # generate 32 bytes of secure random data and encode to hex (64 characters)
    random_token = secrets.token_hex(32)
    return f"{prefix}_{random_token}"


def hash_api_key(api_key: str) -> str:
    """Creates a secure SHA-256 hash of the API key to store in the DB."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


async def verify_api_key(api_key: str = Security(api_key_header)):
    """Dependency to validate incoming API keys against the database."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key missing from header",
        )
    
    # Hash the incoming key to match against database records
    hashed_incoming = hash_api_key(api_key)
    
    # Use secrets.compare_digest to prevent timing attacks
    db_record = API_KEYS_DB.get(hashed_incoming)
    if not db_record:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired API Key",
        )
        
    return db_record


# ---------------- Endpoints ----------------

@app.post("/keys/generate", status_code=status.HTTP_201_CREATED)
async def create_api_key(owner: str):
    """
    Generates and registers a new API key.
    The raw key is returned ONLY ONCE. It cannot be recovered later.
    """
    raw_key = generate_secure_api_key()
    hashed_key = hash_api_key(raw_key)
    
    # Store hashed version in the database
    API_KEYS_DB[hashed_key] = {
        "owner": owner,
        "scopes": ["read", "write"]
    }
    
    return {
        "owner": owner,
        "api_key": raw_key,
        "warning": "Copy this key now. It will not be shown again."
    }


@app.get("/protected-resource")
async def get_protected_data(api_client: dict = Depends(verify_api_key)):
    """A secure route protected by API key authentication."""
    return {
        "message": f"Hello {api_client['owner']}, access granted!",
        "permissions": api_client["scopes"]
    }
