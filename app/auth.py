from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa

from .config import settings


def _load_private_key():
    if settings.kalshi_private_key_pem:
        raw = settings.kalshi_private_key_pem.encode()
    elif settings.kalshi_private_key_path:
        raw = Path(settings.kalshi_private_key_path).expanduser().read_bytes()
    else:
        raise RuntimeError('Kalshi private key is not configured')
    return serialization.load_pem_private_key(raw, password=None)


def sign_text(text: str) -> str:
    key = _load_private_key()
    message = text.encode('utf-8')
    if isinstance(key, ed25519.Ed25519PrivateKey):
        signature = key.sign(message)
    elif isinstance(key, rsa.RSAPrivateKey):
        signature = key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
    else:
        raise TypeError(f'Unsupported Kalshi private-key type: {type(key).__name__}')
    return base64.b64encode(signature).decode('ascii')


def auth_headers(method: str, path: str) -> dict[str, str]:
    timestamp = str(int(time.time() * 1000))
    sign_path = path.split('?', 1)[0]
    message = f'{timestamp}{method.upper()}{sign_path}'
    return {
        'KALSHI-ACCESS-KEY': settings.kalshi_key_id,
        'KALSHI-ACCESS-TIMESTAMP': timestamp,
        'KALSHI-ACCESS-SIGNATURE': sign_text(message),
    }
