#!/usr/bin/env python3
"""
Generate secure production environment variables
"""
import secrets
import string

def generate_secret(length=32):
    """Generate a secure random string with letters, digits, and safe symbols"""
    alphabet = string.ascii_letters + string.digits + '-_'
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def generate_hex_key(length=32):
    """Generate a secure hex key"""
    return secrets.token_hex(length)

def generate_alphanumeric(length=16):
    """Generate alphanumeric password"""
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

if __name__ == "__main__":
    print("# Secure Production Environment Variables")
    print("# Copy these values to your .env.production file")
    print()
    print(f"SECRET_KEY={generate_secret(64)}")
    print(f"ENCRYPTION_KEY={generate_hex_key(16)}")  # 32 characters hex
    print(f"JWT_SECRET_KEY={generate_secret(64)}")
    print(f"POSTGRES_PASSWORD={generate_alphanumeric(24)}")
    print(f"REDIS_PASSWORD={generate_alphanumeric(24)}")
    print(f"GRAFANA_ADMIN_PASSWORD={generate_alphanumeric(16)}")
    print(f"ELASTICSEARCH_PASSWORD={generate_alphanumeric(16)}")
    print()
    print("# Additional secure values")
    print(f"CSRF_SECRET_KEY={generate_secret(32)}")
    print(f"SESSION_SECRET={generate_secret(32)}")