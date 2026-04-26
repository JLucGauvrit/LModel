#!/bin/sh
set -e

CERT_DIR=/etc/nginx/ssl
mkdir -p "$CERT_DIR"

# Toujours localhost + loopback
SANS="DNS:localhost,IP:127.0.0.1"

# Ajouter les IPs LAN passées via HOST_IPS (virgule-séparées)
if [ -n "$HOST_IPS" ]; then
    for ip in $(echo "$HOST_IPS" | tr ',' ' '); do
        SANS="$SANS,IP:$ip"
        echo "[nginx] SAN ajouté : $ip"
    done
fi

echo "[nginx] Génération du certificat TLS (SANs : $SANS)"
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout "$CERT_DIR/key.pem" \
    -out    "$CERT_DIR/cert.pem" \
    -subj   "/CN=localhost/O=WorldModelFPGA" \
    -addext "subjectAltName=$SANS" \
    2>/dev/null

exec nginx -g "daemon off;"
