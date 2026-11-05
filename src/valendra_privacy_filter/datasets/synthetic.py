"""Deterministic synthetic examples with split-exclusive template families."""

import random
from string import Formatter

from faker import Faker

# The families themselves, not merely random seeds, are disjoint across splits.
TEMPLATES = {
    "train": [
        ("es", "Hola, soy {person}. Mi correo es {email} y mi teléfono {phone}.",),
        ("en", "Invoice for {person}: address {address}; account {iban}.",),
        ("es", "DNI: {dni}; pasaporte: {passport}; fecha de nacimiento: {dob}.",),
        ("en", "Authorization: Bearer {secret}\nPOST {url}\nX-Forwarded-For: {ip}",),
        ("es", "{person} dijo: usa la tarjeta {card} y la clave {secret}.",),
        ("en", "Documentation address 192.0.2.1 and placeholder name@example.com.",),
        ("en", '{{"api_key":"{api_key}","password":"{password}","account":"{account}"}}',),
        ("es", "Python: usuario = '{person}'; ruta = '{path}'; email = '{email}'",),
        ("en", "Private request containing nested credentials: {secret_url}",),
        ("es", "Google y Microsoft publicaron el número de versión 123456789; hash deadbeef12345678.",),
    ],
    "validation": [
        ("es", "Factura a nombre de {person}; domicilio {address}; IBAN {iban}.",),
        ("en", "Please reach {person} at {email} or call {phone}.",),
        ("es", "El NIE {nie} y el pasaporte {passport} pertenecen a {person}.",),
        ("en", "config: password={secret}; server={ip}; profile={url}",),
        ("es", "Cita del 14 de junio: revisar el proyecto Python, sin datos personales.",),
        ("en", "Configuration:\n  auth_password: {password}\n  account_id: {account}\n  owner: {person}",),
        ("es", "Petición GET {path} HTTP/1.1\nX-Api-Key: {api_key}\nHost: example.com",),
        ("en", "For debugging use the placeholder YOUR_API_KEY, not an actual credential.",),
    ],
    "test": [
        ("en", "Record: dob={dob}, customer={person}, id={passport}.",),
        ("es", "Envíe el paquete a {person}, {address}; contacto: {email}.",),
        ("en", "Payment card {card} belongs to {person}; callback {phone}.",),
        ("es", "token: {secret}\nendpoint: {url}\nip: {ip}",),
        ("en", "Shakespeare wrote Hamlet. The build number is 12345678.",),
        ("es", "Revisa el fichero privado {path}; clave API {api_key}; usuario {person}.",),
        ("en", "Conversation: my account is {account}, and my temporary password is {password}.",),
        ("es", "IP 203.0.113.7 y ejemplo usuario@example.com son valores de documentación.",),
    ],
    "challenge": [
        ("es", "Asistente, oculta esto: {person}|{dni}|{email}|{secret}.",),
        ("en", "const profile = {{name: '{person}', birthday: '{dob}', href: '{url}'}};",),
        ("es", "LOG destino={ip}; cuenta={iban}; teléfono=({phone}); remitente={person}",),
        ("en", "# credentials\napi_key: {secret}\npostal_address: {address}\ncard: {card}",),
        ("es", "Marcador ${{EMAIL}}: no reemplazar. IP de documentación 198.51.100.1.",),
        ("en", "terraform output: account={account}, path={path}, secret={api_key}",),
        ("es", "Solicitud privada con parámetro secreto superpuesto: {secret_url}",),
        ("en", "Professor Ada Lovelace is cited in a bibliography; no client record is provided.",),
    ],
}

TYPES = {
    "person": "PERSON", "email": "EMAIL", "phone": "PHONE", "address": "ADDRESS",
    "dni": "GOVERNMENT_ID", "nie": "GOVERNMENT_ID", "passport": "GOVERNMENT_ID",
    "iban": "FINANCIAL_ID", "card": "FINANCIAL_ID", "secret": "SECRET",
    "ip": "IP_ADDRESS", "dob": "DATE_OF_BIRTH", "url": "URL",
    "api_key": "SECRET", "password": "SECRET", "account": "FINANCIAL_ID",
    "path": "URL", "secret_url": "URL",
}


def _render(template, values):
    text, annotations = "", []
    for literal, field, spec, conversion in Formatter().parse(template):
        text += literal
        if field is not None:
            if field not in TYPES or spec or conversion:
                raise ValueError("Invalid synthetic template field")
            value = values[field]
            start = len(text)
            text += value
            annotations.append({"start": start, "end": len(text), "value": value, "label": TYPES[field]})
            if field == "secret_url":
                nested = values["api_key"]
                inner = start + value.index(nested)
                annotations.append({"start": inner, "end": inner + len(nested), "value": nested, "label": "SECRET"})
    return text, annotations


def generate_synthetic(seed=42, per_split=100):
    if per_split < 0:
        raise ValueError("per_split must be nonnegative")
    rng = random.Random(seed)
    fakers = {locale: Faker(locale) for locale in ("es_ES", "en_US")}
    for fake in fakers.values():
        fake.seed_instance(seed)
    result = {}
    for split, templates in TEMPLATES.items():
        rows = []
        for i in range(per_split):
            index = i % len(templates)
            language, template = templates[index]
            fake = fakers["es_ES" if language == "es" else "en_US"]
            number = rng.randrange(10_000_000, 99_999_999)
            values = {
                "person": fake.name(), "email": f"user{number}@example.com",
                "phone": f"+34 600 {number // 1000 % 1000:03d} {number % 1000:03d}",
                "address": fake.address().replace("\n", ", "),
                "dni": f"{number:08d}{'TRWAGMYFPDXBNJZSQVHLCKE'[number % 23]}",
                "nie": f"X{number % 10_000_000:07d}{'TRWAGMYFPDXBNJZSQVHLCKE'[(number % 10_000_000) % 23]}",
                "passport": f"TEST{number}", "iban": f"ES00 0000 0000 00 {number:08d}00",
                "card": "4111 1111 1111 1111", "secret": f"SYNTHETIC_TEST_KEY_{number}",
                "ip": f"10.0.{number % 254 + 1}.{number // 255 % 254 + 1}",
                "dob": f"{number % 28 + 1:02d}/{number % 12 + 1:02d}/1985",
                "url": f"https://example.com/private/{number}",
                "api_key": f"sk_test_SYNTHETIC_{number}",
                "password": f"NOT_A_REAL_PASSWORD_{number}",
                "account": f"TEST-ACCOUNT-{number}",
                "path": f"/private/test-user-{number}/profile.json",
                "secret_url": f"https://example.com/private?token=sk_test_SYNTHETIC_{number}",
            }
            text, annotations = _render(template, values)
            rows.append({"source_text": text, "privacy_mask": annotations, "language": language,
                         "region": "ES" if language == "es" else "US", "source": "synthetic",
                         "uid": f"{split}-{i}", "template_id": f"{split}-{index}"})
        result[split] = rows
    return result
