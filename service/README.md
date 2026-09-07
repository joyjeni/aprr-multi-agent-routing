# APRR HTTP microservice

```bash
python service/app.py 43132
curl -X POST http://127.0.0.1:43132/route -H 'Content-Type: application/json' \
  -d '{"query":"compare weather APIs then chart forecast"}'
```

Wraps `APRRRouter` (Eq. 2–4). Default ε=0 for a reproducible hop path.
