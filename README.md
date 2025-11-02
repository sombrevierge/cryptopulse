
# CryptoPulse v3

Интерактивная витрина по крипте и DeFi: свечи на Plotly, EMA/RSI/волатильность, DefiLlama-пулы, скоринг и health проектов.

## Локальный запуск
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Деплой в Streamlit Community Cloud (рекомендуется)
1. Создайте публичный репозиторий на GitHub и загрузите сюда файлы проекта.
2. Зайдите на https://share.streamlit.io/ → `New app`.
3. Выберите ваш репозиторий, ветку `main`, файл `streamlit_app.py`.
4. Python version: 3.10+ (по умолчанию ок). Secrets не требуются.
5. Нажмите Deploy.

## Деплой на Render.com
1. Подключите GitHub к https://render.com
2. New → Web Service → выберите репозиторий.
3. Environment: `Python`
4. Build Command:
```bash
pip install -r requirements.txt
```
5. Start Command:
```bash
streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```
6. Free план подойдёт для демо.

## Структура
- `streamlit_app.py` — основной код
- `requirements.txt` — зависимости
- `.streamlit/config.toml` — тёмная тема и настройки сервера

