sudo SYSTEMD_EDITOR=vim systemctl edit ollama.service
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11435"
sudo systemctl daemon-reload
sudo systemctl restart ollama