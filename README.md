# Projeto de Câmera IA

Este projeto demonstra como usar Python e OpenCV para identificar pessoas em uma sala e contabilizar o tempo que cada uma permaneceu presente. É um exemplo simples de software de gerenciamento de pessoal.

## Dependências

- Python 3.11 ou superior
- [opencv-python](https://pypi.org/project/opencv-python/)

Instale as dependências executando:

```bash
pip install opencv-python
```

## Execução

Para iniciar a contagem, execute:

```bash
python people_tracker.py
```

A aplicação abrirá a câmera padrão do sistema, detectará pessoas usando `HOGDescriptor` do OpenCV e exibirá o vídeo com as marcações na tela. Ao encerrar (tecla `q`), será mostrado o tempo total que cada pessoa ficou na cena.
