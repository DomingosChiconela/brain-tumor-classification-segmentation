# Brain Tumor Classification and Segmentation Using Deep Learning

Este projecto tem como objectivo desenvolver um sistema para **classificação e segmentação de tumores cerebrais a partir de imagens de Ressonância Magnética (MRI), utilizando técnicas de Deep Learning**.

O projecto está a ser desenvolvido no âmbito de um trabalho académico e tem como finalidade investigar diferentes arquitecturas de Deep Learning aplicadas à análise automática de imagens MRI do cérebro.

---

## 📋 Índice

* [Sobre o Projecto](#sobre-o-projecto)
* [Tecnologias Utilizadas](#tecnologias-utilizadas)
* [Configuração do Projecto](#configuração-do-projecto)

  * [1. Clonar o Repositório](#1-clonar-o-repositório)
  * [2. Abrir o Projecto](#2-abrir-o-projecto)
  * [3. Criar o Ambiente Virtual](#3-criar-o-ambiente-virtual)
  * [4. Activar o Ambiente Virtual](#4-activar-o-ambiente-virtual)
  * [5. Instalar as Dependências](#5-instalar-as-dependências)
  * [6. Criar o Dataset](#6-criar-o-dataset)
* [Dataset](#dataset)
* [Kernel dos Jupyter Notebooks](#kernel-dos-jupyter-notebooks)
* [Git Flow e Colaboração](#git-flow-e-colaboração)
* [Branches](#branches)

---

## Sobre o Projecto

O sistema está a ser desenvolvido para explorar técnicas de **Deep Learning** aplicadas a duas tarefas principais:

* **Classificação de Tumores Cerebrais** — identificar a classe ou tipo de tumor presente numa imagem MRI.
* **Segmentação de Tumores Cerebrais** — identificar e delimitar a região correspondente ao tumor numa imagem MRI.

O projecto contém scripts, modelos, notebooks e funções utilitárias necessários para a preparação dos dados, desenvolvimento dos modelos e realização das experiências.

---

## Tecnologias Utilizadas

As principais tecnologias e bibliotecas utilizadas no projecto incluem:

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Jupyter Notebook
* Git
* GitHub

---

# Configuração do Projecto

Para executar o projecto localmente, siga os passos apresentados abaixo.

## 1. Clonar o Repositório

Comece por clonar o repositório utilizando o Git:

```bash
git clone https://github.com/DomingosChiconela/brain-tumor-classification-segmentation.git
```

Depois de clonar o repositório, entre na pasta do projecto:

```bash
cd brain-tumor-classification-segmentation
```

---

## 2. Abrir o Projecto

Caso esteja a utilizar o **Visual Studio Code**, pode abrir o projecto através do comando:

```bash
code .
```

Também pode abrir o Visual Studio Code manualmente e seleccionar a pasta do projecto.

---

## 3. Criar o Ambiente Virtual

É recomendado utilizar um ambiente virtual para manter as dependências deste projecto isoladas das restantes instalações de Python existentes no computador.

Para criar o ambiente virtual, execute:

```bash
python -m venv .venv
```

Este comando irá criar uma pasta chamada `.venv` na raiz do projecto.

---

## 4. Activar o Ambiente Virtual

### Windows — PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows — Command Prompt

```cmd
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Depois da activação, deverá aparecer `(.venv)` no início da linha de comandos:

```text
(.venv)
```

Isto indica que o ambiente virtual está activo.

---

## 5. Instalar as Dependências

Com o ambiente virtual activo, instale todas as dependências necessárias para executar o projecto através do ficheiro `requirements.txt`:

```bash
pip install -r requirements.txt
```

É importante garantir que o ambiente virtual está activo antes de executar este comando, de modo a que as dependências sejam instaladas no ambiente `.venv`.

---

## 6. Criar o Dataset

Depois de instalar todas as dependências, é necessário criar o dataset processado utilizado pelo projecto.

Para isso, execute o ficheiro `build-dataset.py` localizado na raiz do projecto:

```bash
python build-dataset.py
```

Em sistemas onde o comando `python` corresponde a outra versão do Python, pode utilizar:

```bash
python3 build-dataset.py
```

O script irá processar os dados e criar o seguinte ficheiro:

```text
brain_tumor_dataset.npz
```

Este ficheiro contém os dados processados que serão utilizados durante o treino e avaliação dos modelos.

---

# Dataset

O dataset processado utilizado pelo projecto é armazenado no ficheiro:

```text
brain_tumor_dataset.npz
```

### ⚠️ Nota importante

O ficheiro `brain_tumor_dataset.npz` **não é versionado pelo Git**, uma vez que o seu tamanho excede o limite permitido pelo GitHub para ficheiros individuais.

Por esse motivo, depois de clonar o repositório, cada utilizador deverá gerar o dataset localmente através do script:

```bash
python build-dataset.py
```

O ficheiro `brain_tumor_dataset.npz` encontra-se igualmente incluído no `.gitignore` para impedir que seja acidentalmente enviado para o repositório.

---

# Kernel dos Jupyter Notebooks

Algumas partes do projecto são desenvolvidas através de **Jupyter Notebooks (`.ipynb`)**.

Ao executar os notebooks, pode ocorrer um erro semelhante a:

```text
ModuleNotFoundError: No module named 'utils'
```

ou:

```text
ModuleNotFoundError: No module named 'numpy'
```

Quando isso acontece, uma das possíveis causas é o facto de o Jupyter Notebook estar a utilizar um **Python diferente daquele existente no ambiente virtual `.venv`**.

Para verificar qual é o interpretador Python que está a ser utilizado pelo notebook, execute:

```python
import sys

print(sys.executable)
```

O resultado deverá apontar para o Python existente dentro do ambiente virtual do projecto.

Por exemplo, no Windows:

```text
E:\brain-tumor-classification-segmentation\.venv\Scripts\python.exe
```

Se o caminho apresentado não corresponder ao ambiente `.venv` do projecto, deverá seleccionar o interpretador/kernel correcto no Visual Studio Code.


# Git Flow e Colaboração

O projecto utiliza uma estratégia simples de organização através de branches.

Actualmente existem **duas branches principais**:

```text
main
dev
```

Cada uma possui uma finalidade específica.

---

## `main`

A branch `main` representa a versão **estável e validada do projecto**.

O código presente nesta branch deve corresponder a uma versão que foi previamente desenvolvida, testada e considerada suficientemente estável.

```text
main = Código estável / Produção
```

Por esse motivo, alterações experimentais ou funcionalidades ainda não testadas não devem ser directamente desenvolvidas na `main`.

---

## `dev`

A branch `dev` representa o **ambiente de desenvolvimento**.

É nesta branch que devem ser integradas as novas funcionalidades, correcções, melhorias e experiências antes de serem consideradas estáveis.

```text
dev = Desenvolvimento
```

A `dev` funciona, portanto, como uma camada entre o desenvolvimento das funcionalidades e a versão estável existente na `main`.

---

# Fluxo de Desenvolvimento

O fluxo recomendado para trabalhar no projecto é:

```text
feature/*
     │
     ▼
    dev
     │
     │ Testes e validação
     ▼
   main
```

Por exemplo, ao desenvolver uma nova arquitectura de classificação:

```text
feature/alexnet-training
          │
          ▼
         dev
          │
          │ Testes / Avaliação
          ▼
         main
```

---

## Trabalhar na Branch `dev`

Antes de começar o desenvolvimento, actualize a branch `dev`:

```bash
git switch dev
```

Depois:

```bash
git pull origin dev
```

---

## Criar uma Feature Branch

Para desenvolver uma nova funcionalidade, recomenda-se criar uma branch específica a partir da `dev`.

Por exemplo:

```bash
git switch -c feature/model-training
```

Depois de concluir o desenvolvimento, adicione as alterações:

```bash
git add .
```

Crie um commit semântico:

```bash
git commit -m "feat: add model training pipeline"
```

Envie a branch para o GitHub:

```bash
git push origin feature/model-training
```

Posteriormente, a `feature` poderá ser integrada na branch `dev` através de um Pull Request.

Depois de a funcionalidade ser testada e validada na `dev`, as alterações poderão ser integradas na `main`.

---

# Estrutura Geral das Branches

```text
                    ┌──────────────┐
                    │   feature/*  │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │     dev      │
                    │ Development  │
                    └──────┬───────┘
                           │
                     Testes / Validação
                           │
                           ▼
                    ┌──────────────┐
                    │     main     │
                    │   Estável    │
                    └──────────────┘
```

Esta abordagem permite manter a branch `main` estável, enquanto o desenvolvimento e a experimentação são realizados na `dev` e nas respectivas feature branches.

---

# Notas Importantes

* Active sempre o ambiente virtual `.venv` antes de instalar dependências ou executar o projecto.
* Instale as dependências através do `requirements.txt`.
* Execute `build-dataset.py` para gerar o dataset processado.
* O ficheiro `brain_tumor_dataset.npz` não é versionado devido ao seu tamanho.
* Em caso de erros `ModuleNotFoundError` nos notebooks, confirme o Python utilizado através de `sys.executable`.
* Certifique-se de que os Jupyter Notebooks estão a utilizar o kernel do ambiente `.venv`.
* Evite desenvolver directamente na branch `main`.
* Utilize a `dev` para desenvolvimento e testes.
* Apenas código validado e estável deve ser integrado na `main`.
* Utilize commits semânticos para manter um histórico de alterações organizado.

---

# Licença

Este projecto foi desenvolvido para fins académicos e de investigação.
