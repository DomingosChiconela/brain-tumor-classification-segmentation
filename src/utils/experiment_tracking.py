"""
experiment_tracking.py

Responsável por tudo o que tem a ver com RASTREABILIDADE de execuções de
treino: gerar um ID único por execução, guardar os hiperparâmetros e
métricas usados em cada uma, manter um log central de todas as
execuções, e manter um "best.keras" que é sempre o melhor modelo de
sempre para aquela arquitectura.

Segue a convenção comum na comunidade de ML (ex: Cookiecutter Data
Science, MLflow) de SEPARAR:

- checkpoints/  -> artefactos pesados e binários (.keras). Vai para o
                   .gitignore; nunca se versiona pesos de modelo no Git.
- experiments/  -> metadados leves em texto (JSON): hiperparâmetros e
                   métricas de cada execução. Este SIM deve ser
                   versionado no Git, porque documenta a história do
                   projecto a um custo quase nulo (são só KBs de texto).

Esta lógica fica separada do train.py de propósito: train.py sabe
"como treinar", este ficheiro sabe "como não perder o que já foi
treinado".
"""

import json
import shutil
from datetime import datetime
from pathlib import Path


def make_run_id() -> str:
    """Gera um identificador único baseado no timestamp actual."""
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def get_checkpoint_run_dir(checkpoint_dir: Path, model_name: str, run_id: str) -> Path:
    """Pasta onde vai o peso (.keras) desta execução: checkpoints/<task>/<model>/<run_id>/"""
    run_dir = checkpoint_dir / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(experiments_dir: Path, model_name: str, run_id: str, config: dict) -> Path:
    """
    Guarda os hiperparâmetros e métricas desta execução em
    experiments/<task>/<model_name>/<run_id>.json — ficheiro leve,
    pensado para ser versionado no Git.
    """
    run_configs_dir = experiments_dir / model_name
    run_configs_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_configs_dir / f"{run_id}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


def append_to_runs_log(experiments_dir: Path, config: dict) -> Path:
    """
    Acrescenta esta execução ao log central (experiments/<task>/runs_log.json),
    que junta o histórico de TODOS os modelos e execuções — útil para
    comparar tudo de uma vez num notebook, sem abrir cada ficheiro individual.
    """
    experiments_dir.mkdir(parents=True, exist_ok=True)
    log_path = experiments_dir / "runs_log.json"
    log = []
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)

    log.append(config)

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return log_path


def update_best_if_needed(checkpoint_dir: Path, experiments_dir: Path, model_name: str,
                            run_checkpoint_path: Path, current_val_accuracy: float) -> bool:
    """
    Compara esta execução com TODAS as execuções anteriores do mesmo
    modelo (via runs_log.json em experiments/) e, se for a melhor de
    sempre, copia o checkpoint para checkpoints/<task>/<model_name>/best.keras.

    Returns
    -------
    bool
        True se este passou a ser o melhor modelo de sempre para este model_name.
    """
    log_path = experiments_dir / "runs_log.json"
    previous_best = -1.0
    if log_path.exists():
        with open(log_path) as f:
            log = json.load(f)
        previous_runs = [r for r in log if r["model_name"] == model_name]
        if previous_runs:
            previous_best = max(r["best_val_accuracy"] for r in previous_runs)

    if current_val_accuracy >= previous_best:
        best_path = checkpoint_dir / model_name / "best.keras"
        shutil.copy(run_checkpoint_path, best_path)
        return True

    return False