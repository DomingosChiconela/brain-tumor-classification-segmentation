"""
grid_tracking_bridge.py

Traduz os resultados internos do keras_tuner (tuner.oracle.trials) para
a mesma lógica de rastreabilidade já usada em src/utils/experiment_tracking.py,
para que o grid search seja consultável da mesma forma que um treino
normal, sem manter duas fontes de verdade separadas.

Agnóstico ao modelo: recebe model_name apenas como string, para saber
onde escrever. Não precisa de conhecer a arquitectura nem o HyperModel
usado.

Convenção de pastas (paralela à de experiment_tracking.py):

- checkpoints/<task>/grid_search/<model_name>/<grid_id>/
    -> checkpoints internos de cada trial do keras_tuner (pesados,
       .gitignore) + best.keras, o melhor modelo desta grid.
- experiments/<task>/grid_search/<model_name>/<grid_id>/
    -> um .json leve por trial + leaderboard.json (todos os trials
       desta grid, ordenados por val_accuracy desc). Versionável no Git.
"""

import json
from pathlib import Path


def bridge_tuner_results(
    tuner,
    model_name: str,
    grid_id: str,
    checkpoint_dir: Path,
    experiments_dir: Path,
) -> list[dict]:
    """
    Percorre tuner.oracle.trials, escreve um .json por trial, monta o
    leaderboard.json desta grid, e guarda o melhor modelo em best.keras.

    Parameters
    ----------
    tuner : keras_tuner.GridSearch
        Tuner já com tuner.search(...) concluído.
    model_name : str
        Nome do modelo/arquitectura (ex: "alexnet"), usado apenas para
        organizar as pastas de saída.
    grid_id : str
        Identificador único desta sessão de grid search.
    checkpoint_dir : Path
        Pasta raiz de checkpoints da task (ex: checkpoints/classification).
    experiments_dir : Path
        Pasta raiz de experiments da task (ex: experiments/classification).

    Returns
    -------
    list[dict]
        Leaderboard desta grid, ordenado por best_val_accuracy desc.
    """
    grid_ckpt_dir = checkpoint_dir / "grid_search" / model_name / grid_id
    grid_exp_dir = experiments_dir / "grid_search" / model_name / grid_id
    grid_ckpt_dir.mkdir(parents=True, exist_ok=True)
    grid_exp_dir.mkdir(parents=True, exist_ok=True)

    leaderboard = []

    for trial_id, trial in tuner.oracle.trials.items():
        if trial.score is None:
            # trial falhado ou incompleto — não entra no leaderboard.
            continue

        config = {
            "model_name": model_name,
            "grid_id": grid_id,
            "trial_id": trial_id,
            "hyperparameters": trial.hyperparameters.values,
            "best_val_accuracy": trial.score,
        }

        run_path = grid_exp_dir / f"{trial_id}.json"
        with open(run_path, "w") as f:
            json.dump(config, f, indent=2)

        leaderboard.append(config)

    leaderboard.sort(key=lambda r: r["best_val_accuracy"], reverse=True)

    leaderboard_path = grid_exp_dir / "leaderboard.json"
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)

    if leaderboard:
        best_model = tuner.get_best_models(num_models=1)[0]
        best_model.save(grid_ckpt_dir / "best.keras")

    print(f"\nBridge concluído: {len(leaderboard)} trials registados em {grid_exp_dir}")
    if leaderboard:
        print(f"Melhor trial: {leaderboard[0]}")
    else:
        print("Nenhum trial válido foi concluído.")

    return leaderboard