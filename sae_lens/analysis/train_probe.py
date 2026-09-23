import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import wandb
import wandb.sklearn
from datasets import DatasetDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from tqdm.auto import tqdm


class ProbeTrainer:
    def __init__(
        self,
        dataset: DatasetDict,
        flatten_T: Literal["batch", "hidden"],
        wandb_project: Optional[str] = None,
        model_class=LogisticRegression,  # type: ignore
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.dataset = dataset.with_format("np")
        self.probes = {}
        self.model_class = model_class
        self.flatten_T = flatten_T
        self.model_kwargs = model_kwargs if model_kwargs else {}

        self.project_name = wandb_project

    def _prep_data(  # type: ignore
        self, hook_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore
        """
        Prepare data for training and testing.

        Args:
            hook_name (str): Name of the hook to extract data from.
            flatten_T (str): 'batch' or 'hidden', determines how to flatten the T dimension.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """

        def extract_and_reshape(split: str) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
            X: np.ndarray = self.dataset[split][hook_name]  # type: ignore
            y: np.ndarray = self.dataset[split]["label"]  # type: ignore
            B, T, D = X.shape

            if self.flatten_T == "batch":
                X = X.reshape(B * T, D)
                y = np.repeat(y, T)
            else:  # flatten_T == "hidden"
                X = X.reshape(B, T * D)

            return X, y

        X_train, y_train = extract_and_reshape("train")
        X_test, y_test = extract_and_reshape("test")

        return X_train, X_test, y_train, y_test

    def _train_probe(self, hook_name: str):
        if self.project_name is not None:
            wandb.init(
                project=self.project_name,
                name=f"{hook_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
                reinit=True,
            )

        X_train, X_test, y_train, y_test = self._prep_data(hook_name)
        labels = self.dataset["train"].features["label"].names

        clf = self.model_class(**self.model_kwargs)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_probas = clf.predict_proba(X_test)
        print(f"Probe Metrics for {hook_name}:")
        print(classification_report(y_test, y_pred, target_names=labels))

        if self.project_name is not None:
            try:
                wandb.sklearn.plot_classifier(
                    clf,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_pred,
                    y_probas,
                    labels,
                    feature_names=None,
                    model_name=hook_name,
                    log_learning_curve=True,
                )
            except ValueError as e:  # calibration_curve can fail
                wandb.termwarn(f"Could not plot classifier for {hook_name}: {e}")

                wandb.sklearn.plot_roc(y_test, y_probas, labels)
                wandb.termlog("Logged roc curve.")

                wandb.sklearn.plot_precision_recall(y_test, y_probas, labels)
                wandb.termlog("Logged precision-recall curve.")
            finally:
                wandb.finish()

        self.probes[hook_name] = clf

    def train(self):
        hook_names = [x for x in self.dataset["train"].features.keys() if x != "label"]
        pbar = tqdm(hook_names, desc="Training probes")
        for hook_name in pbar:
            pbar.set_description(f"Training probe for {hook_name}")
            self._train_probe(hook_name)

    def save_probes(self, save_dir: Path | str):
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for hook_name, model in self.probes.items():
            save_path = save_dir / f"{hook_name}_{self.model_class.__name__}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Saved probe for {hook_name} to {save_path}")
