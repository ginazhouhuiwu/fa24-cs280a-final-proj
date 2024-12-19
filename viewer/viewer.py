from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import viser

from autocam.viewer.autocam_tab import populate_autocam_tab

from nerfstudio.configs import base_config as cfg
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.viewer.viewer import Viewer


if TYPE_CHECKING:
    from nerfstudio.engine.trainer import Trainer


class ViewerAutocam(Viewer):

    viewer_info: List[str]
    viser_server: viser.ViserServer

    def __init__(
        self,
        config: cfg.ViewerConfig,
        log_filename: Path,
        datapath: Path,
        pipeline: Pipeline,
        trainer: Optional[Trainer] = None,
        train_lock: Optional[threading.Lock] = None,
        share: bool = False,
    ):
        super().__init__(
            config=config,
            log_filename=log_filename,
            datapath=datapath,
            pipeline=pipeline,
            trainer=trainer,
            train_lock=train_lock,
            share=share,
        )

        self.ready = False

        self.init_scene(
            train_dataset=pipeline.datamanager.train_dataset,
            train_state="completed",
            eval_dataset=pipeline.datamanager.eval_dataset,
        )

        with self.tabs.add_tab("Autocam", viser.Icon.SETTINGS_AUTOMATION):
            populate_autocam_tab(
                server=self.viser_server, 
                viewer_model=self.pipeline.model,
                camera_path=self.camera_path,
                duration_number=self.duration_number,
                resolution=self.resolution,
                train_cams=self.camera_handles,
                train_dataset=pipeline.datamanager.train_dataset,
            )

        self.ready = True
        