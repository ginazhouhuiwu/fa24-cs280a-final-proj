from __future__ import annotations

import viser
import torch

import numpy as np

from typing import Dict, List, TYPE_CHECKING

from nerfstudio.models.base_model import Model
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.cameras.camera_utils import quaternion_from_matrix, viewmatrix
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import scale_gradients_by_distance_squared
from nerfstudio.viewer.render_panel import CameraPath, Keyframe

from transformers import CLIPProcessor, CLIPModel

from PIL import Image


if TYPE_CHECKING:
    from viser import CameraFrustumHandle, GuiInputHandle, ViserServer


def keyframe_selector(
        prompt: str,
        train_cams: Dict[int, viser.CameraFrustumHandle],
        train_dataset: InputDataset
) -> List[int]:
        assert train_cams is not None

        instructions = prompt.lower().split(",")

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        images = [train_dataset[idx]["image"] for idx in train_cams]
        images = [Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8)) for img in images]

        text_inputs = processor(text=instructions, return_tensors="pt", padding=True, truncation=True).to("cuda")
        image_inputs = processor(images=images, return_tensors="pt").to("cuda")

        with torch.no_grad():
            text_features = model.get_text_features(**text_inputs)
            image_features = model.get_image_features(**image_inputs)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        similarity_scores = torch.matmul(text_features, image_features.T)

        selected_indices = []
        for text_idx in range(similarity_scores.shape[0]):
            best_score = -float("inf")
            best_idx = None

            for image_idx in range(similarity_scores.shape[1]):
                if image_idx in selected_indices:
                    continue
                
                score = similarity_scores[text_idx, image_idx].item()
                if score > best_score:
                    best_score = score
                    best_idx = image_idx
                    if score > best_score:
                        best_score = score
                        best_idx = image_idx

            selected_indices.append(best_idx)
        
        for idx in train_cams:
            train_cams[idx].visible = (idx in selected_indices)

        return selected_indices

def populate_autocam_tab(
    server: ViserServer,
    viewer_model: Model,
    camera_path: CameraPath,
    duration_number: GuiInputHandle,
    resolution: GuiInputHandle,
    train_cams: Dict[int, CameraFrustumHandle],
    train_dataset: InputDataset,
) -> None:    
    # shot_types = ['close up', 'wide shot', 'low angle', 'high angle']
    # movements = ['zoom in', 'pull out', 'focus', 'tilt', 'orbit', 'pan']

    # example_prompt = "Overview of the table, zoom in the mug, pan to the plate of cookies, pull in the lamb's tag around its neck that says DALLE, pan to the bear."
    # example_prompt = "Zoom in to the yellow rubber duck, show the bunny next to the green apple, show the box of twizzlers, pan to the chair."
    example_prompt = "Close up of the lego bulldozer, pan to the textbook that says computer vision, zoom in to waldo, pan to the typewriter."
    prompt = server.gui.add_text("Prompt", initial_value=example_prompt)
    
    keyframe_selector_button = server.gui.add_button("Select Keyframes")

    @keyframe_selector_button.on_click
    def _(event: viser.GuiEvent) -> None:
        selected_indices = keyframe_selector(prompt.value, train_cams, train_dataset)

        fov = event.client.camera.fov
        for i, idx in enumerate(selected_indices):
            camera_path.add_camera(
                keyframe=Keyframe(
                    position=train_cams[idx].position,
                    wxyz=train_cams[idx].wxyz,
                    override_fov_enabled=False,
                    override_fov_rad=fov,
                    override_time_enabled=False,
                    override_time_val=0.0,
                    aspect=resolution.value[0] / resolution.value[1],
                    override_transition_enabled=False,
                    override_transition_sec=None,
                ),
                keyframe_index=i,
            )
            duration_number.value = camera_path.compute_duration()
            camera_path.update_spline()


    click_position = np.array([0.0, 0.0, 0.0])

    center_folder = server.gui.add_folder("Choose Center")
    with center_folder:
        center_x_handle = server.gui.add_number(
            label="Select center point x",
            initial_value=0.1,
            hint="Choose center point x-coordinate to generate camera path around.",
        )

        center_y_handle = server.gui.add_number(
            label="Select center point y",
            initial_value=0.1,
            hint="Choose center point y-coordinate to generate camera path around.",
        )

        center_z_handle = server.gui.add_number(
            label="Select center point z",
            initial_value=-5.1,
            hint="Choose center point z-coordinate to generate camera path around.",
        )

        add_center_button = server.gui.add_button(
            "Choose Center",
            icon=viser.Icon.FOCUS_CENTERED,
            hint="Choose center point to generate camera path around.",
        )

        @add_center_button.on_click
        def _(event: viser.GuiEvent) -> None:
            nonlocal click_position

            click_position = np.array([center_x_handle.value,
                                       center_y_handle.value, 
                                       center_z_handle.value])
            
            server.scene.add_icosphere(
                f"/center/sphere",
                radius=0.1,
                color=(200, 10, 30),
                position=click_position,
            )

    camera_coords = []

    generate_folder = server.gui.add_folder("Generate")
    with generate_folder:
        num_cameras_handle = server.gui.add_number(
            label="Number of Cameras",
            initial_value=3,
            hint="Total number of cameras generated in path, placed equidistant from neighboring ones.",
        )

        num_points_along_path_handle = server.gui.add_number(
            label="Number Points Along Path",
            initial_value=100,
            hint="Number of points to optimize over along the camera path.",
        )

        radius_handle = server.gui.add_number(
            label="Radius",
            initial_value=4,
            hint="Radius of circular camera path.",
        )

        camera_height_handle = server.gui.add_number(
            label="Height",
            initial_value=2,
            hint="Height of cameras with respect to chosen origin.",
        )

        circular_camera_path_button = server.gui.add_button(
            "Generate Circular Camera Path",
            icon=viser.Icon.CAMERA,
            hint="Automatically generate a circular camera path around selected point.",
        )

        @circular_camera_path_button.on_click
        def _(event: viser.GuiEvent) -> None:
            nonlocal click_position, camera_coords

            num_cameras = num_cameras_handle.value
            radius = radius_handle.value
            camera_height = camera_height_handle.value
            num_points_along_path = num_points_along_path_handle.value

            for i in range(num_cameras):
                camera_coords.append(
                    click_position
                    + np.array(
                        [
                            radius * np.cos(2 * np.pi * i / num_cameras),
                            radius * np.sin(2 * np.pi * i / num_cameras),
                            camera_height,
                        ]
                    )
                )

            fov = event.client.camera.fov

            camera_path.loop = True
            for i, position in enumerate(camera_coords):
                view_direction = click_position - position
                matrix = viewmatrix(lookat=torch.from_numpy(view_direction).float(),
                                    up=torch.tensor([0.0, 0.0, 1.0]),
                                    pos=torch.from_numpy(position).float())
                wxyz = quaternion_from_matrix(matrix)

                camera_path.add_camera(
                    keyframe=Keyframe(
                        position=position,
                        wxyz=wxyz,
                        override_fov_enabled=False,
                        override_fov_rad=fov,
                        override_time_enabled=False,
                        override_time_val=0.0,
                        aspect=resolution.value[0] / resolution.value[1],
                        override_transition_enabled=False,
                        override_transition_sec=None,
                    ),
                    keyframe_index=i,
                )
                duration_number.value = camera_path.compute_duration()
                camera_path.update_spline()
            
        clear_keyframes_button = server.gui.add_button(
            "Clear Keyframes",
            icon=viser.Icon.TRASH,
            hint="Remove all keyframes from the render path.",
        )

        @clear_keyframes_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            client = server.get_clients()[event.client_id]
            with client.atomic(), client.gui.add_modal("Confirm") as modal:
                client.gui.add_markdown("Clear all keyframes?")
                confirm_button = client.gui.add_button("Yes", color="red", icon=viser.Icon.TRASH)
                exit_button = client.gui.add_button("Cancel")

                @confirm_button.on_click
                def _(_) -> None:
                    camera_path.reset()
                    modal.close()

                    duration_number.value = camera_path.compute_duration()

                @exit_button.on_click
                def _(_) -> None:
                    modal.close()

    optimize_folder = server.gui.add_folder("Optimize")
    with optimize_folder:
        num_opt_iterations_handle = server.gui.add_number(
            label="Number Iterations",
            initial_value=100,
            hint="Number of iterations for the optimization loop.",
        )

        optimize_button = server.gui.add_button(
            "Optimize Camera Path",
            icon=viser.Icon.CAMERA,
            hint="Optimizes camera path for object avoidance iteratively.",
        )

        distance_loss_enabled = server.gui.add_checkbox(
            "Distance Loss",
            initial_value=True,
        )

        @optimize_button.on_click
        def _(event: viser.GuiEvent) -> None:
            nonlocal camera_coords

            num_opt_iterations = num_opt_iterations_handle.value

            directions = [
                [1, 0, 0],
                [-1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
                [0, 0, -1],
            ]

            for _ in range(num_opt_iterations):
                for i, position in enumerate(camera_coords):
                    raylen = 2.0
                    origins = torch.tensor(np.tile(position, (6, 1)))
                    pixel_area = torch.ones_like(origins[..., 0:1])
                    camera_indices = torch.zeros_like(origins[..., 0:1]).int()
                    nears = torch.zeros_like(origins[..., 0:1])
                    fars = torch.ones_like(origins[..., 0:1]) * raylen
                    directions_norm = torch.ones_like(origins[..., 0:1])
                    viewer_model.training = False

                    bundle = RayBundle(
                        origins=origins,
                        directions=torch.tensor(directions),
                        pixel_area=pixel_area,
                        camera_indices=camera_indices,
                        nears=nears,
                        fars=fars,
                        metadata={"directions_norm": directions_norm},
                    ).to("cuda")

                    outputs = viewer_model.get_outputs(bundle)
                    distances = outputs["expected_depth"].detach().cpu().numpy()

                    distance_loss = -min(distances) if distance_loss_enabled else 0
                    loss = distance_loss
                    if loss > -0.4:
                        position = position - directions[np.argmin(distances)] * 1
                        view_direction = click_position - position
                        matrix = viewmatrix(lookat=torch.from_numpy(view_direction).float(),
                                            up=torch.tensor([0.0, 0.0, 1.0]),
                                            pos=torch.from_numpy(position).float())
                        wxyz = quaternion_from_matrix(matrix)

                        # backprop through the nerf as the gradient step, input is position
                        camera_path.add_camera(
                            keyframe=Keyframe(
                                position=position,
                                wxyz=wxyz,
                                override_fov_enabled=False,
                                override_fov_rad=event.client.camera.fov,
                                override_time_enabled=False,
                                override_time_val=0.0,
                                aspect=resolution.value[0] / resolution.value[1],
                                override_transition_enabled=False,
                                override_transition_sec=None,
                            ),
                            keyframe_index=i,
                        )

                        duration_number.value = camera_path.compute_duration()
                        camera_path.update_spline()

                        camera_coords[i] = position
                
